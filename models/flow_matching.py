import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
from torchdiffeq import odeint

from models.estimator import Decoder

import josh_hacking

# modified from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/flow_matching.py
class CFMDecoder(torch.nn.Module):
    def __init__(self, noise_channels, cond_channels, hidden_channels, out_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels):
        super().__init__()
        self.noise_channels = noise_channels
        self.cond_channels = cond_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.gin_channels = gin_channels
        self.sigma_min = 1e-4

        self.estimator = Decoder(noise_channels, cond_channels, hidden_channels, out_channels, filter_channels, p_dropout, n_layers, n_heads, kernel_size, gin_channels)

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, c=None, solver=None, cfg_kwargs=None, prefix=None, postfix=None, p_mu=None, p_mask=None, s_mu=None, s_mask=None, blend_opts=None):
        """Forward diffusion with RePaint-style inpainting for prefix/postfix regions.

        When prefix and/or postfix mel spectrograms are provided, uses a RePaint-style
        approach: at each ODE step, the known (prefix/postfix) regions are re-injected
        at the appropriate noise level, forcing the generated region to be coherent with
        the boundary context at every scale.

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            c (torch.Tensor, optional): speaker embedding
                shape: (batch_size, gin_channels)
            solver: ODE solver method (used only when no prefix/postfix; ignored for RePaint loop)
            cfg_kwargs: used for cfg inference
            prefix (torch.Tensor, optional): prefix mel spectrogram (known audio before edit)
            postfix (torch.Tensor, optional): postfix mel spectrogram (known audio after edit)
            p_mu (torch.Tensor, optional): encoder output for prefix text
            p_mask (torch.Tensor, optional): mask for prefix
            s_mu (torch.Tensor, optional): encoder output for suffix text
            s_mask (torch.Tensor, optional): mask for suffix
            blend_opts (dict, optional): blending/RePaint options. Keys:
                - 'repaint_jumps' (bool): enable resampling jumps. Default: False.
                - 'jump_length' (int): how many steps to jump back. Default: 3.
                - 'jump_n_sample' (int): how many resample iterations per jump. Default: 3.

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        if blend_opts is None:
            blend_opts = {}

        has_inpaint = prefix is not None or postfix is not None

        if has_inpaint:
            return self._forward_repaint(
                mu, mask, n_timesteps, temperature, c, cfg_kwargs,
                prefix, postfix, p_mu, p_mask, s_mu, s_mask, blend_opts
            )
        else:
            return self._forward_standard(
                mu, mask, n_timesteps, temperature, c, solver, cfg_kwargs
            )

    def _forward_standard(self, mu, mask, n_timesteps, temperature, c, solver, cfg_kwargs):
        """Standard ODE-based forward pass (no inpainting)."""
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)

        if cfg_kwargs is None:
            estimator = functools.partial(self.estimator, mask=mask, mu=mu, c=c)
        else:
            estimator = functools.partial(self.cfg_wrapper, mask=mask, mu=mu, c=c, cfg_kwargs=cfg_kwargs)

        trajectory = odeint(estimator, z, t_span, method=solver, rtol=1e-5, atol=1e-5)
        return trajectory[-1]

    def _forward_repaint(self, mu, mask, n_timesteps, temperature, c, cfg_kwargs,
                         prefix, postfix, p_mu, p_mask, s_mu, s_mask, blend_opts):
        """RePaint-style forward pass with known-region re-injection at each step.

        Instead of freezing prefix/postfix with a zero-velocity mask, this approach:
        1. Starts everything as noise
        2. At each ODE step, computes velocity for ALL frames (model sees full context)
        3. After each step, replaces known regions with the correct interpolation
           between noise and original audio at the current time t
        4. Optionally performs resampling jumps for better boundary coherence

        Reference: RePaint (Lugmayr et al., 2022) - https://arxiv.org/abs/2201.09865
        Adapted from DDPM to Conditional Flow Matching.
        """
        repaint_jumps = blend_opts.get('repaint_jumps', False)
        jump_length = blend_opts.get('jump_length', 3)
        jump_n_sample = blend_opts.get('jump_n_sample', 3)

        # --- Build the full concatenated sequence ---
        # Generate noise for the editable region
        z_edit = torch.randn_like(mu) * temperature

        # Track prefix/suffix lengths for re-injection
        prefix_len = prefix.shape[-1] if prefix is not None else 0
        postfix_len = postfix.shape[-1] if postfix is not None else 0

        # Build full mu (encoder conditioning) by concatenating prefix_mu + edit_mu + suffix_mu
        full_mu = self._build_full_mu(mu, prefix, postfix, p_mu, s_mu)

        # Build full mask (all 1's — model sees everything in RePaint)
        full_mask = torch.ones(*mask.shape[:-1], prefix_len + mu.shape[-1] + postfix_len, device=mask.device)

        # Store the original clean mel for known regions
        # and the initial noise we'll use for re-injection
        original_prefix = prefix  # clean mel
        original_postfix = postfix  # clean mel

        # Generate noise for the known regions (same noise used throughout for consistency)
        noise_prefix = torch.randn_like(prefix) * temperature if prefix is not None else None
        noise_postfix = torch.randn_like(postfix) * temperature if postfix is not None else None

        # Build the full initial state: noise everywhere
        z_parts = []
        if prefix is not None:
            z_parts.append(noise_prefix)
        z_parts.append(z_edit)
        if postfix is not None:
            z_parts.append(noise_postfix)
        z = torch.cat(z_parts, dim=-1)

        # Build the estimator function (with full mask and full mu)
        if cfg_kwargs is None:
            estimator_fn = functools.partial(self.estimator, mask=full_mask, mu=full_mu, c=c)
        else:
            estimator_fn = functools.partial(self.cfg_wrapper, mask=full_mask, mu=full_mu, c=c, cfg_kwargs=cfg_kwargs)

        # --- RePaint stepping loop ---
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)

        # Build the schedule of steps, including resampling jumps if enabled
        if repaint_jumps and jump_length > 0 and jump_n_sample > 0:
            step_schedule = self._build_repaint_schedule(n_timesteps, jump_length, jump_n_sample)
        else:
            # Simple forward schedule: 0, 1, 2, ..., n_timesteps
            step_schedule = list(range(n_timesteps + 1))

        # Walk through the schedule
        current_step_idx = 0
        i = 0
        while i < len(step_schedule) - 1:
            current_step = step_schedule[i]
            next_step = step_schedule[i + 1]

            t_current = t_span[current_step]
            t_next = t_span[next_step]

            if next_step > current_step:
                # Forward step (denoising): t_current → t_next (t increases)
                dt = t_next - t_current
                velocity = estimator_fn(t_current, z)
                z = z + dt * velocity
            else:
                # Backward step (re-noising): t_current → t_next (t decreases, i.e., add noise)
                # In flow matching: x_t = (1-t)*x_0 + t*x_1
                # To go from t_current to t_next (where t_next < t_current),
                # we re-derive z at t_next from the current estimate of x_1
                # Current best estimate of clean signal: z at t_current ≈ x_1 (approximately)
                # But we need to add noise back. We use the flow matching interpolation:
                # z_t_next = (1 - t_next) * noise_original + t_next * z_current_estimate
                # where noise_original is the noise we started with
                # This is a simplification — we re-noise using the original noise
                z_edit_region = z[:, :, prefix_len:prefix_len + mu.shape[-1]]
                z_edit_renoised = (1 - t_next) * z_edit[: , :, :] + t_next * z_edit_region
                z[:, :, prefix_len:prefix_len + mu.shape[-1]] = z_edit_renoised

            # Re-inject known regions at the current noise level
            # In flow matching: x_t = (1-t)*x_0 + t*x_1
            # where x_0 is noise, x_1 is clean signal
            t_inject = t_next
            if prefix is not None:
                known_at_t = (1 - t_inject) * noise_prefix + t_inject * original_prefix
                z[:, :, :prefix_len] = known_at_t
            if postfix is not None:
                known_at_t = (1 - t_inject) * noise_postfix + t_inject * original_postfix
                z[:, :, -postfix_len:] = known_at_t

            i += 1

        return z

    def _build_full_mu(self, mu, prefix, postfix, p_mu, s_mu):
        """Build the full encoder conditioning tensor by concatenating prefix/edit/suffix mu."""
        parts = []

        if prefix is not None:
            if p_mu is None:
                parts.append(prefix * 0)  # zero conditioning for prefix
            else:
                if prefix.shape[-1] > p_mu.shape[-1]:
                    p_mu = torch.cat((torch.zeros(*p_mu.shape[:-1], prefix.shape[-1] - p_mu.shape[-1], device=p_mu.device), p_mu), dim=-1)
                else:
                    p_mu = p_mu[:, :, p_mu.shape[-1] - prefix.shape[-1]:]
                parts.append(p_mu)

        parts.append(mu)

        if postfix is not None:
            if s_mu is None:
                parts.append(postfix * 0)  # zero conditioning for postfix
            else:
                if postfix.shape[-1] > s_mu.shape[-1]:
                    s_mu = torch.cat((s_mu, torch.zeros(*s_mu.shape[:-1], postfix.shape[-1] - s_mu.shape[-1], device=s_mu.device)), dim=-1)
                else:
                    s_mu = s_mu[:, :, 0:postfix.shape[-1]]
                parts.append(s_mu)

        return torch.cat(parts, dim=-1)

    def _build_repaint_schedule(self, n_timesteps, jump_length, jump_n_sample):
        """Build a step schedule with resampling jumps.

        Every jump_length forward steps, we jump back by jump_length steps and
        redo them, repeating jump_n_sample times total (including the first pass).

        Example with n_timesteps=12, jump_length=3, jump_n_sample=2:
          Forward: 0→1→2→3, jump back: 3→0, redo: 0→1→2→3,
          Forward: 3→4→5→6, jump back: 6→3, redo: 3→4→5→6,
          Forward: 6→7→8→9, jump back: 9→6, redo: 6→7→8→9,
          Forward: 9→10→11→12

        The schedule as step indices: [0,1,2,3, 0,1,2,3, 3,4,5,6, 3,4,5,6, 6,7,8,9, 6,7,8,9, 9,10,11,12]

        Args:
            n_timesteps: total number of ODE steps
            jump_length: how many steps forward before jumping back
            jump_n_sample: total number of times to traverse each segment (including first)

        Returns:
            list of step indices (into t_span). Consecutive pairs define transitions.
        """
        schedule = [0]  # start at step 0
        current = 0

        while current < n_timesteps:
            segment_end = min(current + jump_length, n_timesteps)

            # First forward pass through this segment
            for step in range(current + 1, segment_end + 1):
                schedule.append(step)

            # Resampling: jump back and redo (jump_n_sample - 1 additional times)
            if segment_end < n_timesteps:
                for _ in range(jump_n_sample - 1):
                    # Jump back to segment start
                    schedule.append(current)
                    # Redo forward through segment
                    for step in range(current + 1, segment_end + 1):
                        schedule.append(step)

            current = segment_end

        return schedule

    # cfg inference
    def cfg_wrapper(self, t, x, mask, mu, c, cfg_kwargs):
        fake_speaker = cfg_kwargs['fake_speaker'].repeat(x.size(0), 1)
        fake_content = cfg_kwargs['fake_content'].repeat(x.size(0), 1, x.size(-1))
        cfg_strength = cfg_kwargs['cfg_strength']

        cond_output = self.estimator(t, x, mask, mu, c)
        uncond_output = self.estimator(t, x, mask, fake_content, fake_speaker)

        output = uncond_output + cfg_strength * (cond_output - uncond_output)
        return output

    def compute_loss(self, x1, mask, mu, c):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            c (torch.Tensor, optional): speaker condition.

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        # use cosine timestep scheduler from cosyvoice: https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow_matching.py
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        t = 1 - torch.cos(t * 0.5 * torch.pi)

        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        loss = F.mse_loss(self.estimator(t.squeeze(), y, mask, mu, c), u, reduction="sum") / (torch.sum(mask) * u.size(1))
        return loss, y
