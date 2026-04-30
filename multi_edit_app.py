"""
Multi-Edit App — Gradio UI for iterative audio inpainting using StableTTS
with JLDiff-based multi-edit detection.

Extends mas_app.py to handle multiple edit regions between original and edited
text. Each edit is applied right-to-left, with MAS re-alignment after each step.

Uses:
- multi_edit_diff.py for JLDiff-based edit detection
- alignment.py for MAS text-to-audio alignment + phoneme-level frame mapping
- api.py for TTS inference with RePaint-style inpainting
"""

import os
os.environ['TMPDIR'] = './temps'  # avoid the system default temp folder not having access permissions

import argparse
import tempfile
import numpy as np

import torch
import torchaudio
import gradio as gr

from api import StableTTSAPI
from alignment import align_text_to_audio, text_position_to_mel_frame
from multi_edit_diff import compute_edit_regions, EditRegion, MultiEditResult

# Optional Whisper import — only used for the convenience "Transcribe" button
try:
    import whisper_timestamped as whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

parser = argparse.ArgumentParser(description='StableTTS Multi-Edit Inpainting')
parser.add_argument('--cpu', action='store_true', help='Force CPU inference even if CUDA is available')
parser.add_argument('--tts_model', type=str, default='./checkpoints/checkpoint_josh.pt',
                    help='Path to the TTS model checkpoint to use for inference')
args, _ = parser.parse_known_args()

device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')

tts_model_path = args.tts_model
vocoder_model_path = './vocoders/pretrained/firefly-gan-base-generator.ckpt'
vocoder_type = 'ffgan'
whisper_model_name = "tiny.en"

model = StableTTSAPI(tts_model_path, vocoder_model_path, vocoder_type).to(device)


_whisper_model = None  # cached on first use; never reloaded


def transcribe(audio):
    """Transcribe audio using Whisper (optional convenience)."""
    global _whisper_model
    if not WHISPER_AVAILABLE:
        raise gr.Error("Whisper is not installed. Please provide the transcript manually.")
    if _whisper_model is None:
        _whisper_model = whisper.load_model(whisper_model_name)
    whisper_audio = whisper.load_audio(audio)
    transcription = _whisper_model.transcribe(whisper_audio, word_timestamps=True)
    return transcription['text']


@torch.inference_mode()
def inference(original_text, edited_text, reference_audio, language, step, temperature,
              length_scale, solver, cfg, enable_repaint_jumps, jump_length, jump_n_sample,
              save_trajectory, min_match):
    """
    Multi-edit inpainting pipeline:
    1. Compute edit regions using JLDiff (via multi_edit_diff)
    2. For each edit (right-to-left):
       a. Re-align current text to current audio (MAS)
       b. Map edit character positions to mel frames via piecewise G2P
       c. Run inpainting generation
       d. Update current audio and text
    3. Collect all intermediate results

    Edits are processed right-to-left so that character positions from
    multi_edit_diff remain valid: applying an edit at position N does not
    shift positions 0..N-1.
    """
    if reference_audio is None:
        raise gr.Error("Please provide reference audio.")

    if not original_text.strip():
        raise gr.Error("Please provide the original transcript text.")

    if not edited_text.strip():
        raise gr.Error("Please provide the edited text.")

    if language == 'chinese':
        original_text = original_text.replace(' ', '')
        edited_text = edited_text.replace(' ', '')

    # --- Step 1: Compute edit regions ---
    diff_result = compute_edit_regions(
        original_text.strip(),
        edited_text.strip(),
        min_match=int(min_match),
    )

    if len(diff_result.edits) == 0:
        raise gr.Error("The text hasn't been changed.")

    # Get the phonemizer for this language (needed for text_position_to_mel_frame)
    phonemizer = model.g2p_mapping.get(language)
    if phonemizer is None:
        raise gr.Error(f"Unsupported language: {language}")

    # --- Step 2: Iterative inpainting ---
    current_text = original_text.strip()
    current_audio_path = reference_audio
    intermediate_results = []

    for edit_idx, edit in enumerate(diff_result.edits):
        step_num = edit_idx + 1
        total_steps = len(diff_result.edits)

        # --- 2a: Align current text to current audio ---
        alignment = align_text_to_audio(model, current_text, current_audio_path, language)
        durations = alignment['durations']
        mel = alignment['mel']
        total_mel_frames = mel.size(-1)

        # --- 2b: Map edit character positions to mel frames ---
        # Since edits are right-to-left, edit.original_start/end are still
        # valid in current_text — previous edits were all at higher positions.
        prefix_end_frame = text_position_to_mel_frame(
            current_text, edit.original_start, durations, phonemizer
        )
        suffix_start_frame = text_position_to_mel_frame(
            current_text, edit.original_end, durations, phonemizer
        )

        # Extract text segments using character positions directly
        prefix_text = current_text[:edit.original_start].strip()
        suffix_text = current_text[edit.original_end:].strip()
        edited_portion = edit.new_text.strip()

        # --- 2c: Slice mel and run inpainting ---
        prefix_mel = mel[:, :, :prefix_end_frame] if prefix_end_frame > 0 else None
        suffix_mel = mel[:, :, suffix_start_frame:] if suffix_start_frame < total_mel_frames else None

        blend_opts = {}
        if enable_repaint_jumps:
            blend_opts['repaint_jumps'] = True
            blend_opts['jump_length'] = int(jump_length)
            blend_opts['jump_n_sample'] = int(jump_n_sample)
        if save_trajectory:
            blend_opts['save_trajectory'] = True

        audio_output, mel_output = model.inference(
            edited_portion,
            mel,  # reference audio as mel tensor (for speaker embedding)
            language,
            step,
            temperature,
            length_scale,
            solver,
            cfg,
            prefix=prefix_mel,
            postfix=suffix_mel,
            prefix_text=prefix_text if prefix_text else None,
            suffix_text=suffix_text if suffix_text else None,
            blend_opts=blend_opts if blend_opts else None,
        )

        max_val = torch.max(torch.abs(audio_output))
        if max_val > 1:
            audio_output = audio_output / max_val

        # --- 2d: Update current text and audio ---
        # Use the edit's character positions directly (valid because right-to-left)
        new_text = (current_text[:edit.original_start]
                    + edit.new_text
                    + current_text[edit.original_end:])

        # Save audio to temp file for next iteration's alignment
        os.makedirs('./temps', exist_ok=True)
        temp_audio_path = tempfile.mktemp(suffix='.wav', dir='./temps')
        # Ensure audio is 2D (channels, samples) for torchaudio.save
        audio_to_save = audio_output.cpu()
        if audio_to_save.dim() == 3:
            audio_to_save = audio_to_save.squeeze(0)  # (1, 1, samples) -> (1, samples)
        if audio_to_save.dim() == 1:
            audio_to_save = audio_to_save.unsqueeze(0)  # (samples,) -> (1, samples)
        torchaudio.save(temp_audio_path, audio_to_save, model.mel_config.sample_rate)

        # Collect intermediate result
        # Scale float [-1.0, 1.0] audio to signed 16-bit integer range for Gradio
        INT16_MAX = np.iinfo(np.int16).max  # 32767
        audio_numpy = (audio_output.cpu().squeeze(0).numpy() * INT16_MAX).astype(np.int16)
        intermediate_results.append({
            'step': step_num,
            'total_steps': total_steps,
            'edit': edit,
            'previous_text': current_text,
            'new_text': new_text,
            'audio': (model.mel_config.sample_rate, audio_numpy),
            'prefix_end_frame': prefix_end_frame,
            'suffix_start_frame': suffix_start_frame,
        })

        current_text = new_text
        current_audio_path = temp_audio_path

        # Free GPU memory between edit iterations so the next pass doesn't OOM.
        # inference_mode() suppresses gradient tracking but does NOT reclaim
        # the cached allocator pool; empty_cache() does.
        if device == 'cuda':
            torch.cuda.empty_cache()

    # --- Step 3: Build output ---
    return _build_output(diff_result, intermediate_results)


def _build_output(diff_result: MultiEditResult, intermediate_results: list):
    """Build the Gradio output from the multi-edit results.

    Returns:
        Tuple of (final_audio, results_state) for Gradio outputs.
    """
    if not intermediate_results:
        return None, []

    # Final audio is the last intermediate result
    final_audio = intermediate_results[-1]['audio']

    # Build results list for the dynamic render
    results_data = []
    for result in intermediate_results:
        edit = result['edit']
        results_data.append({
            'step': result['step'],
            'total_steps': result['total_steps'],
            'edit_type': edit.edit_type,
            'old_text': edit.old_text,
            'new_text': edit.new_text,
            'previous_text': result['previous_text'],
            'resulting_text': result['new_text'],
            'audio': result['audio'],
        })

    return final_audio, results_data


# --- Gradio UI ---

gui_title = 'StableTTS — Multi-Edit Inpainting'
gui_description = """Iterative audio inpainting using JLDiff-based multi-edit detection with MAS alignment.

**Workflow:** Provide reference audio + original transcript → edit the text (multiple changes allowed) → Generate.
The system detects all edit regions using JLDiff, then applies each edit right-to-left, re-aligning after each step.
Each edit uses RePaint-style inpainting for boundary-coherent audio generation."""

with gr.Blocks(theme=gr.themes.Base()) as demo:
    demo.load(None, None, js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', 'light');window.location.search = params.toString();}}")

    # State for dynamic rendering
    results_state = gr.State([])

    with gr.Row():
        with gr.Column():
            gr.Markdown(f"# {gui_title}")
            gr.Markdown(gui_description)

    with gr.Row():
        with gr.Column():
            ref_audio_gr = gr.Audio(
                label="Reference Audio",
                type="filepath"
            )

            # Transcribe button (optional)
            transcribe_btn = gr.Button(
                "Transcribe (Whisper)" if WHISPER_AVAILABLE else "Transcribe (Whisper not installed)",
                interactive=WHISPER_AVAILABLE
            )

            original_text_gr = gr.Textbox(
                label="Original Transcript",
                info="The exact text spoken in the reference audio",
                lines=3,
            )

            transcribe_btn.click(fn=transcribe, inputs=ref_audio_gr, outputs=original_text_gr)

            edited_text_gr = gr.Textbox(
                label="Edited Text",
                info="Your modified version — all changed portions will be regenerated iteratively",
                lines=3,
            )

            language_gr = gr.Dropdown(
                label='Language',
                choices=list(model.supported_languages),
                value='english'
            )

            min_match_gr = gr.Slider(
                label='Min Match (coalescence threshold)',
                info="Minimum match length in characters to preserve between edits. "
                     "Lower values keep edits separate; higher values merge nearby edits. "
                     "Default 2 merges edits separated by a single space.",
                minimum=1,
                maximum=20,
                value=2,
                step=1
            )

            step_gr = gr.Slider(
                label='ODE Steps',
                info="Number of ODE integration steps per edit. More = better quality but slower.",
                minimum=1,
                maximum=100,
                value=25,
                step=1
            )

            temperature_gr = gr.Slider(
                label='Temperature',
                minimum=0,
                maximum=2,
                value=1,
            )

            length_scale_gr = gr.Slider(
                label='Length Scale',
                minimum=0,
                maximum=5,
                value=1,
            )

            solver_gr = gr.Dropdown(
                label='ODE Solver',
                info="Solver for non-inpainting regions. RePaint always uses Euler stepping.",
                choices=['euler', 'midpoint', 'dopri5', 'rk4', 'implicit_adams', 'bosh3', 'fehlberg2', 'adaptive_heun'],
                value='dopri5'
            )

            cfg_gr = gr.Slider(
                label='CFG',
                minimum=0,
                maximum=10,
                value=3,
            )

            # --- RePaint Controls ---
            gr.Markdown("### RePaint Inpainting Options")
            gr.Markdown("*Known audio is always re-injected at each ODE step. "
                        "Enable resampling jumps for additional boundary harmonization.*")

            enable_repaint_jumps_gr = gr.Checkbox(
                label="Enable Resampling Jumps",
                info="At intervals, jump back in time and re-denoise.",
                value=False,
            )

            jump_length_gr = gr.Slider(
                label='Jump Length (steps)',
                info="How many ODE steps to advance before jumping back.",
                minimum=1,
                maximum=20,
                value=3,
                step=1
            )

            jump_n_sample_gr = gr.Slider(
                label='Jump Resample Count',
                info="How many times to resample at each jump point.",
                minimum=1,
                maximum=10,
                value=3,
                step=1
            )

            save_trajectory_gr = gr.Checkbox(
                label="Save Trajectory Debug Images",
                info="Saves mel spectrogram snapshots to ./trajectory_debug/",
                value=False,
            )

        with gr.Column():
            final_audio_gr = gr.Audio(label="Final Audio", autoplay=True)
            tts_button = gr.Button("\U0001F3A7 Generate All Edits", elem_id="send-btn", visible=True, variant="primary")

            # Dynamic results section
            @gr.render(inputs=results_state)
            def render_results(results_data):
                if not results_data:
                    gr.Markdown("*No results yet. Click Generate to process edits.*")
                    return

                gr.Markdown(f"## Edit Results ({len(results_data)} edit{'s' if len(results_data) != 1 else ''} applied)")

                for result in results_data:
                    step = result['step']
                    total = result['total_steps']
                    edit_type = result['edit_type']
                    old_text = result['old_text']
                    new_text = result['new_text']

                    # Build description
                    if edit_type == 'substitution':
                        desc = f"**{old_text!r}** → **{new_text!r}**"
                    elif edit_type == 'insertion':
                        desc = f"Insert **{new_text!r}**"
                    elif edit_type == 'deletion':
                        desc = f"Delete **{old_text!r}**"
                    else:
                        desc = f"{edit_type}: {old_text!r} → {new_text!r}"

                    with gr.Accordion(f"Step {step}/{total}: {edit_type.title()}", open=(step == total)):
                        gr.Markdown(f"**Edit:** {desc}")
                        gr.Markdown(f"**Text before:** {result['previous_text']}")
                        gr.Markdown(f"**Text after:** {result['resulting_text']}")
                        gr.Audio(value=result['audio'], label=f"Audio after step {step}")

    tts_button.click(
        inference,
        [original_text_gr, edited_text_gr, ref_audio_gr, language_gr, step_gr, temperature_gr,
         length_scale_gr, solver_gr, cfg_gr, enable_repaint_jumps_gr, jump_length_gr, jump_n_sample_gr,
         save_trajectory_gr, min_match_gr],
        outputs=[final_audio_gr, results_state]
    )


if __name__ == '__main__':
    demo.queue()
    demo.launch(debug=True, show_api=True, share=True)
