"""
MAS Alignment App — Gradio UI for audio inpainting using StableTTS's
built-in Monotonic Alignment Search instead of Whisper for word boundaries.

Clone of streamlined_app.py with Whisper replaced by MAS alignment.
See MAS_IMPLEMENTATION_PLAN.md for design details.
"""

import os
os.environ['TMPDIR'] = './temps'  # avoid the system default temp folder not having access permissions

import re
import numpy as np

import matplotlib.pyplot as plt

import torch
import gradio as gr

from api import StableTTSAPI
from alignment import align_text_to_audio, compute_edit_regions

# Optional Whisper import — only used for the convenience "Transcribe" button
try:
    import whisper_timestamped as whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tts_model_path = './checkpoints/checkpoint_0.pt'
vocoder_model_path = './vocoders/pretrained/firefly-gan-base-generator.ckpt'
vocoder_type = 'ffgan'
whisper_model_name = "tiny.en"

model = StableTTSAPI(tts_model_path, vocoder_model_path, vocoder_type).to(device)


def transcribe(audio):
    """Transcribe audio using Whisper (optional convenience)."""
    if not WHISPER_AVAILABLE:
        raise gr.Error("Whisper is not installed. Please provide the transcript manually.")
    whisper_audio = whisper.load_audio(audio)
    whisper_model = whisper.load_model(whisper_model_name)
    transcription = whisper_model.transcribe(whisper_audio, word_timestamps=True)
    return transcription['text']


@torch.inference_mode()
def inference(original_text, edited_text, reference_audio, language, step, temperature, length_scale, solver, cfg):
    """
    Full MAS-alignment-based inpainting pipeline:
    1. Align original text to reference audio (get word boundaries via MAS)
    2. Diff original vs edited text (find prefix/suffix split points)
    3. Slice mel spectrogram at boundaries
    4. Generate only the edited portion using CFM decoder with prefix/suffix context
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

    # --- Step 1: Align original text to audio using MAS ---
    alignment = align_text_to_audio(model, original_text.strip(), reference_audio, language)
    word_boundaries = alignment['word_boundaries']
    mel = alignment['mel']  # (1, n_mels, mel_length) — already computed
    total_mel_frames = mel.size(-1)

    # --- Step 2: Compute edit regions ---
    edit_info = compute_edit_regions(
        original_text.strip(), edited_text.strip(), word_boundaries, total_mel_frames
    )

    if not edit_info['has_edit']:
        raise gr.Error("The text hasn't been changed.")

    prefix_end_frame = edit_info['prefix_end_frame']
    suffix_start_frame = edit_info['suffix_start_frame']
    prefix_text = edit_info['prefix_text']
    edited_portion = edit_info['edited_text']
    suffix_text = edit_info['suffix_text']

    # --- Step 3: Slice mel spectrogram at boundaries ---
    prefix_mel = mel[:, :, :prefix_end_frame] if prefix_end_frame > 0 else None
    suffix_mel = mel[:, :, suffix_start_frame:] if suffix_start_frame < total_mel_frames else None

    # --- Step 4: Generate using the existing inference pipeline ---
    # Pass mel tensors directly (api.py now accepts tensors via isinstance checks)
    # Also pass prefix_text/suffix_text for prosodic conditioning
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
    )

    max_val = torch.max(torch.abs(audio_output))
    if max_val > 1:
        audio_output = audio_output / max_val

    audio_result = (model.mel_config.sample_rate, (audio_output.cpu().squeeze(0).numpy() * 32767).astype(np.int16))
    mel_plot = plot_mel_spectrogram(mel_output.cpu().squeeze(0).numpy())

    # Generate alignment visualization (phoneme × mel frame correlation matrix)
    alignment_plot = plot_alignment_matrix(
        alignment['attn'], alignment['phonemes'], word_boundaries, edit_info
    )

    # Generate alignment debug info
    alignment_info = format_alignment_info(word_boundaries, edit_info)

    return audio_result, mel_plot, alignment_plot, alignment_info


def plot_mel_spectrogram(mel_spectrogram):
    """Plot mel spectrogram for display."""
    plt.close()  # prevent memory leak
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.imshow(mel_spectrogram, aspect='auto', origin='lower')
    plt.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def plot_alignment_matrix(attn, phonemes, word_boundaries, edit_info):
    """
    Plot the MAS alignment matrix showing phoneme-to-mel-frame correlation.

    X-axis: mel frames (time)
    Y-axis: phoneme tokens (interspersed)
    The visualization shows which mel frames correspond to which phonemes,
    with word boundaries and edit regions marked.

    Args:
        attn: alignment matrix tensor (1, 1, interspersed_text_length, mel_length)
        phonemes: list of phoneme characters from G2P
        word_boundaries: list of word boundary dicts
        edit_info: output from compute_edit_regions()
    """
    plt.close()  # prevent memory leak

    # Extract the alignment matrix as numpy
    # attn shape: (1, 1, text_tokens, mel_frames)
    attn_np = attn.squeeze(0).squeeze(0).cpu().numpy()  # (text_tokens, mel_frames)

    fig, ax = plt.subplots(figsize=(20, 8))

    # Plot the alignment matrix
    im = ax.imshow(attn_np, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')

    # Label axes
    ax.set_xlabel('Mel Frames (time →)', fontsize=12)
    ax.set_ylabel('Phoneme Tokens', fontsize=12)
    ax.set_title('MAS Alignment Matrix (phoneme × mel frame)', fontsize=14)

    # Add phoneme labels on Y-axis (show every Nth for readability)
    # Build interspersed phoneme labels: [_, p0, _, p1, _, p2, ...]
    interspersed_labels = []
    for i in range(attn_np.shape[0]):
        if i % 2 == 0:
            interspersed_labels.append('·')  # blank token
        else:
            phoneme_idx = i // 2
            if phoneme_idx < len(phonemes):
                interspersed_labels.append(phonemes[phoneme_idx])
            else:
                interspersed_labels.append('?')

    # Show phoneme labels (skip blanks for clarity if too many)
    n_tokens = len(interspersed_labels)
    if n_tokens <= 80:
        ax.set_yticks(range(n_tokens))
        ax.set_yticklabels(interspersed_labels, fontsize=7, fontfamily='monospace')
    else:
        # Show only non-blank tokens
        non_blank_indices = [i for i in range(n_tokens) if i % 2 == 1]
        non_blank_labels = [interspersed_labels[i] for i in non_blank_indices]
        ax.set_yticks(non_blank_indices)
        ax.set_yticklabels(non_blank_labels, fontsize=6, fontfamily='monospace')

    # Draw horizontal lines at word boundaries
    for wb in word_boundaries:
        # Word starts at interspersed token index 2 * phoneme_start
        token_y = 2 * wb.get('phoneme_start', 0) if 'phoneme_start' in wb else None
        if token_y is not None:
            ax.axhline(y=token_y - 0.5, color='cyan', linewidth=0.5, alpha=0.7)

    # Draw vertical lines at word frame boundaries and label words
    for wb in word_boundaries:
        # Vertical line at word start frame
        ax.axvline(x=wb['start_frame'], color='cyan', linewidth=0.5, alpha=0.5, linestyle='--')
        # Word label at the midpoint
        mid_frame = (wb['start_frame'] + wb['end_frame']) / 2
        ax.text(mid_frame, -2, wb['word'], ha='center', va='top', fontsize=8,
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='darkblue', alpha=0.7))

    # Highlight the edit region
    if edit_info['has_edit']:
        prefix_end = edit_info['prefix_end_frame']
        suffix_start = edit_info['suffix_start_frame']
        # Draw vertical bands for the edit region
        ax.axvspan(prefix_end, suffix_start, alpha=0.15, color='red', label='Edit region')
        ax.axvline(x=prefix_end, color='lime', linewidth=2, linestyle='-', alpha=0.8, label='Prefix end')
        ax.axvline(x=suffix_start, color='lime', linewidth=2, linestyle='-', alpha=0.8, label='Suffix start')

    ax.legend(loc='upper right', fontsize=9)

    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    fig.tight_layout()
    return fig


def format_alignment_info(word_boundaries, edit_info):
    """Format alignment and edit info as debug text."""
    lines = ["## Word Boundaries (MAS Alignment)", ""]
    lines.append("| Word | Start (s) | End (s) | Frames |")
    lines.append("|------|-----------|---------|--------|")
    for wb in word_boundaries:
        lines.append(
            f"| {wb['word']} | {wb['start_time']:.3f} | {wb['end_time']:.3f} | "
            f"{wb['start_frame']}-{wb['end_frame']} |"
        )

    lines.append("")
    lines.append("## Edit Detection")
    lines.append(f"- **Prefix text:** \"{edit_info['prefix_text']}\"")
    lines.append(f"- **Edited portion:** \"{edit_info['edited_text']}\"")
    lines.append(f"- **Suffix text:** \"{edit_info['suffix_text']}\"")
    lines.append(f"- **Prefix end frame:** {edit_info['prefix_end_frame']}")
    lines.append(f"- **Suffix start frame:** {edit_info['suffix_start_frame']}")

    return "\n".join(lines)


gui_title = 'StableTTS — MAS Alignment'
gui_description = """Audio inpainting using StableTTS's built-in Monotonic Alignment Search for word boundary detection.

**Workflow:** Provide reference audio + original transcript → edit the text → Generate.
The system aligns the original text to the audio using MAS, detects what changed, and regenerates only the edited portion."""

with gr.Blocks(theme=gr.themes.Base()) as demo:
    demo.load(None, None, js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', 'light');window.location.search = params.toString();}}")

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

            # Transcribe button (optional — only works if Whisper is installed)
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
                info="Your modified version — only the changed portion will be regenerated",
                lines=3,
            )

            language_gr = gr.Dropdown(
                label='Language',
                choices=list(model.supported_languages),
                value='english'
            )

            step_gr = gr.Slider(
                label='Step',
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
                choices=['euler', 'midpoint', 'dopri5', 'rk4', 'implicit_adams', 'bosh3', 'fehlberg2', 'adaptive_heun'],
                value='dopri5'
            )

            cfg_gr = gr.Slider(
                label='CFG',
                minimum=0,
                maximum=10,
                value=3,
            )

        with gr.Column():
            mel_gr = gr.Plot(label="Mel Spectrogram")
            audio_gr = gr.Audio(label="Synthesised Audio", autoplay=True)
            tts_button = gr.Button("\U0001F3A7 Generate", elem_id="send-btn", visible=True, variant="primary")

            alignment_plot_gr = gr.Plot(label="MAS Alignment Matrix")
            alignment_info_gr = gr.Markdown(label="Alignment Debug Info", value="")

    tts_button.click(
        inference,
        [original_text_gr, edited_text_gr, ref_audio_gr, language_gr, step_gr, temperature_gr, length_scale_gr, solver_gr, cfg_gr],
        outputs=[audio_gr, mel_gr, alignment_plot_gr, alignment_info_gr]
    )


if __name__ == '__main__':
    demo.queue()
    demo.launch(debug=True, show_api=True, share=True)
