"""
batch_inpaint.py — Batch audio inpainting using StableTTS multi-edit pipeline.

Reads source audio + transcriptions from a "from" CSV, target transcriptions
from a "to" CSV, and produces inpainted audio files saved alongside an output CSV.

The inpainting logic mirrors multi_edit_app.py's iterative right-to-left edit
pipeline, with one additional filter: edits that do not add or remove any
alphabetic characters are skipped (e.g., punctuation-only changes have no
audible content to splice in).

Usage:
    python3 batch_inpaint.py \\
        --from_csv path/to/source.csv \\
        --to_csv   path/to/target.csv \\
        --out_csv  path/to/output.csv \\
        [--force] \\
        [--target_verse "MAT 1:1"] \\
        [--only_inpainted] \\
        [--language english] \\
        [--step 25] \\
        [--temperature 1.0] \\
        [--length_scale 1.0] \\
        [--solver dopri5] \\
        [--cfg 3.0] \\
        [--min_match 2] \\
        [--repaint_jumps] \\
        [--jump_length 3] \\
        [--jump_n_sample 3] \\
        [--cpu]

CSV schemas
-----------
from_csv  : verse_id, transcription, file_name
to_csv    : verse_id, transcription
out_csv   : verse_id, transcription, file_name   (written by this script)
"""

import os
os.environ['TMPDIR'] = './temps'  # keep temp files local

import re
import sys
import csv
import argparse
import tempfile
import warnings

import numpy as np
import torch
import torchaudio

from api import StableTTSAPI
from alignment import align_text_to_audio, text_position_to_mel_frame
from multi_edit_diff import compute_edit_regions, EditRegion


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class PunctuationOnlyEdits(Exception):
    """Raised when all detected edits are punctuation/whitespace-only.

    The caller should treat this the same as 'no edits' and copy the source
    audio through unchanged rather than skipping the verse entirely.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_alpha(text: str) -> bool:
    """Return True if *text* contains at least one alphabetic character."""
    return bool(re.search(r'[A-Za-z]', text))


def _edit_changes_alpha(edit: EditRegion) -> bool:
    """Return True if this edit adds or removes alphabetic characters.

    An edit is considered *alphabetically significant* when either:
    - the old text contains alphabetic characters (something spoken is removed), or
    - the new text contains alphabetic characters (something spoken is added).

    Edits that only touch punctuation, whitespace, or digits are skipped because
    there is no audible content to splice in or out.
    """
    return _has_alpha(edit.old_text) or _has_alpha(edit.new_text)


def _load_csv(path: str) -> list[dict]:
    """Load a CSV file and return a list of row dicts."""
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _save_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    """Write *rows* to *path* as a CSV with the given *fieldnames*."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _audio_out_dir(out_csv_path: str) -> str:
    """Return the audio output directory (sibling 'audio' folder of out_csv)."""
    return os.path.join(os.path.dirname(os.path.abspath(out_csv_path)), 'audio')


def _resolve_audio_path(file_name: str, csv_path: str) -> str:
    """Resolve *file_name* relative to the directory of *csv_path*."""
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    return os.path.join(csv_dir, file_name)


# ---------------------------------------------------------------------------
# Core inpainting logic (adapted from multi_edit_app.py)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_inpainting(
    model: StableTTSAPI,
    original_text: str,
    edited_text: str,
    reference_audio_path: str,
    language: str,
    step: int,
    temperature: float,
    length_scale: float,
    solver: str,
    cfg: float,
    min_match: int,
    device: str,
    blend_opts: dict | None = None,
) -> tuple[torch.Tensor, int]:
    """Run the multi-edit inpainting pipeline for a single verse.

    Returns:
        (audio_tensor, sample_rate) where audio_tensor is shape (1, samples).

    Raises:
        PunctuationOnlyEdits: if no alphabetically-significant edits are found
            (either no edits at all, or all edits are punctuation/whitespace-only).
            The caller should copy the source audio through unchanged.
        ValueError: if the language is unsupported or another logic error occurs.
    """
    # --- Compute edit regions ---
    diff_result = compute_edit_regions(
        original_text.strip(),
        edited_text.strip(),
        min_match=min_match,
    )

    if len(diff_result.edits) == 0:
        raise PunctuationOnlyEdits("No edits detected between original and target text.")

    # Filter out edits that don't change any alphabetic characters
    alpha_edits = [e for e in diff_result.edits if _edit_changes_alpha(e)]

    if len(alpha_edits) == 0:
        raise PunctuationOnlyEdits(
            "All detected edits are punctuation/whitespace-only — nothing to inpaint."
        )

    skipped = len(diff_result.edits) - len(alpha_edits)
    if skipped > 0:
        print(f"    [info] Skipping {skipped} punctuation-only edit(s).")

    # Replace the edit list with the filtered one (still in right-to-left order)
    diff_result.edits[:] = alpha_edits

    # Get phonemizer
    phonemizer = model.g2p_mapping.get(language)
    if phonemizer is None:
        raise ValueError(f"Unsupported language: {language!r}")

    current_text = original_text.strip()
    current_audio_path = reference_audio_path
    audio_output: torch.Tensor = torch.zeros(1)  # initialised; always overwritten in loop

    for edit_idx, edit in enumerate(diff_result.edits):
        step_num = edit_idx + 1
        total_steps = len(diff_result.edits)
        print(f"    [edit {step_num}/{total_steps}] {edit!r}")

        # Align current text to current audio
        alignment = align_text_to_audio(model, current_text, current_audio_path, language)
        durations = alignment['durations']
        mel = alignment['mel']
        total_mel_frames = mel.size(-1)

        # Map edit character positions to mel frames
        prefix_end_frame = text_position_to_mel_frame(
            current_text, edit.original_start, durations, phonemizer
        )
        suffix_start_frame = text_position_to_mel_frame(
            current_text, edit.original_end, durations, phonemizer
        )

        # Extract text segments
        prefix_text = current_text[:edit.original_start].strip()
        suffix_text = current_text[edit.original_end:].strip()
        edited_portion = edit.new_text.strip()

        # Slice mel for prefix/suffix conditioning
        prefix_mel = mel[:, :, :prefix_end_frame] if prefix_end_frame > 0 else None
        suffix_mel = mel[:, :, suffix_start_frame:] if suffix_start_frame < total_mel_frames else None

        audio_output, _mel_output = model.inference(
            edited_portion,
            mel,          # full mel as speaker reference
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
            blend_opts=blend_opts,
        )

        # Normalise
        max_val = torch.max(torch.abs(audio_output))
        if max_val > 1:
            audio_output = audio_output / max_val

        # Update text
        current_text = (
            current_text[:edit.original_start]
            + edit.new_text
            + current_text[edit.original_end:]
        )

        # Save intermediate audio for next iteration's alignment
        os.makedirs('./temps', exist_ok=True)
        temp_audio_path = tempfile.mktemp(suffix='.wav', dir='./temps')
        audio_to_save = audio_output.cpu()
        if audio_to_save.dim() == 3:
            audio_to_save = audio_to_save.squeeze(0)
        if audio_to_save.dim() == 1:
            audio_to_save = audio_to_save.unsqueeze(0)
        torchaudio.save(temp_audio_path, audio_to_save, model.mel_config.sample_rate)

        current_audio_path = temp_audio_path

        if device == 'cuda':
            torch.cuda.empty_cache()

    # Return the final audio
    final_audio = audio_output.cpu()
    if final_audio.dim() == 3:
        final_audio = final_audio.squeeze(0)
    if final_audio.dim() == 1:
        final_audio = final_audio.unsqueeze(0)

    return final_audio, model.mel_config.sample_rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch audio inpainting using StableTTS multi-edit pipeline.'
    )
    parser.add_argument(
        '--from_csv', required=True,
        help='Input CSV with columns: verse_id, transcription, file_name'
    )
    parser.add_argument(
        '--to_csv', required=True,
        help='Target CSV with columns: verse_id, transcription'
    )
    parser.add_argument(
        '--out_csv', required=True,
        help='Output CSV path. Audio files are saved in an "audio" folder next to it.'
    )
    parser.add_argument(
        '--out_format', default='wav', choices=['wav', 'mp3', 'flac', 'ogg'],
        help='Output audio format (default: wav). '
             'The inpainting pipeline always produces WAV internally; '
             'this controls the final saved format.'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Re-generate audio even if the output file already exists.'
    )
    parser.add_argument(
        '--target_verse', default=None,
        help='Process only this verse ID (e.g. "MAT 1:1"). Implies --force.'
    )
    # TTS parameters
    parser.add_argument('--language', default='english',
                        choices=['english', 'chinese', 'japanese'])
    parser.add_argument('--step', type=int, default=25,
                        help='ODE integration steps (default: 25)')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--length_scale', type=float, default=1.0)
    parser.add_argument('--solver', default='dopri5',
                        choices=['euler', 'midpoint', 'dopri5', 'rk4',
                                 'implicit_adams', 'bosh3', 'fehlberg2', 'adaptive_heun'])
    parser.add_argument('--cfg', type=float, default=3.0,
                        help='Classifier-free guidance scale (default: 3.0)')
    parser.add_argument('--min_match', type=int, default=2,
                        help='JLDiff coalescence threshold (default: 2)')
    # RePaint resampling-jump options (passed through as blend_opts)
    parser.add_argument('--repaint_jumps', action='store_true',
                        help='Enable RePaint-style resampling jumps for better boundary coherence.')
    parser.add_argument('--jump_length', type=int, default=3,
                        help='Number of forward steps before each resampling jump (default: 3).')
    parser.add_argument('--jump_n_sample', type=int, default=3,
                        help='Total traversals per segment including first pass (default: 3).')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU inference even if CUDA is available')
    parser.add_argument('--only_inpainted', action='store_true',
                        help='Only print output for verses that actually undergo inpainting '
                             '(suppress headers for identical-text, punctuation-only, and '
                             'already-skipped verses).')
    # Model paths
    parser.add_argument('--tts_model', required=True,
                        help='Path to TTS model checkpoint (required)')
    parser.add_argument('--vocoder_model',
                        default='./vocoders/pretrained/firefly-gan-base-generator.ckpt',
                        help='Path to vocoder checkpoint')
    parser.add_argument('--vocoder_type', default='ffgan',
                        choices=['ffgan', 'vocos'])
    return parser.parse_args()


def main():
    args = parse_args()

    # --target_verse implies --force
    force = args.force or (args.target_verse is not None)

    # -----------------------------------------------------------------------
    # Load CSVs
    # -----------------------------------------------------------------------
    print(f"Loading from_csv: {args.from_csv}")
    from_rows = _load_csv(args.from_csv)
    print(f"Loading to_csv:   {args.to_csv}")
    to_rows = _load_csv(args.to_csv)

    # Validate required columns
    required_from = {'verse_id', 'transcription', 'file_name'}
    required_to = {'verse_id', 'transcription'}
    if from_rows:
        missing_from = required_from - set(from_rows[0].keys())
        if missing_from:
            sys.exit(f"ERROR: from_csv is missing columns: {missing_from}")
    if to_rows:
        missing_to = required_to - set(to_rows[0].keys())
        if missing_to:
            sys.exit(f"ERROR: to_csv is missing columns: {missing_to}")

    # Build lookup dicts keyed by verse_id
    from_by_id: dict[str, dict] = {r['verse_id']: r for r in from_rows}
    to_by_id:   dict[str, dict] = {r['verse_id']: r for r in to_rows}

    # Warn about mismatches
    from_ids = set(from_by_id.keys())
    to_ids   = set(to_by_id.keys())
    for vid in sorted(from_ids - to_ids):
        print(f"WARNING: verse {vid!r} is in from_csv but not in to_csv — skipping.")
    for vid in sorted(to_ids - from_ids):
        print(f"WARNING: verse {vid!r} is in to_csv but not in from_csv — skipping.")

    # Verses to process: intersection, optionally filtered to target_verse
    common_ids = sorted(from_ids & to_ids)
    if args.target_verse is not None:
        if args.target_verse not in common_ids:
            sys.exit(
                f"ERROR: target_verse {args.target_verse!r} not found in both CSVs."
            )
        verses_to_process = [args.target_verse]
    else:
        verses_to_process = common_ids

    if not verses_to_process:
        sys.exit("ERROR: No verses to process (empty intersection of from_csv and to_csv).")

    # -----------------------------------------------------------------------
    # Prepare output paths
    # -----------------------------------------------------------------------
    audio_out_dir = _audio_out_dir(args.out_csv)
    os.makedirs(audio_out_dir, exist_ok=True)

    # Load existing out_csv if it exists (so we can append / update rows)
    out_csv_exists = os.path.isfile(args.out_csv)
    if out_csv_exists and args.target_verse is not None:
        # When targeting a single verse, load existing rows so we don't lose others
        existing_out_rows = _load_csv(args.out_csv)
        out_by_id: dict[str, dict] = {r['verse_id']: r for r in existing_out_rows}
    else:
        out_by_id = {}

    # -----------------------------------------------------------------------
    # Load model (deferred until we know there's work to do)
    # -----------------------------------------------------------------------
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading TTS model from {args.tts_model} ...")
    model = StableTTSAPI(args.tts_model, args.vocoder_model, args.vocoder_type).to(device)
    print("Model loaded.")

    # -----------------------------------------------------------------------
    # Process verses
    # -----------------------------------------------------------------------
    for verse_id in verses_to_process:
        from_row = from_by_id[verse_id]
        to_row   = to_by_id[verse_id]

        original_text = from_row['transcription']
        edited_text   = to_row['transcription']

        # Determine output audio filename
        # Use a filesystem-safe version of the verse_id as the filename
        safe_id = re.sub(r'[^\w\-.]', '_', verse_id)
        out_audio_filename = f"{safe_id}.{args.out_format}"
        out_audio_path = os.path.join(audio_out_dir, out_audio_filename)
        # Relative path from out_csv directory
        out_audio_rel = os.path.join('audio', out_audio_filename)

        def _verse_header():
            print(f"\n{'='*60}")
            print(f"Verse: {verse_id}")
            print(f"  Original : {original_text!r}")
            print(f"  Target   : {edited_text!r}")
            print(f"  Out audio: {out_audio_path}")

        # Skip if output already exists and not forcing
        if os.path.isfile(out_audio_path) and not force:
            if not args.only_inpainted:
                _verse_header()
                print(f"  [skip] Output already exists. Use --force to regenerate.")
            # Still record in output CSV
            out_by_id[verse_id] = {
                'verse_id': verse_id,
                'transcription': edited_text,
                'file_name': out_audio_rel,
            }
            continue

        # Resolve source audio path relative to from_csv
        src_audio_path = _resolve_audio_path(from_row['file_name'], args.from_csv)
        if not os.path.isfile(src_audio_path):
            _verse_header()
            print(f"  [error] Source audio not found: {src_audio_path} — skipping.")
            continue

        # Check if texts are identical
        if original_text.strip() == edited_text.strip():
            if not args.only_inpainted:
                _verse_header()
                print(f"  [skip] Original and target text are identical — transcoding source audio.")
            # Decode the source (any format) and re-save in the requested output format
            try:
                src_waveform, src_sr = torchaudio.load(src_audio_path)
                torchaudio.save(out_audio_path, src_waveform, src_sr)
            except Exception as exc:
                if args.only_inpainted:
                    _verse_header()
                print(f"  [error] Could not transcode source audio: {exc} — skipping.")
                continue
            out_by_id[verse_id] = {
                'verse_id': verse_id,
                'transcription': edited_text,
                'file_name': out_audio_rel,
            }
            continue

        # Build blend_opts from repaint args
        blend_opts = {
            'repaint_jumps': args.repaint_jumps,
            'jump_length': args.jump_length,
            'jump_n_sample': args.jump_n_sample,
        }

        # Run inpainting (header is printed only if inpainting actually succeeds,
        # so that --only_inpainted suppresses passthrough verses)
        try:
            final_audio, sample_rate = run_inpainting(
                model=model,
                original_text=original_text,
                edited_text=edited_text,
                reference_audio_path=src_audio_path,
                language=args.language,
                step=args.step,
                temperature=args.temperature,
                length_scale=args.length_scale,
                solver=args.solver,
                cfg=args.cfg,
                min_match=args.min_match,
                device=device,
                blend_opts=blend_opts,
            )
        except PunctuationOnlyEdits as exc:
            if not args.only_inpainted:
                _verse_header()
                print(f"  [info] {exc}")
                print(f"  [info] Transcoding source audio unchanged (punctuation-only diff).")
            try:
                src_waveform, src_sr = torchaudio.load(src_audio_path)
                torchaudio.save(out_audio_path, src_waveform, src_sr)
            except Exception as copy_exc:
                _verse_header()
                print(f"  [error] Could not transcode source audio: {copy_exc} — skipping.")
                continue
            out_by_id[verse_id] = {
                'verse_id': verse_id,
                'transcription': edited_text,
                'file_name': out_audio_rel,
            }
            continue
        except ValueError as exc:
            _verse_header()
            print(f"  [error] {exc} — skipping.")
            continue
        except Exception as exc:
            _verse_header()
            print(f"  [error] Unexpected error: {exc} — skipping.")
            import traceback
            traceback.print_exc()
            continue

        # Inpainting succeeded — always show the header and result
        _verse_header()
        # Save audio
        torchaudio.save(out_audio_path, final_audio, sample_rate)
        print(f"  [done] Saved to {out_audio_path}")

        out_by_id[verse_id] = {
            'verse_id': verse_id,
            'transcription': edited_text,
            'file_name': out_audio_rel,
        }

    # -----------------------------------------------------------------------
    # Write output CSV
    # -----------------------------------------------------------------------
    # When targeting a single verse, preserve all other rows from the existing CSV.
    # When running over everything, write only the processed verses (in to_csv order).
    if args.target_verse is not None:
        # Merge: start from all to_csv rows, overlay with what we have in out_by_id
        out_rows = []
        for row in to_rows:
            vid = row['verse_id']
            if vid in out_by_id:
                out_rows.append(out_by_id[vid])
            else:
                # Preserve any pre-existing row from the old out_csv if available
                # (already loaded into out_by_id above)
                pass  # not in out_by_id means it was never processed
        # Also include any rows that were in the old out_csv but not in to_csv
        to_ids_set = {r['verse_id'] for r in to_rows}
        for vid, row in out_by_id.items():
            if vid not in to_ids_set:
                out_rows.append(row)
    else:
        # Full run: write all processed verses in to_csv order
        out_rows = []
        for row in to_rows:
            vid = row['verse_id']
            if vid in out_by_id:
                out_rows.append(out_by_id[vid])

    _save_csv(args.out_csv, out_rows, fieldnames=['verse_id', 'transcription', 'file_name'])
    print(f"\nOutput CSV written: {args.out_csv} ({len(out_rows)} rows)")


if __name__ == '__main__':
    main()
