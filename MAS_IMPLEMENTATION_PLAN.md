# MAS Alignment Implementation Plan

## Overview

Replace Whisper-based word boundary detection in the audio inpainting pipeline with StableTTS's built-in Monotonic Alignment Search (MAS). This eliminates the Whisper dependency for alignment while producing more accurate boundaries using the fine-tuned model's own learned representations.

## Files

| File | Action | Purpose |
|------|--------|---------|
| `alignment.py` | CREATE | Alignment module — MAS-based text-to-audio alignment + edit region computation |
| `mas_app.py` | CREATE | Gradio app (clone of `streamlined_app.py`) using MAS alignment instead of Whisper |
| `api.py` | MODIFY | Small change to `inference()` to accept mel tensors in addition to file paths |

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     mas_app.py (Gradio UI)                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User provides: reference_audio + original_text + edited_text    │
│                                                                  │
│  1. alignment.align_text_to_audio(api, text, audio_path)         │
│     → word boundaries (frame-level)                              │
│                                                                  │
│  2. alignment.compute_edit_regions(original, edited, boundaries) │
│     → prefix_end_frame, suffix_start_frame, text segments        │
│                                                                  │
│  3. Slice original mel at frame boundaries → prefix_mel,         │
│     suffix_mel (tensor slicing, no pydub/temp files)             │
│                                                                  │
│  4. api.inference(edited_text, ref_audio, ...,                   │
│       prefix=prefix_mel, postfix=suffix_mel,                     │
│       prefix_text=prefix_text, suffix_text=suffix_text)          │
│     → generated audio with inpainted region                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Module Details

### alignment.py

#### `align_text_to_audio(api: StableTTSAPI, text: str, audio_path: str, language: str = 'english') -> dict`

Uses the StableTTS model's encoder + MAS to align text to audio. Replicates the alignment computation from the training forward pass (`models/model.py` lines 211-220).

**Accesses from `api`:**
- `api.tts_model` — the StableTTS model (encoder, ref_encoder)
- `api.mel_extractor` — LogMelSpectrogram instance
- `api.mel_config` — MelConfig (hop_length, sample_rate)
- `api.g2p_mapping` — G2P functions dict

**Steps:**
1. Run text through G2P (`english_to_ipa2()` etc.) → phoneme list
2. Find word boundaries by locating space characters in phoneme list
3. Convert phonemes to token IDs, intersperse with blanks
4. Load audio → mel spectrogram
5. Run ref_encoder on mel → speaker embedding `c`
6. Run text encoder conditioned on `c` → `mu_x`
7. Compute log-likelihood matrix between `mu_x` and mel
8. Run MAS → alignment matrix `attn`
9. Extract per-token durations from `attn`
10. Map token durations → word-level frame boundaries (accounting for intersperse indexing)

**Returns:**
```python
{
    'attn': alignment_matrix,          # (1, 1, text_length, mel_length)
    'durations': per_token_durations,  # (interspersed_text_length,)
    'word_boundaries': [               # list of dicts
        {'word': str, 'start_frame': int, 'end_frame': int,
         'start_time': float, 'end_time': float},
        ...
    ],
    'mel': mel_spectrogram,            # (1, n_mels, mel_length) — reusable
    'mel_config': mel_config,
}
```

#### Word Boundary Detection (Simple Approach)

Run full text through `english_to_ipa2()` as a single call. The function preserves spaces between words. Locate space characters in the resulting phoneme list to identify word boundaries.

**Intersperse mapping:**
- Original phoneme at index `i` → interspersed token at index `2*i + 1`
- Blank tokens at even indices also receive durations from MAS
- Word boundary = sum of durations for all tokens (phonemes + surrounding blanks) belonging to that word

#### `compute_edit_regions(original_text: str, edited_text: str, word_boundaries: list) -> dict`

Finds the longest common prefix and suffix at the word level (case-sensitive comparison), then maps to mel frame positions.

**Algorithm:**
```python
original_words = original_text.split()
edited_words = edited_text.split()

# Walk forward to find common prefix length
prefix_word_count = 0
for o, e in zip(original_words, edited_words):
    if o == e:
        prefix_word_count += 1
    else:
        break

# Walk backward to find common suffix length
suffix_word_count = 0
for o, e in zip(reversed(original_words), reversed(edited_words)):
    if o == e:
        suffix_word_count += 1
    else:
        break

# Guard: ensure prefix + suffix don't overlap
# Map word counts to frame positions using word_boundaries
```

**Returns:**
```python
{
    'prefix_end_frame': int,       # frame where unchanged prefix ends
    'suffix_start_frame': int,     # frame where unchanged suffix begins
    'prefix_text': str,            # unchanged prefix text (for encoder conditioning)
    'edited_text': str,            # the changed portion of text
    'suffix_text': str,            # unchanged suffix text (for encoder conditioning)
    'has_edit': bool,              # False if texts are identical
}
```

**Edge cases:**
- No edit (texts identical) → `has_edit = False`, raise user-friendly error in app
- Edit at very start (no prefix) → `prefix_end_frame = 0`, no prefix mel
- Edit at very end (no suffix) → `suffix_start_frame = total_frames`, no suffix mel
- Entire text replaced → no prefix or suffix, full regeneration

### api.py Modification

Add `isinstance` checks so prefix/postfix/ref_audio parameters accept either file paths (str) or pre-computed mel tensors (torch.Tensor):

```python
# In inference():
if prefix is not None and not isinstance(prefix, torch.Tensor):
    prefix = load_and_resample_audio(prefix, self.mel_config.sample_rate).to(device)
    prefix = self.mel_extractor(prefix)

if postfix is not None and not isinstance(postfix, torch.Tensor):
    postfix = load_and_resample_audio(postfix, self.mel_config.sample_rate).to(device)
    postfix = self.mel_extractor(postfix)

if ref_audio is not None and not isinstance(ref_audio, torch.Tensor):
    ref_audio = load_and_resample_audio(ref_audio, self.mel_config.sample_rate).to(device)
    ref_audio = self.mel_extractor(ref_audio)
```

This is backward-compatible — existing code passing file paths continues to work.

### mas_app.py

Clone of `streamlined_app.py` with these changes:

**Removed:**
- Hard dependency on `whisper_timestamped`
- `run_wisper()`, `determine_prefix_text()`, `determine_postfix_text()` functions
- `pydub` audio cutting (no temp WAV files)

**Added:**
- Import `alignment.py` module
- Optional Whisper import (try/except) for transcription convenience
- MAS alignment + edit region computation in the inference function
- Mel tensor slicing for prefix/postfix (direct tensor operations)
- Alignment visualization output (debug info)
- Fade/crossfade at mask edges (future enhancement, noted in code)

**UI Flow:**
1. User uploads reference audio
2. User provides original transcript (via optional Transcribe button or manual entry)
3. User edits the text
4. User clicks "Generate"
5. Behind the scenes: align → diff → slice mel → inference → output audio + mel plot + alignment debug

**Whisper handling:**
```python
try:
    import whisper_timestamped as whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
```
Transcribe button is only shown/enabled if Whisper is available.

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pass `StableTTSAPI` into alignment functions | Avoids duplicate model in memory; alignment reuses existing mel_extractor, g2p_mapping, etc. |
| Simple space-based word boundary detection | `english_to_ipa2()` preserves spaces; simpler than per-word G2P processing; sufficient for English |
| Longest common prefix/suffix for edit detection | Matches existing prefix/postfix architecture; composable for future multi-edit loops |
| Case-sensitive text comparison | Avoids edge cases; capitalization can affect pronunciation |
| Mel tensor slicing instead of pydub cutting | More precise (frame-level); no temp files; mel already computed for alignment |
| `isinstance` checks for tensor/path flexibility | Minimal change to api.py; backward compatible |
| Prefix/suffix text always passed to inference | Essential for prosodic continuity — without it, generated audio sounds like sentence start |
| Whisper optional (not hard dependency) | User can manually provide transcript; graceful degradation |
| Fade/crossfade at mask edges | Planned for smoother blending at edit boundaries |

## Future Enhancements

- **Multi-edit support:** Loop over individual edits, applying each sequentially (reuses single-edit pipeline)
- **Fade/crossfade at boundaries:** Linear ramp over N frames at prefix→edit and edit→suffix transitions
- **Alignment visualization:** Plot alignment matrix with word boundaries overlaid on mel spectrogram
- **Multi-language support:** Currently English-focused; extend word boundary detection for other languages
- **Alignment caching:** If the same audio is used multiple times, cache the alignment result

## Dependencies

**Required (already in project):**
- `torch`, `torchaudio`
- `monotonic_align` (project module)
- `gradio`

**Optional:**
- `whisper_timestamped` — only for transcription convenience button

**Removed from pipeline:**
- `pydub` — no longer needed for audio cutting (mel tensor slicing instead)

## Testing Strategy

1. **Alignment quality:** Run `align_text_to_audio()` on known audio, verify word boundaries are reasonable
2. **Round-trip test:** Generate audio with known text, align it, verify boundaries match expected durations
3. **Edit detection:** Test `compute_edit_regions()` with various edit patterns (replacement, insertion, deletion, no-edit)
4. **End-to-end:** Full pipeline test — provide audio + text, make an edit, verify output audio sounds correct
