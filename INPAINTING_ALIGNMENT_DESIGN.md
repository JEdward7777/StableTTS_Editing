# StableTTS Audio Inpainting via Internal Alignment

## Overview

This document describes how to use StableTTS's built-in Monotonic Alignment Search (MAS) to determine word boundaries in audio, enabling a complete end-to-end inpainting pipeline for translation editing. The key insight is that the same alignment mechanism used during training can be repurposed at inference time to align **existing recorded audio** against its known transcript, producing a phoneme-to-mel-frame mapping that tells us exactly where each word lives in the audio.

## Problem Statement

Translators have already recorded audio for their work. They want to make textual edits (word insertions, deletions, replacements) and have only the changed portions of the audio regenerated while preserving the rest. The challenge is: **given text and its corresponding audio, determine which mel-spectrogram frames correspond to which words.**

## Architecture: How StableTTS Already Solves This

### The Training Forward Pass Contains the Answer

During training, StableTTS's `forward()` method in `models/model.py` (lines 176-240) already aligns arbitrary (text, audio) pairs using Monotonic Alignment Search:

```python
# From models/model.py forward() - this is the key code
with torch.no_grad():
    s_p_sq_r = torch.ones_like(mu_x)  # [b, d, t]
    neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - torch.zeros_like(mu_x), [1], keepdim=True)
    neg_cent2 = torch.einsum("bdt, bds -> bts", -0.5 * (y**2), s_p_sq_r)
    neg_cent3 = torch.einsum("bdt, bds -> bts", y, (mu_x * s_p_sq_r))
    neg_cent4 = torch.sum(-0.5 * (mu_x**2) * s_p_sq_r, [1], keepdim=True)
    neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()
```

This computes a log-likelihood matrix between each text token position and each mel frame, then finds the optimal monotonic path through it. The result `attn` is a binary matrix of shape `(batch, 1, text_length, mel_length)` where each text token maps to a contiguous span of mel frames.

**This works on ANY (text, audio) pair** — not just generated audio. It works on real recorded audio as long as the model has been fine-tuned on the speaker (so that `mu_x` — the encoder's predicted mel statistics — are calibrated to the speaker's actual mel spectrograms).

### The Alignment Matrix IS the Word Boundary Map

Given the alignment matrix `attn`:
- `attn.sum(dim=3)` → duration of each phoneme token in mel frames
- `torch.cumsum(attn.sum(dim=3), dim=2)` → cumulative frame boundaries for each token
- Since we know the mapping from words → phoneme tokens (from the G2P pipeline), we can reconstruct word → frame boundaries

## Implementation Plan

### Step 1: Create `align_text_to_audio()` Function

Create a new file `alignment.py` in the project root. This function takes a trained StableTTS model, text, and audio, and returns the alignment.

**Location:** `/home/lansford/work2/Mission_Mutual/StableTTS_Editing/alignment.py`

```python
import math
import torch
import monotonic_align
from config import MelConfig
from models.model import StableTTS
from text import cleaned_text_to_sequence
from text.english import english_to_ipa2
from datas.dataset import intersperse
from utils.audio import LogMelSpectrogram, load_and_resample_audio
from utils.mask import sequence_mask
from dataclasses import asdict


def align_text_to_audio(model: StableTTS, text: str, audio_path: str, language: str = 'english',
                        mel_config: MelConfig = None, device: str = 'cuda'):
    """
    Use the trained StableTTS model's encoder + MAS to align text to audio.

    This replicates the alignment computation from the training forward pass,
    but is designed to be called at inference time on arbitrary (text, audio) pairs.

    Args:
        model: A trained/fine-tuned StableTTS model
        text: The transcript text corresponding to the audio
        audio_path: Path to the audio file
        language: Language for G2P conversion
        mel_config: Mel spectrogram configuration (uses default if None)
        device: Device to run on

    Returns:
        dict with:
            - 'attn': alignment matrix (1, 1, text_length, mel_length)
            - 'durations': duration of each token in mel frames (1, text_length)
            - 'phonemes': list of phoneme characters
            - 'word_boundaries': list of (word, start_frame, end_frame) tuples
            - 'mel_config': the mel config used (for frame-to-time conversion)
    """
    if mel_config is None:
        mel_config = MelConfig()

    mel_extractor = LogMelSpectrogram(**asdict(mel_config)).to(device)

    # G2P mapping
    g2p_mapping = {
        'english': english_to_ipa2,
        # Add other languages as needed
    }
    phonemizer = g2p_mapping[language]

    # Convert text to phonemes, tracking word boundaries
    phonemes, word_to_phoneme_map = text_to_phonemes_with_word_map(text, phonemizer)

    # Convert to token IDs with interspersing (blank tokens between phonemes)
    token_ids = intersperse(cleaned_text_to_sequence(phonemes), item=0)
    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    x_lengths = torch.tensor([x.size(-1)], dtype=torch.long, device=device)

    # Load and convert audio to mel spectrogram
    audio = load_and_resample_audio(audio_path, mel_config.sample_rate).to(device)
    y = mel_extractor(audio)  # (1, n_mels, mel_length)
    y_lengths = torch.tensor([y.size(-1)], dtype=torch.long, device=device)

    # Run encoder to get mu_x (predicted mel statistics per token)
    y_mask = sequence_mask(y_lengths, y.size(2)).unsqueeze(1).to(y.dtype)

    # Get speaker embedding from the audio itself
    c = model.ref_encoder(y, y_mask)

    # Encode text conditioned on speaker
    _, mu_x, x_mask = model.encoder(x, c, x_lengths)

    # Compute alignment using MAS (same as training forward pass)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)

    with torch.no_grad():
        s_p_sq_r = torch.ones_like(mu_x)
        neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - torch.zeros_like(mu_x), [1], keepdim=True)
        neg_cent2 = torch.einsum("bdt, bds -> bts", -0.5 * (y**2), s_p_sq_r)
        neg_cent3 = torch.einsum("bdt, bds -> bts", y, (mu_x * s_p_sq_r))
        neg_cent4 = torch.sum(-0.5 * (mu_x**2) * s_p_sq_r, [1], keepdim=True)
        neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

        attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

    # Extract durations per token
    durations = attn.sum(dim=3).squeeze(0).squeeze(0)  # (text_length,)

    # Map phoneme-level durations back to word-level boundaries
    word_boundaries = compute_word_boundaries(
        durations, word_to_phoneme_map, mel_config
    )

    return {
        'attn': attn,
        'durations': durations,
        'phonemes': phonemes,
        'word_boundaries': word_boundaries,
        'mel_config': mel_config,
    }
```

### Step 2: Create Word-to-Phoneme Tracking

The critical piece is tracking which phonemes came from which word during G2P conversion. The current `english_to_ipa2()` function in `text/english.py` converts the entire text at once and returns a flat list of characters. We need a version that preserves word boundaries.

**Key insight about interspersing:** The `intersperse()` function in `datas/dataset.py` inserts a blank token (0) between every phoneme. So if a word produces phonemes `[a, b, c]`, after interspersing the full sequence looks like `[0, a, 0, b, 0, c, 0, ...]`. The word-to-token mapping must account for this doubling + 1 offset.

```python
def text_to_phonemes_with_word_map(text: str, phonemizer) -> tuple:
    """
    Convert text to phonemes while tracking which phonemes belong to which word.

    Args:
        text: Input text string
        phonemizer: G2P function (e.g., english_to_ipa2)

    Returns:
        phonemes: list of phoneme characters (as returned by phonemizer)
        word_to_phoneme_map: list of dicts with:
            - 'word': the original word string
            - 'phoneme_start': start index in the phoneme list
            - 'phoneme_end': end index in the phoneme list (exclusive)
            - 'token_start': start index in the interspersed token list
            - 'token_end': end index in the interspersed token list (exclusive)
    """
    words = text.split()
    word_to_phoneme_map = []

    # Process each word individually to track boundaries
    all_phonemes = []
    current_phoneme_idx = 0

    for i, word in enumerate(words):
        # Add space between words (except first)
        if i > 0:
            word_with_context = ' ' + word
        else:
            word_with_context = word

        # Get phonemes for this word
        word_phonemes = phonemizer(word_with_context)

        phoneme_start = current_phoneme_idx
        phoneme_end = current_phoneme_idx + len(word_phonemes)

        # Account for interspersing: each phoneme at index i becomes token at index 2*i + 1
        # The blank tokens are at even indices
        token_start = 2 * phoneme_start + 1  # first real token for this word
        token_end = 2 * (phoneme_end - 1) + 2  # last real token + 1 for this word

        word_to_phoneme_map.append({
            'word': word,
            'phoneme_start': phoneme_start,
            'phoneme_end': phoneme_end,
            'token_start': token_start,
            'token_end': token_end,
        })

        all_phonemes.extend(word_phonemes)
        current_phoneme_idx = phoneme_end

    return all_phonemes, word_to_phoneme_map


def compute_word_boundaries(durations, word_to_phoneme_map, mel_config):
    """
    Convert per-token durations to word-level time boundaries.

    Args:
        durations: tensor of shape (interspersed_text_length,) with frame counts per token
        word_to_phoneme_map: output from text_to_phonemes_with_word_map
        mel_config: MelConfig for frame-to-time conversion

    Returns:
        list of (word, start_time_seconds, end_time_seconds, start_frame, end_frame) tuples
    """
    # Cumulative sum gives us frame boundaries
    cum_durations = torch.cumsum(durations, dim=0)

    frame_to_seconds = mel_config.hop_length / mel_config.sample_rate

    word_boundaries = []
    for entry in word_to_phoneme_map:
        token_start = entry['token_start']
        token_end = entry['token_end']

        # Start frame is the cumulative duration up to (but not including) this word's first token
        start_frame = int(cum_durations[token_start - 1].item()) if token_start > 0 else 0
        # End frame is the cumulative duration through this word's last token
        end_frame = int(cum_durations[min(token_end - 1, len(durations) - 1)].item())

        start_time = start_frame * frame_to_seconds
        end_time = end_frame * frame_to_seconds

        word_boundaries.append({
            'word': entry['word'],
            'start_time': start_time,
            'end_time': end_time,
            'start_frame': start_frame,
            'end_frame': end_frame,
        })

    return word_boundaries
```

### Step 3: Text Diff → Mel Frame Mask

Given original text and edited text, compute which mel frames need to be regenerated.

```python
import difflib

def compute_edit_mask(original_text: str, edited_text: str, word_boundaries: list, total_mel_frames: int):
    """
    Given original text, edited text, and word boundaries from alignment,
    compute a binary mask over mel frames indicating which frames to regenerate.

    Args:
        original_text: The original transcript
        edited_text: The edited transcript
        word_boundaries: Output from align_text_to_audio()['word_boundaries']
        total_mel_frames: Total number of mel frames in the original audio

    Returns:
        dict with:
            - 'mask': tensor of shape (total_mel_frames,) where 1 = regenerate, 0 = keep
            - 'unchanged_prefix_end': frame index where unchanged prefix ends
            - 'unchanged_suffix_start': frame index where unchanged suffix begins
            - 'edits': list of edit operations with frame ranges
    """
    original_words = original_text.split()
    edited_words = edited_text.split()

    # Use SequenceMatcher to find the diff
    matcher = difflib.SequenceMatcher(None, original_words, edited_words)

    mask = torch.zeros(total_mel_frames)
    edits = []

    # Find the longest common prefix and suffix
    unchanged_prefix_end = 0
    unchanged_suffix_start = total_mel_frames

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == 'equal':
            continue
        elif op in ('replace', 'delete', 'insert'):
            # For 'replace' and 'delete': mark the original word frames for regeneration
            if op in ('replace', 'delete'):
                start_frame = word_boundaries[i1]['start_frame']
                end_frame = word_boundaries[i2 - 1]['end_frame']
                mask[start_frame:end_frame] = 1.0

                edits.append({
                    'operation': op,
                    'original_words': original_words[i1:i2],
                    'new_words': edited_words[j1:j2] if op == 'replace' else [],
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                })
            elif op == 'insert':
                # For insertions, we need to mark the boundary where new content goes
                # The insertion point is between word_boundaries[i1-1] end and word_boundaries[i1] start
                if i1 > 0:
                    insert_frame = word_boundaries[i1 - 1]['end_frame']
                else:
                    insert_frame = 0

                edits.append({
                    'operation': 'insert',
                    'original_words': [],
                    'new_words': edited_words[j1:j2],
                    'insert_at_frame': insert_frame,
                })

    # Compute unchanged prefix/suffix boundaries
    # Find first masked frame
    masked_indices = (mask == 1.0).nonzero(as_tuple=True)[0]
    if len(masked_indices) > 0:
        unchanged_prefix_end = masked_indices[0].item()
        unchanged_suffix_start = masked_indices[-1].item() + 1

    return {
        'mask': mask,
        'unchanged_prefix_end': unchanged_prefix_end,
        'unchanged_suffix_start': unchanged_suffix_start,
        'edits': edits,
    }
```

### Step 4: Inpainting via the CFM Decoder

The existing hackathon code in `models/flow_matching.py` already implements a prefix/postfix approach where known audio is concatenated and masked during ODE integration. The inpainting approach generalizes this:

```python
def inpaint_audio(model: StableTTS, original_audio_path: str, original_text: str,
                  edited_text: str, language: str = 'english',
                  n_timesteps: int = 25, temperature: float = 1.0,
                  length_scale: float = 1.0, solver: str = 'dopri5', cfg: float = 3.0,
                  crossfade_frames: int = 5):
    """
    Full inpainting pipeline:
    1. Align original text to original audio (get word boundaries)
    2. Diff original vs edited text (get edit mask)
    3. Regenerate only the masked region using CFM decoder

    Args:
        model: Fine-tuned StableTTS model
        original_audio_path: Path to the original recorded audio
        original_text: Original transcript
        edited_text: Edited transcript
        language: Language for G2P
        n_timesteps: ODE solver steps
        temperature: Noise temperature
        length_scale: Duration scaling
        solver: ODE solver method
        cfg: Classifier-free guidance strength
        crossfade_frames: Number of frames for soft crossfade at boundaries

    Returns:
        Regenerated audio tensor and mel spectrogram
    """
    # Step 1: Align
    alignment = align_text_to_audio(model, original_text, original_audio_path, language)
    word_boundaries = alignment['word_boundaries']
    mel_config = alignment['mel_config']

    # Step 2: Compute edit mask
    original_mel = ...  # load original mel
    edit_info = compute_edit_mask(original_text, edited_text, word_boundaries, original_mel.size(-1))

    # Step 3: Determine prefix/suffix audio from the mask
    prefix_end = edit_info['unchanged_prefix_end']
    suffix_start = edit_info['unchanged_suffix_start']

    prefix_mel = original_mel[:, :, :prefix_end]
    suffix_mel = original_mel[:, :, suffix_start:]

    # Step 4: Get the new text for the edited region
    # ... (extract the changed portion of edited_text)

    # Step 5: Use the existing prefix/postfix synthesis approach
    # This leverages the already-implemented flow_matching.py prefix/postfix logic
    # The model generates the middle portion conditioned on prefix and suffix context

    # Step 6: Apply soft crossfade at boundaries
    # ... (linear ramp over crossfade_frames at each boundary)
```

## Critical Implementation Details

### 1. The Interspersing Problem

The `intersperse()` function doubles the token sequence length plus one:
```python
# Original phonemes: [p1, p2, p3]  (length 3)
# After intersperse: [0, p1, 0, p2, 0, p3, 0]  (length 7 = 2*3 + 1)
```

So phoneme at original index `i` becomes token at index `2*i + 1`. The blank tokens at even indices also get durations assigned by MAS. When computing word boundaries, you must sum the durations of both the phoneme tokens AND the surrounding blank tokens that belong to that word.

### 2. G2P Word Boundary Tracking

The current `english_to_ipa2()` function processes the entire text at once through `eng_to_ipa.convert()`. This is problematic because:
- It doesn't preserve word boundaries in the output
- Some phoneme transformations span word boundaries

**Recommended approach:** Process words individually through the G2P pipeline, then concatenate. The space character between words in IPA serves as a natural delimiter. In the phoneme list output of `english_to_ipa2()`, spaces correspond to word boundaries.

A simpler alternative: since `english_to_ipa2()` returns a list of characters and spaces are preserved, you can find word boundaries by locating space characters in the phoneme list:

```python
def find_word_boundaries_in_phonemes(phonemes: list) -> list:
    """Find word boundary indices by locating spaces in the phoneme list."""
    boundaries = [0]  # start of first word
    for i, p in enumerate(phonemes):
        if p == ' ':
            boundaries.append(i + 1)  # start of next word
    boundaries.append(len(phonemes))  # end of last word
    return boundaries
```

### 3. Mel Frame to Time Conversion

From `config.py`:
- `sample_rate = 44100`
- `hop_length = 512`

Each mel frame represents `hop_length / sample_rate = 512 / 44100 ≈ 0.0116 seconds ≈ 11.6ms`.

To convert frame index to time: `time_seconds = frame_index * 512 / 44100`

### 4. Why This Works on Non-Generated Audio

MAS finds the alignment that maximizes the log-likelihood of the mel spectrogram given the encoder's predicted distribution. When the model is fine-tuned on a speaker:
- The encoder learns to predict mel statistics (`mu_x`) that match that speaker's voice characteristics
- MAS then finds where each predicted phoneme pattern best matches the actual audio
- This is essentially a **forced alignment** using the TTS model's learned representations

The quality depends on:
- How well the model is fine-tuned to the speaker (more training data = better alignment)
- How clean the audio is (background noise degrades alignment)
- Whether the text exactly matches what was spoken (mismatches cause alignment errors)

### 5. Handling Duration Mismatches in Edits

When the edited text has different content than the original:
- **Replacements:** The new words may have different natural duration than the old words. Let the duration predictor determine the new duration for the replacement segment.
- **Insertions:** The model predicts duration for the new words. The total audio length changes.
- **Deletions:** Simply remove the frames corresponding to deleted words. The total audio length shrinks.

## File Structure for Implementation

```
StableTTS_Editing/
├── alignment.py              # NEW: align_text_to_audio(), word boundary computation
├── inpainting.py             # NEW: compute_edit_mask(), inpaint_audio()
├── text/
│   └── english.py            # MODIFY: add word-boundary-aware G2P variant
├── models/
│   ├── model.py              # MODIFY: add align() method to StableTTS class
│   └── flow_matching.py      # MODIFY: generalize inpainting mask (already has prefix/postfix)
├── app_inpainting.py         # NEW: Gradio UI for the inpainting workflow
└── INPAINTING_ALIGNMENT_DESIGN.md  # THIS FILE
```

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INPAINTING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUT: original_audio + original_text + edited_text                 │
│                                                                      │
│  1. G2P: original_text → phonemes (with word boundary tracking)      │
│  2. Mel: original_audio → mel spectrogram                            │
│  3. Encode: phonemes → mu_x (encoder predicted mel stats)            │
│  4. MAS: align mu_x against mel → alignment matrix (attn)           │
│  5. Extract: attn → per-token durations → word boundaries            │
│  6. Diff: original_text vs edited_text → identify changed words      │
│  7. Map: changed words → mel frame ranges (using word boundaries)    │
│  8. Split: original mel into prefix | edit_region | suffix           │
│  9. Generate: new mel for edit_region using CFM decoder               │
│     - Conditioned on prefix/suffix context                           │
│     - Text input is the edited portion                               │
│ 10. Stitch: prefix_mel + generated_mel + suffix_mel                  │
│ 11. Vocoder: combined mel → audio waveform                           │
│                                                                      │
│  OUTPUT: audio with only the edited portion regenerated              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Relationship to Existing Hackathon Code

The hackathon code in `streamlined_app.py` used **Whisper** for word-level timestamps to determine prefix/postfix boundaries. This design replaces that with the **model's own alignment**, which:

1. **Eliminates the Whisper dependency** — no external ASR model needed
2. **Works for any language** the TTS model supports (not just English)
3. **Is more accurate for the specific speaker** because the alignment uses the fine-tuned model's learned representations
4. **Creates a fully self-contained pipeline** — only the StableTTS model is needed

The existing `models/flow_matching.py` prefix/postfix concatenation approach is the right foundation for the actual regeneration step. The alignment module provides the missing piece: **automatically determining where to split the audio** rather than requiring manual prefix/postfix audio files.

## Testing Strategy

1. **Unit test alignment quality:** Run `align_text_to_audio()` on training data where you know the ground truth alignment (from the training forward pass). Compare the inferred alignment to the training-time MAS alignment.

2. **Visual validation:** Plot the alignment matrix overlaid on the mel spectrogram with word boundaries marked. This immediately shows if alignment is reasonable.

3. **Round-trip test:** Take generated audio (where you know exact boundaries from synthesis), run alignment on it, verify boundaries match.

4. **Real audio test:** Fine-tune on a speaker, record new audio, run alignment, listen to the segments to verify word boundaries are correct.
