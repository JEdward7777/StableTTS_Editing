"""
MAS-based text-to-audio alignment module.

Uses StableTTS's built-in Monotonic Alignment Search (MAS) to determine word
boundaries in audio, enabling automatic prefix/suffix detection for the
inpainting pipeline.

See MAS_IMPLEMENTATION_PLAN.md and INPAINTING_ALIGNMENT_DESIGN.md for details.
"""

import math
import torch
import monotonic_align

from text import cleaned_text_to_sequence
from datas.dataset import intersperse
from utils.audio import load_and_resample_audio
from utils.mask import sequence_mask


def align_text_to_audio(
    api,
    text: str,
    audio_path: str | None,
    language: str = 'english',
    mel_tensor: 'torch.Tensor | None' = None,
) -> dict:
    """
    Use the trained StableTTS model's encoder + MAS to align text to audio.

    This replicates the alignment computation from the training forward pass
    (models/model.py lines 211-220), but is designed to be called at inference
    time on arbitrary (text, audio) pairs.

    Args:
        api: A StableTTSAPI instance (provides tts_model, mel_extractor,
             mel_config, g2p_mapping — avoids duplicating the model in memory)
        text: The transcript text corresponding to the audio
        audio_path: Path to the audio file.  May be None when *mel_tensor* is
            provided.
        language: Language for G2P conversion (default: 'english')
        mel_tensor: Optional pre-computed mel spectrogram tensor of shape
            (1, n_mels, mel_length).  When supplied, *audio_path* is ignored
            and no file I/O is performed.  This is the preferred path when the
            caller already has the mel from a previous inference step.

    Returns:
        dict with:
            - 'attn': alignment matrix (1, 1, interspersed_text_length, mel_length)
            - 'durations': duration of each interspersed token in mel frames (interspersed_text_length,)
            - 'word_boundaries': list of dicts with word, start_frame, end_frame, start_time, end_time
            - 'mel': the mel spectrogram tensor (1, n_mels, mel_length) — reusable downstream
            - 'mel_config': the MelConfig used
            - 'phonemes': the phoneme character list from G2P
    """
    device = next(api.parameters()).device
    mel_config = api.mel_config

    # --- G2P: text → phonemes ---
    phonemizer = api.g2p_mapping.get(language)
    if phonemizer is None:
        raise ValueError(f"Unsupported language: {language}. Supported: {list(api.g2p_mapping.keys())}")

    phonemes = phonemizer(text)  # list of characters (spaces preserved between words)

    # --- Find word boundaries by locating spaces in the phoneme list ---
    word_phoneme_boundaries = _find_word_boundaries_in_phonemes(phonemes, text)

    # --- Convert phonemes to interspersed token IDs ---
    token_ids = intersperse(cleaned_text_to_sequence(phonemes), item=0)
    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    x_lengths = torch.tensor([x.size(-1)], dtype=torch.long, device=device)

    # --- Load audio → mel spectrogram (or use pre-computed tensor) ---
    if mel_tensor is not None:
        # Fast path: caller already has the mel — no file I/O needed.
        mel = mel_tensor.to(device)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # ensure (1, n_mels, mel_length)
    else:
        audio = load_and_resample_audio(audio_path, mel_config.sample_rate).to(device)
        mel = api.mel_extractor(audio)  # (1, n_mels, mel_length)
    y = mel
    y_lengths = torch.tensor([y.size(-1)], dtype=torch.long, device=device)

    # --- Run encoder to get mu_x (predicted mel statistics per token) ---
    y_mask = sequence_mask(y_lengths, y.size(2)).unsqueeze(1).to(y.dtype)

    # Get speaker embedding from the audio
    c = api.tts_model.ref_encoder(y, y_mask)

    # Encode text conditioned on speaker
    _, mu_x, x_mask = api.tts_model.encoder(x, c, x_lengths)

    # --- Compute alignment using MAS (same as training forward pass) ---
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)

    with torch.no_grad():
        s_p_sq_r = torch.ones_like(mu_x)  # [b, d, t]
        neg_cent1 = torch.sum(
            -0.5 * math.log(2 * math.pi) - torch.zeros_like(mu_x), [1], keepdim=True
        )
        neg_cent2 = torch.einsum("bdt, bds -> bts", -0.5 * (y ** 2), s_p_sq_r)
        neg_cent3 = torch.einsum("bdt, bds -> bts", y, (mu_x * s_p_sq_r))
        neg_cent4 = torch.sum(-0.5 * (mu_x ** 2) * s_p_sq_r, [1], keepdim=True)
        neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

        attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

    # --- Extract per-token durations ---
    # attn shape from MAS: (1, 1, mel_length, interspersed_text_length)
    # Summing over dim=2 (mel_length) gives duration per text token
    # This matches the training code: logw_ = torch.log(1e-8 + attn.sum(2)) * x_mask
    durations = attn.sum(dim=2).squeeze(0).squeeze(0)  # (interspersed_text_length,)

    # --- Map token durations → word-level frame boundaries ---
    word_boundaries = _compute_word_boundaries(durations, word_phoneme_boundaries, mel_config)

    return {
        'attn': attn,
        'durations': durations,
        'word_boundaries': word_boundaries,
        'mel': mel,
        'mel_config': mel_config,
        'phonemes': phonemes,
    }


def _find_word_boundaries_in_phonemes(phonemes: list, original_text: str) -> list:
    """
    Find word boundary indices by locating spaces in the phoneme list.

    The G2P function (e.g., english_to_ipa2) preserves spaces between words.
    We locate those spaces to determine which phonemes belong to which word.

    Args:
        phonemes: list of phoneme characters from G2P (spaces preserved)
        original_text: the original text string (for word labels)

    Returns:
        list of dicts with:
            - 'word': the original word string
            - 'phoneme_start': start index in the phoneme list (inclusive)
            - 'phoneme_end': end index in the phoneme list (exclusive)
    """
    words = original_text.split()

    # Find indices of space characters in the phoneme list
    space_indices = [i for i, p in enumerate(phonemes) if p == ' ']

    # Build word boundaries from space positions
    # Words are separated by spaces, so:
    #   word 0: phonemes[0 : space_indices[0]]
    #   word 1: phonemes[space_indices[0]+1 : space_indices[1]]
    #   ...
    #   word N: phonemes[space_indices[-1]+1 : len(phonemes)]
    boundaries = []
    start = 0
    word_idx = 0

    for space_idx in space_indices:
        if start < space_idx:  # non-empty word
            boundaries.append({
                'word': words[word_idx] if word_idx < len(words) else f'word_{word_idx}',
                'phoneme_start': start,
                'phoneme_end': space_idx,
            })
            word_idx += 1
        start = space_idx + 1

    # Last word (after the last space, or the entire phoneme list if no spaces)
    if start < len(phonemes):
        boundaries.append({
            'word': words[word_idx] if word_idx < len(words) else f'word_{word_idx}',
            'phoneme_start': start,
            'phoneme_end': len(phonemes),
        })

    return boundaries


def _compute_word_boundaries(durations: torch.Tensor, word_phoneme_boundaries: list, mel_config) -> list:
    """
    Convert per-token durations to word-level time boundaries.

    Accounts for the intersperse mapping: original phoneme at index i becomes
    interspersed token at index 2*i + 1. Blank tokens at even indices also
    receive durations from MAS.

    For each word, we sum the durations of all interspersed tokens that belong
    to it, including the blank tokens on either side of its phonemes.

    Blank token assignment:
    - The leading blank (index 0) belongs to the first word
    - Blanks between words (at the space position) are split: the blank
      immediately after a word's last phoneme belongs to that word
    - The trailing blank belongs to the last word

    Args:
        durations: tensor of shape (interspersed_text_length,) with frame counts per token
        word_phoneme_boundaries: output from _find_word_boundaries_in_phonemes
        mel_config: MelConfig for frame-to-time conversion

    Returns:
        list of dicts with word, start_frame, end_frame, start_time, end_time
    """
    frame_to_seconds = mel_config.hop_length / mel_config.sample_rate

    # Cumulative durations for frame position lookup
    cum_durations = torch.cumsum(durations, dim=0)

    word_boundaries = []

    for i, entry in enumerate(word_phoneme_boundaries):
        phoneme_start = entry['phoneme_start']
        phoneme_end = entry['phoneme_end']  # exclusive

        # Map phoneme indices to interspersed token indices
        # Phoneme at original index j → interspersed token at index 2*j + 1
        # We want to include the blank tokens surrounding this word's phonemes.
        #
        # For the first word, include the leading blank (token 0).
        # For each word, include up to the blank after its last phoneme.
        if i == 0:
            token_start = 0  # include leading blank
        else:
            # Start at the blank just before this word's first phoneme
            token_start = 2 * phoneme_start  # the blank at position 2*phoneme_start

        # End includes the blank after the last phoneme of this word
        # Last phoneme is at original index (phoneme_end - 1) → token index 2*(phoneme_end-1) + 1
        # The blank after it is at token index 2*(phoneme_end-1) + 2 = 2*phoneme_end
        token_end = 2 * phoneme_end  # exclusive; this is the blank after last phoneme

        # For the last word, include the trailing blank
        if i == len(word_phoneme_boundaries) - 1:
            token_end = len(durations)  # include trailing blank

        # Clamp to valid range
        token_start = max(0, min(token_start, len(durations)))
        token_end = min(len(durations), token_end)

        # Compute frame boundaries
        start_frame = int(cum_durations[token_start - 1].item()) if token_start > 0 else 0
        end_frame = int(cum_durations[token_end - 1].item()) if token_end > 0 else 0

        start_time = start_frame * frame_to_seconds
        end_time = end_frame * frame_to_seconds

        word_boundaries.append({
            'word': entry['word'],
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_time': start_time,
            'end_time': end_time,
            'phoneme_start': phoneme_start,
            'phoneme_end': phoneme_end,
        })

    return word_boundaries


def text_position_to_mel_frame(
    text: str,
    char_position: int,
    durations: 'torch.Tensor',
    phonemizer,
) -> int:
    """Convert a character position in text to a mel frame index.

    Uses piecewise G2P: runs the phonemizer on the text up to char_position,
    counts the resulting phonemes, maps to the interspersed token index, and
    sums MAS durations to get the exact mel frame.

    The G2P pipeline (e.g., eng_to_ipa) strips trailing whitespace, so if the
    text up to char_position ends with a space, we add it back to the phoneme
    list to maintain correct alignment with the full-text phoneme sequence.

    Args:
        text: The full text string that was aligned with MAS.
        char_position: Character index in ``text`` (0-based). The returned frame
            is the boundary *before* this character — i.e., the mel frame where
            the audio for characters [0, char_position) ends.
        durations: Per-interspersed-token duration tensor from MAS alignment.
            Shape: (interspersed_text_length,).
        phonemizer: The G2P function (e.g., ``english_to_ipa2``). Must accept a
            string and return a list of phoneme characters.

    Returns:
        Mel frame index corresponding to the boundary at ``char_position``.
    """
    if char_position <= 0:
        return 0
    if char_position >= len(text):
        return int(durations.sum().item())

    # Extract the prefix text up to the character position
    prefix_text = text[:char_position]

    # Run G2P on the stripped prefix
    prefix_phonemes = phonemizer(prefix_text.strip())

    # G2P strips trailing whitespace. If the prefix ends with a space,
    # add it back so the phoneme count matches the full-text alignment.
    if prefix_text.endswith(' '):
        prefix_phonemes = list(prefix_phonemes) + [' ']

    # Map phoneme count to interspersed token index.
    # After intersperse, phoneme at index i becomes token 2*i + 1.
    # The prefix occupies tokens 0 through 2*N (inclusive), where N = len(prefix_phonemes).
    # Token 2*N is the blank after the last prefix phoneme.
    n_prefix_phonemes = len(prefix_phonemes)
    interspersed_end = 2 * n_prefix_phonemes  # exclusive: sum durations [0, interspersed_end)

    # Clamp to valid range
    interspersed_end = min(interspersed_end, len(durations))

    if interspersed_end <= 0:
        return 0

    # Sum durations for the prefix tokens to get the mel frame
    return int(durations[:interspersed_end].sum().item())


def compute_edit_regions(original_text: str, edited_text: str, word_boundaries: list, total_mel_frames: int) -> dict:
    """
    Given original text, edited text, and word boundaries from alignment,
    compute the prefix/suffix split points for inpainting.

    Uses longest common prefix/suffix at the word level (case-sensitive).

    Args:
        original_text: The original transcript
        edited_text: The edited transcript
        word_boundaries: Output from align_text_to_audio()['word_boundaries']
        total_mel_frames: Total number of mel frames in the original audio

    Returns:
        dict with:
            - 'has_edit': bool — False if texts are identical
            - 'prefix_end_frame': frame index where unchanged prefix ends
            - 'suffix_start_frame': frame index where unchanged suffix begins
            - 'prefix_text': unchanged prefix text (for encoder conditioning)
            - 'edited_text': the changed portion of the edited text
            - 'suffix_text': unchanged suffix text (for encoder conditioning)
            - 'prefix_words': list of unchanged prefix words
            - 'suffix_words': list of unchanged suffix words
    """
    original_words = original_text.split()
    edited_words = edited_text.split()

    # Guard: no edit
    if original_words == edited_words:
        return {
            'has_edit': False,
            'prefix_end_frame': 0,
            'suffix_start_frame': total_mel_frames,
            'prefix_text': original_text,
            'edited_text': '',
            'suffix_text': '',
            'prefix_words': original_words,
            'suffix_words': [],
        }

    # Find longest common prefix (word-level, case-sensitive)
    prefix_word_count = 0
    for o, e in zip(original_words, edited_words):
        if o == e:
            prefix_word_count += 1
        else:
            break

    # Find longest common suffix (word-level, case-sensitive)
    suffix_word_count = 0
    for o, e in zip(reversed(original_words), reversed(edited_words)):
        if o == e:
            suffix_word_count += 1
        else:
            break

    # Guard: ensure prefix + suffix don't overlap
    # This can happen if the edit is in the middle and the total word count is small
    max_overlap = min(len(original_words), len(edited_words))
    if prefix_word_count + suffix_word_count > max_overlap:
        suffix_word_count = max_overlap - prefix_word_count

    # Compute frame boundaries from word_boundaries
    if prefix_word_count > 0 and prefix_word_count <= len(word_boundaries):
        # Prefix ends at the end_frame of the last prefix word
        prefix_end_frame = word_boundaries[prefix_word_count - 1]['end_frame']
    else:
        prefix_end_frame = 0

    if suffix_word_count > 0 and suffix_word_count <= len(word_boundaries):
        # Suffix starts at the start_frame of the first suffix word
        suffix_word_idx = len(word_boundaries) - suffix_word_count
        suffix_start_frame = word_boundaries[suffix_word_idx]['start_frame']
    else:
        suffix_start_frame = total_mel_frames

    # Extract text segments
    prefix_words = original_words[:prefix_word_count]
    suffix_words = original_words[len(original_words) - suffix_word_count:] if suffix_word_count > 0 else []

    # The edited text is the portion between prefix and suffix in the EDITED text
    edited_start = prefix_word_count
    edited_end = len(edited_words) - suffix_word_count if suffix_word_count > 0 else len(edited_words)
    edited_portion_words = edited_words[edited_start:edited_end]

    prefix_text = ' '.join(prefix_words)
    edited_portion_text = ' '.join(edited_portion_words)
    suffix_text = ' '.join(suffix_words)

    return {
        'has_edit': True,
        'prefix_end_frame': prefix_end_frame,
        'suffix_start_frame': suffix_start_frame,
        'prefix_text': prefix_text,
        'edited_text': edited_portion_text,
        'suffix_text': suffix_text,
        'prefix_words': prefix_words,
        'suffix_words': suffix_words,
    }
