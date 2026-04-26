"""
Multi-edit diff module for text-to-audio inpainting.

Uses JLDiff (https://github.com/JEdward7777/JLDiff) to compute word-level diffs
between an original and edited text string, then extracts discrete edit regions
suitable for iterative audio inpainting.

The pipeline:
1. compute_diff_by_words() — token-level diff (words, whitespace, punctuation)
2. coalesce_diff() — merge short coincidental matches below a threshold
3. extract_edit_regions() — walk diff nodes, extract contiguous edit regions
4. Return edit regions in right-to-left order for stable iterative application

Each edit region carries:
- Character positions in both original and edited text
- The old and new substrings
- An edit type (insertion, deletion, or substitution)
"""

from dataclasses import dataclass
from typing import List, Optional

from JLDiff import (
    compute_diff_by_words,
    coalesce_diff,
    STATE_MATCH,
    STATE_PASSING_1ST,
    STATE_PASSING_2ND,
)


@dataclass
class EditRegion:
    """A single contiguous edit region between original and edited text.

    Attributes:
        original_start: Start character index in the original string (inclusive).
        original_end:   End character index in the original string (exclusive).
        edited_start:   Start character index in the edited string (inclusive).
        edited_end:     End character index in the edited string (exclusive).
        old_text:       The substring being replaced (from original). Empty for insertions.
        new_text:       The replacement substring (from edited). Empty for deletions.
        edit_type:      One of 'insertion', 'deletion', or 'substitution'.
    """
    original_start: int
    original_end: int
    edited_start: int
    edited_end: int
    old_text: str
    new_text: str
    edit_type: str  # 'insertion', 'deletion', or 'substitution'

    def __repr__(self) -> str:
        return (
            f"EditRegion({self.edit_type}: "
            f"orig[{self.original_start}:{self.original_end}]={self.old_text!r} → "
            f"edit[{self.edited_start}:{self.edited_end}]={self.new_text!r})"
        )


@dataclass
class MultiEditResult:
    """Result of computing multi-edit diff between two strings.

    Attributes:
        original_text:  The original input string.
        edited_text:    The edited input string.
        edits:          List of EditRegion objects, ordered right-to-left
                        (last edit in the text first) for stable iterative application.
        min_match:      The min_match threshold that was used for coalescence.
    """
    original_text: str
    edited_text: str
    edits: List[EditRegion]
    min_match: int

    def get_intermediate_texts(self) -> List[str]:
        """Generate the sequence of intermediate text states.

        Starting from the original text, applies each edit (right-to-left order)
        and returns the text state after each edit. The final element equals
        edited_text.

        Returns:
            List of strings, one per edit, showing the text after that edit
            is applied. If there are no edits, returns an empty list.
        """
        if not self.edits:
            return []

        states = []
        current = self.original_text

        for edit in self.edits:
            # Since edits are right-to-left, each edit's original_start/end
            # positions are valid in the current text (earlier edits haven't
            # shifted these positions yet).
            current = current[:edit.original_start] + edit.new_text + current[edit.original_end:]
            states.append(current)

        return states


def compute_edit_regions(
    original: str,
    edited: str,
    min_match: int = 2,
    talk: bool = False,
) -> MultiEditResult:
    """Compute multiple edit regions between two text strings.

    Uses JLDiff's word-level diff with match coalescence to find discrete
    edit regions, then extracts them with character positions in both strings.

    Args:
        original: The original text string.
        edited:   The edited text string.
        min_match: Minimum match length (in characters) to preserve. Match
                   groups shorter than this are merged into surrounding changes.
                   Default is 2, which merges edits separated only by a single
                   space or punctuation character. Set to 1 to disable coalescence.
        talk:     If True, JLDiff prints progress to stdout. Default False.

    Returns:
        MultiEditResult with edit regions in right-to-left order.
    """
    # Step 1: Word-level diff
    diff_nodes = compute_diff_by_words(original, edited, talk=talk)

    # Step 2: Coalesce short matches
    if min_match > 1:
        diff_nodes = coalesce_diff(diff_nodes, min_match=min_match)

    # Step 3: Extract edit regions by walking the diff nodes
    edits = _extract_regions_from_diff(diff_nodes)

    # Step 4: Reverse to right-to-left order
    edits.reverse()

    return MultiEditResult(
        original_text=original,
        edited_text=edited,
        edits=edits,
        min_match=min_match,
    )


def _extract_regions_from_diff(diff_nodes) -> List[EditRegion]:
    """Walk coalesced diff nodes and extract contiguous edit regions.

    Diff nodes after coalescence are consolidated: between surviving match
    groups, all deletions are in one node and all insertions are in one node.
    We track character positions in both original and edited text as we walk.

    Args:
        diff_nodes: List of diff nodes from JLDiff (after coalesce_diff).

    Returns:
        List of EditRegion objects in left-to-right order.
    """
    edits: List[EditRegion] = []

    # Current character position in original and edited text
    orig_pos = 0
    edit_pos = 0

    # Accumulate contiguous non-match content into an edit region
    current_old = []  # parts from STATE_PASSING_1ST (deletions from original)
    current_new = []  # parts from STATE_PASSING_2ND (insertions into edited)
    region_orig_start: Optional[int] = None
    region_edit_start: Optional[int] = None

    def flush_region():
        """Emit the accumulated edit region if non-empty."""
        nonlocal current_old, current_new, region_orig_start, region_edit_start

        old_text = ''.join(current_old)
        new_text = ''.join(current_new)

        if not old_text and not new_text:
            # Nothing accumulated (e.g., empty match nodes)
            current_old = []
            current_new = []
            region_orig_start = None
            region_edit_start = None
            return

        # Determine edit type
        if not old_text:
            edit_type = 'insertion'
        elif not new_text:
            edit_type = 'deletion'
        else:
            edit_type = 'substitution'

        edits.append(EditRegion(
            original_start=region_orig_start if region_orig_start is not None else orig_pos,
            original_end=region_orig_start + len(old_text) if region_orig_start is not None else orig_pos,
            edited_start=region_edit_start if region_edit_start is not None else edit_pos,
            edited_end=region_edit_start + len(new_text) if region_edit_start is not None else edit_pos,
            old_text=old_text,
            new_text=new_text,
            edit_type=edit_type,
        ))

        current_old = []
        current_new = []
        region_orig_start = None
        region_edit_start = None

    for node in diff_nodes:
        content = node.content if node.content else ''

        if node.state == STATE_MATCH:
            # Flush any pending edit region before advancing past a match
            flush_region()

            # Advance both positions by the match length
            orig_pos += len(content)
            edit_pos += len(content)

        elif node.state == STATE_PASSING_1ST:
            # Content from original only (deletion)
            if region_orig_start is None:
                region_orig_start = orig_pos
            if region_edit_start is None:
                region_edit_start = edit_pos
            current_old.append(content)
            orig_pos += len(content)

        elif node.state == STATE_PASSING_2ND:
            # Content from edited only (insertion)
            if region_orig_start is None:
                region_orig_start = orig_pos
            if region_edit_start is None:
                region_edit_start = edit_pos
            current_new.append(content)
            edit_pos += len(content)

    # Flush any trailing edit region
    flush_region()

    return edits
