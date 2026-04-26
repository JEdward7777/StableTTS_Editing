"""
Unit tests for multi_edit_diff module.

Run with:
    ./venv311/bin/python3 -m pytest test_multi_edit_diff.py -v
or:
    ./venv311/bin/python3 test_multi_edit_diff.py
"""

import unittest
from multi_edit_diff import compute_edit_regions, EditRegion, MultiEditResult


class TestComputeEditRegions(unittest.TestCase):
    """Tests for compute_edit_regions()."""

    # --- Basic functionality ---

    def test_no_changes(self):
        """Identical strings produce no edits."""
        result = compute_edit_regions("Hello world", "Hello world")
        self.assertEqual(len(result.edits), 0)
        self.assertEqual(result.original_text, "Hello world")
        self.assertEqual(result.edited_text, "Hello world")
        self.assertEqual(result.get_intermediate_texts(), [])

    def test_single_word_substitution(self):
        """Changing one word produces a single substitution."""
        result = compute_edit_regions("The cat sat", "The dog sat")
        self.assertEqual(len(result.edits), 1)
        edit = result.edits[0]
        self.assertEqual(edit.edit_type, "substitution")
        self.assertEqual(edit.old_text, "cat")
        self.assertEqual(edit.new_text, "dog")

    def test_single_word_insertion(self):
        """Inserting a word produces an insertion edit."""
        result = compute_edit_regions("Hello world", "Hello beautiful world")
        self.assertEqual(len(result.edits), 1)
        edit = result.edits[0]
        self.assertEqual(edit.edit_type, "insertion")
        self.assertEqual(edit.old_text, "")
        self.assertEqual(edit.new_text, "beautiful ")

    def test_single_word_deletion(self):
        """Deleting a word produces a deletion edit."""
        result = compute_edit_regions("Hello beautiful world", "Hello world")
        self.assertEqual(len(result.edits), 1)
        edit = result.edits[0]
        self.assertEqual(edit.edit_type, "deletion")
        self.assertEqual(edit.old_text, "beautiful ")
        self.assertEqual(edit.new_text, "")

    # --- Coalescence / merging ---

    def test_nearby_edits_merge_with_default_min_match(self):
        """Two edits separated by a single space merge with default min_match=2."""
        result = compute_edit_regions("The cat sat on the mat.", "The dog ran on the mat.")
        self.assertEqual(len(result.edits), 1)
        edit = result.edits[0]
        self.assertEqual(edit.edit_type, "substitution")
        self.assertEqual(edit.old_text, "cat sat")
        self.assertEqual(edit.new_text, "dog ran")

    def test_nearby_edits_stay_separate_with_min_match_1(self):
        """Two edits separated by a space stay separate when min_match=1 (no coalescence)."""
        result = compute_edit_regions(
            "The cat sat on the mat.", "The dog ran on the mat.", min_match=1
        )
        self.assertEqual(len(result.edits), 2)
        # Right-to-left order: sat->ran first, then cat->dog
        self.assertEqual(result.edits[0].old_text, "sat")
        self.assertEqual(result.edits[0].new_text, "ran")
        self.assertEqual(result.edits[1].old_text, "cat")
        self.assertEqual(result.edits[1].new_text, "dog")

    def test_distant_edits_stay_separate(self):
        """Edits separated by many words stay separate even with coalescence."""
        result = compute_edit_regions(
            "Hello beautiful world today is nice",
            "Goodbye beautiful earth today was",
        )
        self.assertEqual(len(result.edits), 3)

    # --- Character positions ---

    def test_original_positions_correct(self):
        """Character positions in original text are correct."""
        result = compute_edit_regions("The cat sat", "The dog sat")
        edit = result.edits[0]
        self.assertEqual(result.original_text[edit.original_start:edit.original_end], edit.old_text)

    def test_edited_positions_correct(self):
        """Character positions in edited text are correct."""
        result = compute_edit_regions("The cat sat", "The dog sat")
        edit = result.edits[0]
        self.assertEqual(result.edited_text[edit.edited_start:edit.edited_end], edit.new_text)

    def test_positions_correct_multiple_edits(self):
        """Character positions are correct for all edits in a multi-edit result."""
        result = compute_edit_regions(
            "Hello beautiful world today is nice",
            "Goodbye beautiful earth today was",
        )
        for edit in result.edits:
            self.assertEqual(
                result.original_text[edit.original_start:edit.original_end],
                edit.old_text,
                f"Original position mismatch for {edit}",
            )
            self.assertEqual(
                result.edited_text[edit.edited_start:edit.edited_end],
                edit.new_text,
                f"Edited position mismatch for {edit}",
            )

    # --- Right-to-left ordering ---

    def test_edits_ordered_right_to_left(self):
        """Edits are returned in right-to-left order (highest position first)."""
        result = compute_edit_regions(
            "Hello beautiful world today is nice",
            "Goodbye beautiful earth today was",
        )
        # Each edit's original_start should be >= the next edit's original_start
        for i in range(len(result.edits) - 1):
            self.assertGreaterEqual(
                result.edits[i].original_start,
                result.edits[i + 1].original_start,
                "Edits not in right-to-left order",
            )

    # --- Intermediate text states ---

    def test_intermediate_texts_single_edit(self):
        """Single edit produces one intermediate text equal to the edited text."""
        result = compute_edit_regions("The cat sat", "The dog sat")
        intermediates = result.get_intermediate_texts()
        self.assertEqual(len(intermediates), 1)
        self.assertEqual(intermediates[-1], "The dog sat")

    def test_intermediate_texts_multiple_edits(self):
        """Multiple edits produce correct intermediate states, ending at edited text."""
        result = compute_edit_regions(
            "Hello beautiful world today is nice",
            "Goodbye beautiful earth today was",
        )
        intermediates = result.get_intermediate_texts()
        self.assertEqual(len(intermediates), len(result.edits))
        # Final intermediate should equal the edited text
        self.assertEqual(intermediates[-1], result.edited_text)

    def test_intermediate_texts_no_edits(self):
        """No edits produce empty intermediate list."""
        result = compute_edit_regions("Hello world", "Hello world")
        self.assertEqual(result.get_intermediate_texts(), [])

    # --- Punctuation handling ---

    def test_punctuation_change_does_not_affect_adjacent_word(self):
        """Changing punctuation only affects the punctuation, not the adjacent word."""
        result = compute_edit_regions("Hello, world!", "Hello world.")
        # Should have two edits: comma removal and !->.
        # Neither should include "Hello" or "world" in old_text/new_text
        for edit in result.edits:
            self.assertNotIn("Hello", edit.old_text)
            self.assertNotIn("world", edit.old_text)
            self.assertNotIn("Hello", edit.new_text)
            self.assertNotIn("world", edit.new_text)

    def test_punctuation_only_deletion(self):
        """Removing a comma is a deletion edit."""
        result = compute_edit_regions("Hello, world", "Hello world")
        # Find the comma edit
        comma_edits = [e for e in result.edits if "," in e.old_text]
        self.assertEqual(len(comma_edits), 1)
        self.assertEqual(comma_edits[0].edit_type, "deletion")

    # --- Edge cases ---

    def test_empty_strings(self):
        """Both empty strings produce no edits."""
        result = compute_edit_regions("", "")
        self.assertEqual(len(result.edits), 0)

    def test_empty_to_text(self):
        """Empty original to non-empty edited is a single insertion."""
        result = compute_edit_regions("", "Hello world")
        self.assertEqual(len(result.edits), 1)
        self.assertEqual(result.edits[0].edit_type, "insertion")
        self.assertEqual(result.edits[0].new_text, "Hello world")

    def test_text_to_empty(self):
        """Non-empty original to empty edited is a single deletion."""
        result = compute_edit_regions("Hello world", "")
        self.assertEqual(len(result.edits), 1)
        self.assertEqual(result.edits[0].edit_type, "deletion")
        self.assertEqual(result.edits[0].old_text, "Hello world")

    def test_complete_replacement(self):
        """Completely different texts produce a single substitution."""
        result = compute_edit_regions("Hello world", "Goodbye earth")
        self.assertEqual(len(result.edits), 1)
        self.assertEqual(result.edits[0].edit_type, "substitution")

    def test_edit_at_start(self):
        """Edit at the very beginning of the text."""
        result = compute_edit_regions("Bad morning everyone", "Good morning everyone")
        self.assertEqual(len(result.edits), 1)
        self.assertEqual(result.edits[0].old_text, "Bad")
        self.assertEqual(result.edits[0].new_text, "Good")
        self.assertEqual(result.edits[0].original_start, 0)

    def test_edit_at_end(self):
        """Edit at the very end of the text."""
        result = compute_edit_regions("The cat is here", "The cat is there")
        self.assertEqual(len(result.edits), 1)
        self.assertEqual(result.edits[0].old_text, "here")
        self.assertEqual(result.edits[0].new_text, "there")

    # --- MultiEditResult dataclass ---

    def test_result_stores_min_match(self):
        """Result stores the min_match value used."""
        result = compute_edit_regions("a b", "a c", min_match=5)
        self.assertEqual(result.min_match, 5)

    def test_result_stores_original_and_edited(self):
        """Result stores the original and edited text."""
        result = compute_edit_regions("hello", "world")
        self.assertEqual(result.original_text, "hello")
        self.assertEqual(result.edited_text, "world")


class TestEditRegionRepr(unittest.TestCase):
    """Tests for EditRegion string representation."""

    def test_repr_substitution(self):
        """Substitution repr includes type and both texts."""
        edit = EditRegion(4, 7, 4, 7, "cat", "dog", "substitution")
        r = repr(edit)
        self.assertIn("substitution", r)
        self.assertIn("cat", r)
        self.assertIn("dog", r)

    def test_repr_insertion(self):
        """Insertion repr shows empty old_text."""
        edit = EditRegion(5, 5, 5, 10, "", "hello", "insertion")
        r = repr(edit)
        self.assertIn("insertion", r)
        self.assertIn("hello", r)


if __name__ == "__main__":
    unittest.main()
