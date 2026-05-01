"""
demo_inpainting.py

Gradio demo app for comparing "from" (source) and "to" (inpainted) audio projects.

Usage:
    python3 demo_inpainting.py --from_csv /path/to/from.csv --to_csv /path/to/to.csv

Each CSV must have at least these columns:
    verse_id      - e.g. "MAT 1:1" or "MAT 1:1-3"
    file_name     - path to audio file, relative to the CSV file's directory
    transcription - the text for that verse
"""

import argparse
import os
import sys
import pandas as pd
import gradio as gr

os.environ.setdefault('TMPDIR', './temps')


# ---------------------------------------------------------------------------
# Verse ID parsing
# ---------------------------------------------------------------------------

def parse_verse_id(verse_id: str):
    """
    Parse a verse_id string into a (book, chapter, verse_start, verse_end) tuple.

    Examples:
        "MAT 1:1"     -> ("MAT", 1, 1, 1)
        "MAT 1:1-3"   -> ("MAT", 1, 1, 3)
        "1CO 3:10-11" -> ("1CO", 3, 10, 11)
    """
    try:
        parts = verse_id.strip().split(' ', 1)
        book = parts[0]
        chapter_verse = parts[1]
        chapter_str, verse_part = chapter_verse.split(':', 1)
        chapter = int(chapter_str)
        if '-' in verse_part:
            v_start_str, v_end_str = verse_part.split('-', 1)
            verse_start = int(v_start_str)
            verse_end = int(v_end_str)
        else:
            verse_start = int(verse_part)
            verse_end = verse_start
        return book, chapter, verse_start, verse_end
    except Exception:
        return None, None, None, None


def sort_key(verse_id: str):
    """Return a sortable tuple for a verse_id."""
    book, chapter, verse_start, verse_end = parse_verse_id(verse_id)
    if book is None:
        return ('', 0, 0, 0)
    return (book, chapter, verse_start, verse_end)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(csv_path: str) -> pd.DataFrame:
    """Load a CSV and add resolved absolute audio paths."""
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    df = pd.read_csv(csv_path)

    required = {'verse_id', 'file_name', 'transcription'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV '{csv_path}' is missing columns: {missing}")

    # Resolve audio paths relative to the CSV directory
    def resolve_path(fn):
        if pd.isna(fn) or str(fn).strip() == '':
            return None
        p = os.path.join(csv_dir, str(fn))
        return p if os.path.exists(p) else None

    df['audio_path'] = df['file_name'].apply(resolve_path)

    # Parse verse components
    parsed = df['verse_id'].apply(parse_verse_id)
    df['book'] = parsed.apply(lambda x: x[0])
    df['chapter'] = parsed.apply(lambda x: x[1])
    df['verse_start'] = parsed.apply(lambda x: x[2])
    df['verse_end'] = parsed.apply(lambda x: x[3])

    return df


def build_chapter_index(from_df: pd.DataFrame, to_df: pd.DataFrame):
    """
    Build a sorted list of (book, chapter) pairs present in either CSV,
    and a dict mapping verse_id -> row for each CSV.
    """
    from_lookup = {row['verse_id']: row for _, row in from_df.iterrows()}
    to_lookup = {row['verse_id']: row for _, row in to_df.iterrows()}

    all_verse_ids = set(from_lookup.keys()) | set(to_lookup.keys())

    chapters = set()
    for vid in all_verse_ids:
        b2, c2, _, _ = parse_verse_id(str(vid))
        if b2 is not None:
            chapters.add((b2, c2))

    # Sort chapters preserving book order from from_df first, then to_df
    book_order: dict = {}
    for df in [from_df, to_df]:
        for book in df['book'].dropna():
            if book not in book_order:
                book_order[book] = len(book_order)

    sorted_chapters = sorted(chapters, key=lambda bc: (book_order.get(bc[0], 9999), bc[1]))

    return sorted_chapters, from_lookup, to_lookup


def get_verses_for_chapter(book: str, chapter: int,
                            from_lookup: dict, to_lookup: dict):
    """
    Return a sorted list of (verse_id, from_row_or_None, to_row_or_None)
    for all verse_ids belonging to the given book/chapter.
    """
    all_verse_ids = set()
    for vid in list(from_lookup.keys()) + list(to_lookup.keys()):
        b, c, _, _ = parse_verse_id(str(vid))
        if b == book and c == chapter:
            all_verse_ids.add(vid)

    sorted_ids = sorted(all_verse_ids, key=sort_key)

    rows = []
    for vid in sorted_ids:
        from_row = from_lookup.get(vid)
        to_row = to_lookup.get(vid)
        rows.append((vid, from_row, to_row))
    return rows


# ---------------------------------------------------------------------------
# Gradio UI builder
# ---------------------------------------------------------------------------

def build_app(from_csv: str, to_csv: str):
    from_df = load_csv(from_csv)
    to_df = load_csv(to_csv)

    # Collect all directories that contain audio files so Gradio can serve them
    audio_dirs: set = set()
    for df in [from_df, to_df]:
        for p in df['audio_path'].dropna():
            audio_dirs.add(os.path.dirname(os.path.abspath(str(p))))
    # Also add the CSV directories themselves
    audio_dirs.add(os.path.dirname(os.path.abspath(from_csv)))
    audio_dirs.add(os.path.dirname(os.path.abspath(to_csv)))
    allowed_paths = sorted(audio_dirs)

    sorted_chapters, from_lookup, to_lookup = build_chapter_index(from_df, to_df)

    # Build human-readable chapter labels like "MAT 1", "MAT 2", ...
    chapter_labels = [f"{b} {c}" for b, c in sorted_chapters]
    chapter_label_to_bc = {label: bc for label, bc in zip(chapter_labels, sorted_chapters)}

    # Collect unique books for the book dropdown (in order)
    book_list: list = []
    seen_books: set = set()
    for b, _ in sorted_chapters:
        if b not in seen_books:
            book_list.append(b)
            seen_books.add(b)

    def chapters_for_book(book: str):
        return [f"{b} {c}" for b, c in sorted_chapters if b == book]

    # Determine the initial chapter label
    initial_chapter_label = chapters_for_book(book_list[0])[0] if book_list else ""

    def render_chapter(chapter_label: str):
        """Return verse rows for the given chapter label."""
        if not chapter_label or chapter_label not in chapter_label_to_bc:
            return []
        book, chapter = chapter_label_to_bc[chapter_label]
        return get_verses_for_chapter(book, chapter, from_lookup, to_lookup)

    # -----------------------------------------------------------------------
    # Fixed pool of verse row components.
    # Gradio doesn't support truly dynamic component counts, so we pre-allocate
    # MAX_VERSE_ROWS rows and show/hide them as needed.
    # -----------------------------------------------------------------------
    MAX_VERSE_ROWS = 200  # upper bound on verses per chapter

    with gr.Blocks(title="Audio Inpainting Demo") as demo:
        gr.Markdown("# 🎙️ Audio Inpainting Demo")
        gr.Markdown(
            f"**From:** `{os.path.abspath(from_csv)}`  \n"
            f"**To:** `{os.path.abspath(to_csv)}`"
        )

        # ---- Navigation row ------------------------------------------------
        with gr.Row():
            book_dd = gr.Dropdown(
                label="Book",
                choices=book_list,
                value=book_list[0] if book_list else None,
                scale=2,
            )
            chapter_dd = gr.Dropdown(
                label="Chapter",
                choices=chapters_for_book(book_list[0]) if book_list else [],
                value=initial_chapter_label if initial_chapter_label else None,
                scale=2,
            )
            prev_btn = gr.Button("◀ Prev Chapter", scale=1)
            next_btn = gr.Button("Next Chapter ▶", scale=1)

        # ---- Chapter info --------------------------------------------------
        chapter_info = gr.Markdown("")

        # ---- Column headers ------------------------------------------------
        with gr.Row():
            gr.Markdown("### Verse")
            gr.Markdown("### Source Text")
            gr.Markdown("### Source Audio")
            gr.Markdown("### Inpainted Text")
            gr.Markdown("### Inpainted Audio")

        # ---- Verse row pool ------------------------------------------------
        verse_label_components = []
        from_text_components = []
        from_audio_components = []
        to_text_components = []
        to_audio_components = []
        row_groups = []

        for _ in range(MAX_VERSE_ROWS):
            with gr.Row(visible=False) as row_grp:
                v_lbl = gr.Markdown("")
                f_txt = gr.Textbox(
                    label="", lines=3, interactive=False,
                    show_label=False, scale=3
                )
                f_aud = gr.Audio(
                    label="", type="filepath", interactive=False,
                    show_label=False, scale=2
                )
                t_txt = gr.Textbox(
                    label="", lines=3, interactive=False,
                    show_label=False, scale=3
                )
                t_aud = gr.Audio(
                    label="", type="filepath", interactive=False,
                    show_label=False, scale=2
                )

            row_groups.append(row_grp)
            verse_label_components.append(v_lbl)
            from_text_components.append(f_txt)
            from_audio_components.append(f_aud)
            to_text_components.append(t_txt)
            to_audio_components.append(t_aud)

        # ---- State ---------------------------------------------------------
        current_chapter_state = gr.State(initial_chapter_label)

        # ---- All outputs list (order must match update_display return) ------
        all_outputs = (
            [chapter_info]
            + row_groups
            + verse_label_components
            + from_text_components
            + from_audio_components
            + to_text_components
            + to_audio_components
        )

        # ---- Callbacks -----------------------------------------------------

        def on_book_change(book):
            chaps = chapters_for_book(book)
            new_val = chaps[0] if chaps else None
            return gr.update(choices=chaps, value=new_val), new_val

        def on_chapter_change(chapter_label):
            return chapter_label  # just update state

        def update_display(chapter_label: str):
            """
            Returns a flat list of gr.update() values for all_outputs:
              [chapter_info, *row_groups, *verse_labels,
               *from_texts, *from_audios, *to_texts, *to_audios]
            """
            verse_rows = render_chapter(chapter_label)
            n = len(verse_rows)

            info_updates = [gr.update(value=f"**{chapter_label}** — {n} verse(s)")]
            row_updates = []
            label_updates = []
            from_text_updates = []
            from_audio_updates = []
            to_text_updates = []
            to_audio_updates = []

            for i in range(MAX_VERSE_ROWS):
                if i < n:
                    vid, from_row, to_row = verse_rows[i]
                    row_updates.append(gr.update(visible=True))
                    label_updates.append(gr.update(value=f"**{vid}**"))
                    from_text_updates.append(
                        gr.update(value=from_row['transcription'] if from_row is not None else "")
                    )
                    from_audio_updates.append(
                        gr.update(value=from_row['audio_path'] if from_row is not None else None)
                    )
                    to_text_updates.append(
                        gr.update(value=to_row['transcription'] if to_row is not None else "")
                    )
                    to_audio_updates.append(
                        gr.update(value=to_row['audio_path'] if to_row is not None else None)
                    )
                else:
                    row_updates.append(gr.update(visible=False))
                    label_updates.append(gr.update(value=""))
                    from_text_updates.append(gr.update(value=""))
                    from_audio_updates.append(gr.update(value=None))
                    to_text_updates.append(gr.update(value=""))
                    to_audio_updates.append(gr.update(value=None))

            return (
                info_updates
                + row_updates
                + label_updates
                + from_text_updates
                + from_audio_updates
                + to_text_updates
                + to_audio_updates
            )

        def prev_chapter(current_label: str):
            if not current_label or current_label not in chapter_labels:
                return gr.update(), current_label
            idx = chapter_labels.index(current_label)
            new_label = chapter_labels[max(0, idx - 1)]
            return gr.update(value=new_label), new_label

        def next_chapter(current_label: str):
            if not current_label or current_label not in chapter_labels:
                return gr.update(), current_label
            idx = chapter_labels.index(current_label)
            new_label = chapter_labels[min(len(chapter_labels) - 1, idx + 1)]
            return gr.update(value=new_label), new_label

        # Wire book dropdown -> chapter dropdown + state
        book_dd.change(
            fn=on_book_change,
            inputs=[book_dd],
            outputs=[chapter_dd, current_chapter_state],
        ).then(
            fn=update_display,
            inputs=[current_chapter_state],
            outputs=all_outputs,
        )

        # Wire chapter dropdown -> state -> display
        chapter_dd.change(
            fn=on_chapter_change,
            inputs=[chapter_dd],
            outputs=[current_chapter_state],
        ).then(
            fn=update_display,
            inputs=[current_chapter_state],
            outputs=all_outputs,
        )

        # Wire prev/next buttons
        prev_btn.click(
            fn=prev_chapter,
            inputs=[current_chapter_state],
            outputs=[chapter_dd, current_chapter_state],
        ).then(
            fn=update_display,
            inputs=[current_chapter_state],
            outputs=all_outputs,
        )

        next_btn.click(
            fn=next_chapter,
            inputs=[current_chapter_state],
            outputs=[chapter_dd, current_chapter_state],
        ).then(
            fn=update_display,
            inputs=[current_chapter_state],
            outputs=all_outputs,
        )

        # Initial page load — use a closure so the label is baked in
        def initial_load():
            return update_display(initial_chapter_label)

        demo.load(
            fn=initial_load,
            inputs=None,
            outputs=all_outputs,
        )

    # Attach allowed_paths so main() can pass them to launch()
    demo._allowed_paths_for_launch = allowed_paths  # type: ignore[attr-defined]
    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gradio demo for comparing source vs inpainted audio projects."
    )
    parser.add_argument(
        '--from_csv',
        required=True,
        help='Path to the source (from) CSV file with verse_id, file_name, transcription columns.'
    )
    parser.add_argument(
        '--to_csv',
        required=True,
        help='Path to the target (to/inpainted) CSV file with verse_id, file_name, transcription columns.'
    )
    parser.add_argument(
        '--share', action='store_true',
        help='Create a public Gradio share link.'
    )
    parser.add_argument(
        '--port', type=int, default=7860,
        help='Port to run the Gradio server on.'
    )

    # parse_known_args avoids conflicts with any args Gradio injects
    args, _ = parser.parse_known_args()

    if not os.path.isfile(args.from_csv):
        print(f"ERROR: --from_csv file not found: {args.from_csv}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.to_csv):
        print(f"ERROR: --to_csv file not found: {args.to_csv}", file=sys.stderr)
        sys.exit(1)

    demo = build_app(args.from_csv, args.to_csv)
    allowed = getattr(demo, '_allowed_paths_for_launch', [])
    print(f"Allowing Gradio to serve files from: {allowed}")

    demo.queue()
    demo.launch(
        server_port=args.port,
        share=args.share,
        debug=False,
        show_api=False,
        allowed_paths=allowed,
    )


if __name__ == '__main__':
    main()
