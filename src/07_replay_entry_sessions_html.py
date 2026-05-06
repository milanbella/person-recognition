from pipeline.entry_session import build_entry_session_html_argparser, write_entry_session_html


def main() -> None:
    args = build_entry_session_html_argparser(
        description="Replay entrypoint: render HTML for saved entry-session artifacts."
    ).parse_args()
    write_entry_session_html(args.output_root)


if __name__ == "__main__":
    main()
