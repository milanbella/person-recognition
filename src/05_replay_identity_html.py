from pipeline.identity import build_identity_html_argparser, write_identity_html


def main() -> None:
    args = build_identity_html_argparser(
        description="Replay entrypoint: render HTML for saved local identity assignments."
    ).parse_args()
    write_identity_html(args.identity_runs_dir, args.embedding_runs_dir, args.evidence_dir)


if __name__ == "__main__":
    main()
