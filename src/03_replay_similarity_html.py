from pipeline.review import build_similarity_html_argparser, write_similarity_html


def main() -> None:
    args = build_similarity_html_argparser(
        description="Replay entrypoint: render HTML for saved embedding similarity review."
    ).parse_args()
    write_similarity_html(args.embedding_runs_dir, args.evidence_dir, args.top_k)


if __name__ == "__main__":
    main()
