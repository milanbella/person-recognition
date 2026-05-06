from pipeline.review import build_similarity_review_argparser, run_similarity_review


def main() -> None:
    args = build_similarity_review_argparser(
        description="Replay entrypoint: review cosine similarities between saved event embeddings."
    ).parse_args()
    run_similarity_review(args.embedding_runs_dir, args.top_k)


if __name__ == "__main__":
    main()
