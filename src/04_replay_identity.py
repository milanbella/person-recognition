from pipeline.identity import build_identity_match_argparser, run_identity_pipeline


def main() -> None:
    args = build_identity_match_argparser(
        description="Replay entrypoint: assign local identities from saved event embeddings."
    ).parse_args()
    run_identity_pipeline(
        embedding_runs_dir=args.embedding_runs_dir,
        output_dir=args.output_dir,
        match_threshold=args.match_threshold,
        min_face_count=args.min_face_count,
    )


if __name__ == "__main__":
    main()
