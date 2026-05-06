from pipeline.embedding import build_embedding_argparser, run_embedding_pipeline


def main() -> None:
    args = build_embedding_argparser(
        description="Replay entrypoint: generate face embeddings from saved evidence artifacts."
    ).parse_args()
    run_embedding_pipeline(
        evidence_dir=args.evidence_dir,
        output_dir=args.output_dir,
        cache_root=args.cache_root,
        model_pack=args.model_pack,
        det_size=(args.det_width, args.det_height),
        det_thresh=args.det_thresh,
        max_events=args.max_events,
    )


if __name__ == "__main__":
    main()
