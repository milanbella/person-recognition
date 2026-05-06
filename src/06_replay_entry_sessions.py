from pipeline.entry_session import build_entry_session_argparser, run_entry_session_pipeline


def main() -> None:
    args = build_entry_session_argparser(
        description="Replay entrypoint: build typed entry-event and entry-session artifacts."
    ).parse_args()
    run_entry_session_pipeline(
        shop_id=args.shop_id,
        camera_id=args.camera_id,
        camera_map_json=args.camera_map_json,
        evidence_dir=args.evidence_dir,
        embedding_runs_dir=args.embedding_runs_dir,
        output_root=args.output_root,
        merge_window_seconds=args.merge_window_seconds,
        line_axis=args.line_axis,
        line_position=args.line_position,
        min_face_similarity=args.min_face_similarity,
        ambiguity_face_similarity=args.ambiguity_face_similarity,
        min_same_camera_similarity=args.min_same_camera_similarity,
    )


if __name__ == "__main__":
    main()
