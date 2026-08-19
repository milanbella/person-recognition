from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from model_training.api import ModelTrainingService
from model_training.clients import LiveServiceClient, ShopCatalogClient


DEFAULT_STATE_ROOT = Path("/var/lib/person-recognition/model-training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Product-model collection and review service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8004)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--api-token", required=True)
    parser.add_argument("--live-base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--live-operator-token", required=True)
    parser.add_argument("--browser-stream-base-url", default="")
    parser.add_argument("--shop-api-base-url", required=True)
    parser.add_argument("--shop-api-key", required=True)
    parser.add_argument("--shop-id", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = ModelTrainingService(
        state_root=args.state_root,
        api_token=args.api_token,
        shop_id=args.shop_id,
        live_client=LiveServiceClient(args.live_base_url, args.live_operator_token),
        catalog_client=ShopCatalogClient(args.shop_api_base_url, args.shop_api_key, args.shop_id),
        assets_root=Path(__file__).resolve().parent / "model_training_ui",
        browser_stream_base_url=args.browser_stream_base_url,
    )
    uvicorn.run(service.app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
