#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, login


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push a trained model to Hugging Face Hub.")
    parser.add_argument(
        "--model-dir",
        default="final-model/final",
        help="Local directory containing the trained model and model card (README.md).",
    )
    # Replace with your Hugging Face username and model name
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo ID (e.g. your-username/your-model-name).")
    parser.add_argument("--private", action="store_true", help="Create repo as private.")
    return parser.parse_args()


def push_folder(api: HfApi, local_dir: Path, repo_id: str, private: bool) -> None:
    if not local_dir.exists():
        raise FileNotFoundError(f"Missing local directory: {local_dir}")
    readme = local_dir / "README.md"
    if not readme.exists():
        raise FileNotFoundError(
            f"Missing model card: {readme}. The Hugging Face repo needs a README.md with pipeline_tag."
        )

    print(f"Creating repo: {repo_id}")
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)

    print(f"Uploading {local_dir} -> {repo_id}")
    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id)
    print(f"Done: {repo_id}")


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token)

    api = HfApi(token=token)
    push_folder(api, Path(args.model_dir), args.repo_id, args.private)


if __name__ == "__main__":
    main()
