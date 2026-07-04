"""Download and cache Hugging Face models needed by the project."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.utils import clear_dead_proxy_env


def download_repo(repo_id: str, local_dir: Path, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token or None,
        resume_download=True,
    )


def main() -> None:
    settings = get_settings()
    clear_dead_proxy_env()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    targets = [
        (settings.embedding_model_name, settings.embedding_model_dir),
        (settings.indic_to_en_model, settings.indic_to_en_model_dir),
        (settings.en_to_indic_model, settings.en_to_indic_model_dir),
    ]

    for repo_id, local_dir in targets:
        print(f"Downloading {repo_id} -> {local_dir}")
        try:
            download_repo(repo_id, local_dir, settings.hf_hub_token)
        except Exception as exc:
            message = str(exc)
            if "Please enable access to public gated repositories" in message:
                print(
                    f"Skipped {repo_id}: your Hugging Face token needs the "
                    "fine-grained permission for public gated repositories."
                )
            else:
                print(f"Skipped {repo_id}: {message}")

    print("All model downloads completed.")


if __name__ == "__main__":
    main()
