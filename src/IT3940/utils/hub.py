from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, login
import os
from dotenv import load_dotenv


load_dotenv()


def upload_checkpoint(
    local_path: str | Path,
    repo_id: str,
    path_in_repo: str | None = None,
) -> str:
    """Upload a checkpoint to the Hugging Face Hub."""
    local_path = Path(local_path)
    path_in_repo = path_in_repo or local_path.name

    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    api = HfApi()
    
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {path_in_repo}",
    )

    return f"https://huggingface.co/{repo_id}"


def download_checkpoint(
    repo_id: str,
    filename: str,
    revision: str | None = None,
) -> Path:
    """Download a checkpoint from the Hugging Face Hub."""
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=os.getenv("HUGGINGFACE_TOKEN"),
    )
    return Path(downloaded_path)
    