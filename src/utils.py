# -*- coding: utf-8 -*-
"""
Shared utilities: hyperparameter naming, HuggingFace Hub checkpoint management, and resume logic.
"""

import os
import json
from pathlib import Path
from loguru import logger

HF_TOKEN = os.environ.get("HF_TOKEN", None)
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)
HF_USERNAME = os.environ.get("HF_USERNAME", "your-hf-username")


def build_run_name(stage: str, **kwargs) -> str:
    """
    Build a deterministic run name from hyperparameters.
    Example: sft-qwen2.5-3b-lr2e5-r16-a32-bs4-ep3
    """
    parts = [stage]

    model = kwargs.get("model", "qwen2.5-3b")
    model_short = model.split("/")[-1].lower().replace("-instruct", "")
    parts.append(model_short)

    if "lr" in kwargs:
        lr_str = f"{float(kwargs['lr']):.0e}".replace("+", "").replace("0", "")
        parts.append(f"lr{lr_str}")
    if "lora_rank" in kwargs:
        parts.append(f"r{kwargs['lora_rank']}")
    if "lora_alpha" in kwargs:
        parts.append(f"a{kwargs['lora_alpha']}")
    if "batch_size" in kwargs:
        parts.append(f"bs{kwargs['batch_size']}")
    if "epochs" in kwargs:
        parts.append(f"ep{kwargs['epochs']}")
    if "max_length" in kwargs:
        parts.append(f"ml{kwargs['max_length']}")

    return "-".join(str(p) for p in parts)


def get_hf_repo_id(run_name: str) -> str:
    """Build HuggingFace repo ID from run name."""
    return f"{HF_USERNAME}/{run_name}"


def check_hf_checkpoint_exists(repo_id: str) -> bool:
    """Check if a checkpoint exists on HuggingFace Hub."""
    try:
        from huggingface_hub import repo_exists
        return repo_exists(repo_id, token=HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not check HF Hub for {repo_id}: {e}")
        return False


def get_latest_checkpoint(output_dir: str) -> str | None:
    """Get the latest checkpoint directory from local output_dir."""
    if not os.path.exists(output_dir):
        return None

    checkpoints = []
    for d in os.listdir(output_dir):
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d)):
            try:
                step = int(d.split("-")[-1])
                checkpoints.append((step, os.path.join(output_dir, d)))
            except ValueError:
                continue

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def download_checkpoint_from_hub(repo_id: str, local_dir: str) -> str | None:
    """Download checkpoint from HuggingFace Hub if it exists."""
    if not check_hf_checkpoint_exists(repo_id):
        logger.info(f"No checkpoint found on HF Hub: {repo_id}")
        return None

    try:
        from huggingface_hub import snapshot_download
        logger.info(f"Downloading checkpoint from HF Hub: {repo_id} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=HF_TOKEN,
        )
        logger.info(f"Checkpoint downloaded to {local_dir}")
        return local_dir
    except Exception as e:
        logger.warning(f"Failed to download checkpoint from {repo_id}: {e}")
        return None


def upload_checkpoint_to_hub(local_dir: str, repo_id: str, commit_message: str = "Upload checkpoint"):
    """Upload a checkpoint directory to HuggingFace Hub."""
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not set, skipping upload to HuggingFace Hub")
        return

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)

        # Create repo if it doesn't exist
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)

        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        logger.info(f"Checkpoint uploaded to HF Hub: {repo_id}")
    except Exception as e:
        logger.warning(f"Failed to upload checkpoint to {repo_id}: {e}")


def resolve_resume_checkpoint(output_dir: str, repo_id: str) -> str | None:
    """
    Resume logic: check local first, then HuggingFace Hub.
    Returns checkpoint path or None.
    """
    # 1. Check local
    local_ckpt = get_latest_checkpoint(output_dir)
    if local_ckpt:
        logger.info(f"Resuming from local checkpoint: {local_ckpt}")
        return local_ckpt

    # 2. Check HuggingFace Hub
    hub_local_dir = os.path.join(output_dir, "hub-checkpoint")
    downloaded = download_checkpoint_from_hub(repo_id, hub_local_dir)
    if downloaded:
        logger.info(f"Resuming from HF Hub checkpoint: {downloaded}")
        return downloaded

    logger.info("No checkpoint found, starting from scratch")
    return None


def setup_wandb(project_name: str, run_name: str):
    """Setup wandb if API key is available."""
    if WANDB_API_KEY:
        import wandb
        wandb.login(key=WANDB_API_KEY)
        logger.info(f"wandb initialized: project={project_name}, run={run_name}")
    else:
        logger.warning("WANDB_API_KEY not set, wandb logging disabled")
        os.environ["WANDB_DISABLED"] = "true"


def save_hyperparams(output_dir: str, **kwargs):
    """Save hyperparameters to a JSON file for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    hp_path = os.path.join(output_dir, "hyperparams.json")
    with open(hp_path, "w") as f:
        json.dump(kwargs, f, indent=2, ensure_ascii=False)
    logger.info(f"Hyperparameters saved to {hp_path}")
