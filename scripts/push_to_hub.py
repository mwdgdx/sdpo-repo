#!/usr/bin/env python3
"""
Push trained model checkpoint to Hugging Face Hub.

Usage:
    python scripts/push_to_hub.py \
        --checkpoint_path /path/to/checkpoint \
        --repo_id your-username/model-name \
        --token YOUR_HF_TOKEN

Environment Variables (alternative to command line args):
    HF_TOKEN    - Hugging Face token
    HF_REPO_ID  - Repository ID (e.g., username/model-name)
"""

import argparse
import os
import sys
from pathlib import Path


def push_to_hub(
    checkpoint_path: str,
    repo_id: str,
    token: str = None,
    private: bool = False,
    commit_message: str = None,
):
    """Push checkpoint to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi, upload_folder
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: Please install huggingface_hub and transformers:")
        print("  pip install huggingface_hub transformers")
        sys.exit(1)
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)
    
    # Get token from environment if not provided
    if token is None:
        token = os.environ.get("HF_TOKEN")
    
    if token is None:
        print("ERROR: No Hugging Face token provided.")
        print("Set HF_TOKEN environment variable or use --token argument")
        sys.exit(1)
    
    print(f"Pushing checkpoint to: {repo_id}")
    print(f"Checkpoint path: {checkpoint_path}")
    
    api = HfApi(token=token)
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        print(f"Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Warning: Could not create repo: {e}")
    
    # Check if this is a transformers-compatible checkpoint or raw checkpoint
    has_config = (checkpoint_path / "config.json").exists()
    
    if has_config:
        # Load and push using transformers
        print("Loading model with transformers...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
            )
            
            print("Pushing to Hub...")
            model.push_to_hub(
                repo_id,
                token=token,
                private=private,
                commit_message=commit_message or f"Upload checkpoint from {checkpoint_path.name}",
            )
            tokenizer.push_to_hub(
                repo_id,
                token=token,
                private=private,
            )
        except Exception as e:
            print(f"Error loading with transformers: {e}")
            print("Falling back to raw upload...")
            upload_folder(
                folder_path=str(checkpoint_path),
                repo_id=repo_id,
                token=token,
                commit_message=commit_message or f"Upload checkpoint from {checkpoint_path.name}",
            )
    else:
        # Raw upload
        print("Uploading raw checkpoint files...")
        upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=repo_id,
            token=token,
            commit_message=commit_message or f"Upload checkpoint from {checkpoint_path.name}",
        )
    
    print(f"\nSuccess! Model uploaded to: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Push model checkpoint to Hugging Face Hub")
    parser.add_argument(
        "--checkpoint_path", "-c",
        type=str,
        required=True,
        help="Path to the checkpoint directory"
    )
    parser.add_argument(
        "--repo_id", "-r",
        type=str,
        default=os.environ.get("HF_REPO_ID"),
        help="Hugging Face repo ID (e.g., username/model-name). Can also use HF_REPO_ID env var."
    )
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=None,
        help="Hugging Face token. Can also use HF_TOKEN env var."
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--commit_message", "-m",
        type=str,
        default=None,
        help="Commit message for the upload"
    )
    
    args = parser.parse_args()
    
    if args.repo_id is None:
        print("ERROR: --repo_id is required or set HF_REPO_ID environment variable")
        sys.exit(1)
    
    push_to_hub(
        checkpoint_path=args.checkpoint_path,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
