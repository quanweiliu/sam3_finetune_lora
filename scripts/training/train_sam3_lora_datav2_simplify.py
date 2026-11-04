#!/usr/bin/env python3
"""SAM3 LoRA Training Script

VPRef v0.3 (Vaihingen, 480px, union masks):
  python scripts/training/train_sam3_lora_datav2.py --config configs/vpref_v03_expansion_lora_config.yaml --check-data
  python scripts/training/train_sam3_lora_datav2.py --config configs/vpref_v03_expansion_lora_config.yaml --device 0

Validation Strategy (Following SAM3):
  - During training: Only compute validation LOSS (fast, no metrics)
  - After training: Run validate_sam3_lora.py for full metrics (mAP, cgF1) with NMS

This approach significantly speeds up training by avoiding expensive metric computation
during each epoch, while still monitoring overfitting via validation loss.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
from sam3Trainer import SAM3TrainerNative


def launch_distributed_training(args):
    """Launch training with multiple GPUs using torchrun subprocess."""
    import subprocess
    import sys

    devices = args.device
    num_gpus = len(devices)
    device_str = ",".join(map(str, devices))

    print(f"Launching distributed training on GPUs: {devices}")
    print(f"Number of processes: {num_gpus}")

    # Build the command
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "--master_port",
        str(args.master_port),
        sys.argv[0],  # This script
        "--config",
        args.config,
        "--device",
        *map(str, devices),
        "--_launched_by_torchrun",  # Internal flag to indicate we're in subprocess
    ]
    if args.max_steps is not None:
        cmd.extend(["--max-steps", str(args.max_steps)])

    # Set environment variable for visible devices
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_str

    # Run the subprocess
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SAM3 with LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""""",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/full_lora_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--device",
        type=int,
        nargs="+",
        default=[0],
        help="GPU device ID(s) to use. Single value for single GPU, multiple values for multi-GPU. "
        "Example: --device 0 (single GPU), --device 0 1 2 (3 GPUs)",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Master port for distributed training (default: 29500)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set automatically by torchrun)",
    )
    parser.add_argument(
        "--_launched_by_torchrun",
        action="store_true",
        help=argparse.SUPPRESS,  # Hidden argument for internal use
    )
    parser.add_argument("--max-steps", type=int,
                        help="Stop after this many optimizer steps (e.g. 1 for a smoke test)")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")

    # Determine if multi-GPU training is requested
    num_devices = len(args.device)
    is_torchrun_subprocess = args._launched_by_torchrun or "LOCAL_RANK" in os.environ

    if num_devices > 1 and not is_torchrun_subprocess:
        # Multi-GPU requested but not yet in torchrun - launch it
        launch_distributed_training(args)
    else:
        # Single GPU or already in torchrun subprocess
        multi_gpu = num_devices > 1 and is_torchrun_subprocess

        if not multi_gpu and num_devices == 1:
            # Single GPU mode - set the device
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[0])
            print(f"Using single GPU: {args.device[0]}")

        trainer = SAM3TrainerNative(args.config, multi_gpu=multi_gpu)
        trainer.train(max_steps=args.max_steps)


"""
Multi-GPU Training:
  Single GPU:
    python scripts/training/train_sam3_lora_datav2_simplify.py  --config configs/vpref_v07_lora_config.yaml  --device 0
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_sam3_lora_datav2_simplify.py  --config configs/vpref_v08_lora_config.yaml  --device 1

  Multi-GPU (DDP):
    torchrun --nproc_per_node=2 train_sam3_lora_datav2_simplify.py --config configs/vpref_v08_lora_config.yaml --multi-gpu

  Multi-GPU with specific GPUs:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_sam3_lora_datav2_simplify.py --config configs/vpref_v08_lora_config.yaml --multi-gpu
    torchrun --nproc_per_node 2 scripts/training/train_sam3_lora_datav2_simplify.py --config configs/vpref_v08_lora_config.yaml --device 0 1
"""