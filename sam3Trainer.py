import yaml
import math
import json
import hashlib
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.distributed as dist

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from sam3.model.model_misc import SAM3Output

from lora_layers import (
    LoRAConfig,
    apply_lora_to_model,
    count_parameters,
    save_lora_weights,
)

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api
# from sam3.train.data.sam3_image_dataset import (
#     Datapoint,
#     FindQueryLoaded,
#     Image,
#     InferenceMetadata,
#     Object,
# )
from sam3.train.loss.loss_fns import CORE_LOSS_KEY, Boxes, IABCEMdetr, Masks
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.masks_ops import rle_encode  # For encoding masks to RLE format
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from scripts.data.vpref_v03_dataset import VPRefV03Dataset, collate_vpref
from scripts.tools.utils import setup_distributed, cleanup_distributed, patch_dynamic_rope
from scripts.tools.utils import is_main_process, get_world_size, get_rank, print_rank0


def build_vpref_dataset(config, split="train"):
    """Keep the data-check command and training on the same selection/configuration."""
    training = config["training"]
    if split not in ("train", "val"):
        raise ValueError("Expected train or val split")
    selection = training
    if split == "val":
        selection = training.get("validation")
        if selection is None or selection.get("enabled", True) is False:
            return None
        if not selection.get("manifest"):
            raise ValueError("training.validation.manifest is required")
    return VPRefV03Dataset(
        data_dir=selection.get("data_dir", training["data_dir"]),
        manifest=selection.get("manifest", "train.jsonl"),
        domain=training.get("domain", "Vaihingen"),
        resolution=training.get("resolution", 480),
        target_mode=training.get("target_mode", "union"),
        rotation_degrees=training.get("rotation_degrees", [0]) if split == "train" else [0],
        allow_unreviewed_proposals=selection.get("allow_unreviewed_proposals", False),
        split=split,
        task_families=selection.get("task_families", training.get("task_families")),
    )


def vpref_selection_metadata(dataset):
    
    return dict(
        manifest=str(dataset.manifest_path),
        manifest_sha256=hashlib.sha256(dataset.manifest_path.read_bytes()).hexdigest(),
        split=dataset.split, domain=dataset.domain, query_count=len(dataset),
        image_count=len(dataset.image_ids), resolution=dataset.resolution, target_mode="union",
        rotation_degrees=list(dataset.rotation_degrees),
        task_families=list(dataset.task_families) if dataset.task_families else None,
        task_family_counts=dataset.task_family_counts,
        # sample_ids=[r["sample_id"] for r in dataset.samples],
        source_scenes=sorted({r["source_scene"] for r in dataset.samples}),
        uses_unreviewed_proposals=dataset.uses_unreviewed_proposals,
        annotation_status="pending_human_review" if dataset.uses_unreviewed_proposals else "accepted",
    )

def vpref_model_parameters_metadata(model):
    """Return a dict of model parameters and their SHA256 hashes."""

    stats = count_parameters(model)
    total = stats['total_parameters']
    trainable = stats['trainable_parameters']
    frozen = total - trainable
    print_rank0("=" * 50)
    print_rank0("  Parameter Statistics")
    print_rank0("=" * 50)
    print_rank0(f"  Total parameters:      {total / 1e6:>10.2f}M")
    print_rank0(f"  Trainable parameters:  {trainable / 1e6:>10.2f}M  ({stats['trainable_percentage']:.2f}%)")
    print_rank0(f"  Frozen parameters:     {frozen / 1e6:>10.2f}M  ({100 - stats['trainable_percentage']:.2f}%)")
    print_rank0("=" * 50)

    return dict(
        total_parameters=total,
        trainable_parameters=trainable,
        frozen_parameters=frozen,
    )


def attach_vpref_valid_pixels(targets, batch_dict, device):
    if "vpref_valid_pixels" in batch_dict:
        valid = batch_dict["vpref_valid_pixels"].to(device)
        for target in targets:
            target["vpref_valid_pixels"] = valid


class VPRefMasks(Masks):
    """The original mask loss with optional per-query void pixels, including O2M."""
    def get_loss(self, outputs, targets, indices, num_boxes):
        if "vpref_valid_pixels" not in targets:
            return super().get_loss(outputs, targets, indices, num_boxes)
        if self.num_sample_points is not None or outputs.get("is_video_grounding_batch", False):
            raise ValueError("VPRef pixel-ignore supports dense image masks only")
        source = outputs["pred_masks"][(indices[0], indices[1])]
        target = targets["masks"] if indices[2] is None else targets["masks"][indices[2]]
        keep = targets["is_valid_mask"] if indices[2] is None else targets["is_valid_mask"][indices[2]]
        valid = targets["vpref_valid_pixels"][indices[0]][keep]
        source, target = source[keep], target[keep]
        if source.shape[0] == 0:
            zero = outputs["pred_masks"].sum() * 0
            return {"loss_mask": zero, "loss_dice": zero}
        source = torch.nn.functional.interpolate(source[:, None].float(), size=target.shape[-2:],
                    mode="bilinear", align_corners=False)[:, 0].flatten(1)
        target = target.to(source).flatten(1)
        valid = valid.to(source).flatten(1)
        denominator = valid.sum(1)
        if (denominator == 0).any() or (target * (1-valid)).any():
            raise ValueError("Invalid ignore support or foreground/ignore overlap")
        prob = source.sigmoid()
        ce = torch.nn.functional.binary_cross_entropy_with_logits(source, target, reduction="none")
        pt = prob * target + (1-prob) * (1-target)
        alpha = self.focal_alpha * target + (1-self.focal_alpha) * (1-target)
        focal = ((ce * (1-pt).pow(self.focal_gamma) * alpha * valid).sum(1) / denominator).sum() / num_boxes
        numerator = 2 * (prob * target * valid).sum(1)
        dice = (1-(numerator+1)/((prob*valid).sum(1)+(target*valid).sum(1)+1)).sum() / num_boxes
        return {"loss_mask": focal, "loss_dice": dice}


class SAM3TrainerNative:
    def __init__(self, config_path, multi_gpu=False):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        seed = int(self.config["training"].get("seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create output directory; the source dataset remains read-only.
        out_dir = Path(self.config["output"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.config, stream, allow_unicode=True)
        
        # Multi-GPU setup
        self.multi_gpu = multi_gpu
        self.local_rank = 0
        self.world_size = 1

        if self.multi_gpu:
            self.local_rank = setup_distributed()
            self.world_size = get_world_size()
            self.device = torch.device(f"cuda:{self.local_rank}")
            print_rank0(f"Multi-GPU training enabled with {self.world_size} GPUs")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Single-GPU training enabled on {self.device}")

        # Build Model
        print_rank0("Building SAM3 model...")
        model_config = self.config.get("model", {})
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            checkpoint_path=model_config.get("checkpoint_path", "/home/icclab/Documents/lqw/sam3/weights/sam3.pt"),
            load_from_HF=False,  # Tries to download from HF if checkpoint_path is None
            bpe_path=model_config.get("bpe_path", "/home/icclab/Documents/lqw/sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
            eval_mode=False,
        )

        # Patch RoPE for dynamic resolution support (enables non-1008 inputs)
        n_patched = patch_dynamic_rope(self.model)
        print_rank0(f"Dynamic RoPE patch applied to {n_patched} attention layers")

        # Apply LoRA
        print_rank0("Applying LoRA...")
        lora_cfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        self.model = apply_lora_to_model(self.model, lora_config)

        # ── Parameter Statistics ──
        # stats = count_parameters(self.model)
        # total = stats['total_parameters']
        # trainable = stats['trainable_parameters']
        # frozen = total - trainable
        # print_rank0("=" * 50)
        # print_rank0("  Parameter Statistics")
        # print_rank0("=" * 50)
        # print_rank0(f"  Total parameters:      {total / 1e6:>10.2f}M")
        # print_rank0(f"  Trainable parameters:  {trainable / 1e6:>10.2f}M  ({stats['trainable_percentage']:.2f}%)")
        # print_rank0(f"  Frozen parameters:     {frozen / 1e6:>10.2f}M  ({100 - stats['trainable_percentage']:.2f}%)")
        # print_rank0("=" * 50)
        

        self.model.to(self.device)

        # Wrap model with DDP if multi-GPU
        if self.multi_gpu:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,  # Frozen params (requires_grad=False) don't need this flag
            )
            print_rank0("Model wrapped with DistributedDataParallel")

        # Store reference to unwrapped model for accessing custom methods
        self._unwrapped_model = self.model.module if self.multi_gpu else self.model

        # Optimizer
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=self.config["training"]["weight_decay"],
        )

        # LR scheduler will be created in train() once dataloader length is known
        self.lr_scheduler = None

        # Matcher & Loss
        self.matcher = BinaryHungarianMatcherV2(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True
        )

        # Loss weights tuned for small-dataset LoRA fine-tuning (not massive pre-training).
        loss_fns = [
            Boxes(weight_dict={"loss_bbox": 2.0, "loss_giou": 1.0}),
            IABCEMdetr(
                pos_weight=5.0,
                weight_dict={"loss_ce": 5.0, "presence_loss": 2.0},
                pos_focal=False,
                alpha=0.25,
                gamma=2,
                use_presence=True,
                pad_n_queries=100,
            ),
            VPRefMasks(
                weight_dict={
                    "loss_mask": 5.0,
                    "loss_dice": 5.0,
                },
                focal_alpha=0.25,
                focal_gamma=2.0,
                compute_aux=False,
            ),
        ]

        # Create one-to-many matcher for auxiliary outputs
        o2m_matcher = BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4)

        # Use Sam3LossWrapper for proper loss computation
        self.loss_wrapper = Sam3LossWrapper(
            loss_fns_find=loss_fns,
            matcher=self.matcher,
            o2m_matcher=o2m_matcher,
            o2m_weight=2.0,
            use_o2m_matcher_on_o2m_aux=False,
            normalization="local",  # Use local normalization (no distributed training)
            normalize_by_valid_object_num=False,
        )

    def train(self, max_steps=None):
        # Get data directory from config
        data_dir = self.config["training"]["data_dir"]
        dataset_type = self.config["training"].get("dataset", "coco")

        # Load datasets
        print_rank0(f"\nLoading training data from {data_dir} (dataset={dataset_type})...")

        if dataset_type == "vpref_v03":
            train_ds = build_vpref_dataset(self.config)
            val_ds = build_vpref_dataset(self.config, split="val")
            has_validation = val_ds is not None
            # Never sample validation box/point prompts from ground-truth masks.
            self._unwrapped_model.num_interactive_steps_val = 0
            print_rank0(f"VPRef domain={train_ds.domain}, queries={len(train_ds)}, "
                        f"images={len(train_ds.image_ids)}, resolution={train_ds.resolution}, "
                        f"target_mode=union, rotations={train_ds.rotation_degrees}")
            
            if train_ds.uses_unreviewed_proposals:
                print_rank0("EXPERIMENT: targets are unreviewed geometric proposals; "
                            "this run does not constitute human acceptance.")
            if has_validation:
                print_rank0(f"VPRef validation: queries={len(val_ds)}, images={len(val_ds.image_ids)}, "
                            f"original split=val, rotations={val_ds.rotation_degrees}; "
                            "source scenes disjoint from train")
                if val_ds.uses_unreviewed_proposals:
                    print_rank0("Validation loss uses unreviewed development proposals, not gold test labels.")
            
            if is_main_process():
                # ── Parameter Statistics ──
                stats = vpref_model_parameters_metadata(self.model)

                # ── Data Statistics ──
                out = Path(self.config["output"]["output_dir"])
                # (out / "selected_train_samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in train_ds.samples))
                
                selection = dict(
                    **vpref_selection_metadata(train_ds),
                    has_validation=has_validation,
                    has_independent_validation=False,
                    validation_role="original_val_development_loss" if has_validation else None,
                    experiment=self.config.get("experiment", {}),
                )
                if has_validation:
                    # (out / "selected_val_samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in val_ds.samples))
                    val_selection = vpref_selection_metadata(val_ds)
                    
                    selection["validation"] = val_selection
                    (out / "validation_data.json").write_text(json.dumps(val_selection, indent=2) + "\n")
                (out / "training_data.json").write_text(json.dumps(selection, indent=2) + "\n")
                (out / "model_parameters.json").write_text(json.dumps(stats, indent=2) + "\n")

            if not has_validation:
                val_ds = None

        def collate_fn(batch):
            return collate_vpref(batch)

        # Create samplers for distributed training
        train_sampler = None
        val_sampler = None

        if self.multi_gpu:
            train_sampler = DistributedSampler(
                train_ds, num_replicas=self.world_size, rank=get_rank(), shuffle=True
            )
            if has_validation:
                val_sampler = DistributedSampler(
                    val_ds, num_replicas=self.world_size, rank=get_rank(), shuffle=False
                )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config["training"]["batch_size"],
            shuffle=(train_sampler is None),  # Only shuffle if not using sampler
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=self.config["training"].get("num_workers", 0),
            pin_memory=True,
        )

        if has_validation:
            val_loader = DataLoader(
                val_ds,
                batch_size=self.config["training"]["batch_size"],
                shuffle=False,
                sampler=val_sampler,
                collate_fn=collate_fn,
                num_workers=self.config["training"].get("num_workers", 0),
                pin_memory=True,
            )
        else:
            val_loader = None

        self.model.train()

        epochs = int(self.config["training"]["num_epochs"])
        best_val_loss = float("inf")
        print_rank0(f"Starting training for {epochs} epochs...")

        if has_validation:
            print_rank0(
                f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}"
            )
        else:
            print_rank0(f"Training samples: {len(train_ds)}")
            print_rank0("⚠️  No validation data found - training without validation")

        if self.multi_gpu:
            print_rank0(
                f"Effective batch size: {self.config['training']['batch_size']} x {self.world_size} = {self.config['training']['batch_size'] * self.world_size}"
            )

        # Helper to move BatchedDatapoint to device
        def move_to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            if isinstance(obj, list):
                return [move_to_device(x, device) for x in obj]
            if isinstance(obj, tuple):
                return tuple(move_to_device(x, device) for x in obj)
            if isinstance(obj, dict):
                return {k: move_to_device(v, device) for k, v in obj.items()}
            if hasattr(obj, "__dataclass_fields__"):
                for field in obj.__dataclass_fields__:
                    val = getattr(obj, field)
                    setattr(obj, field, move_to_device(val, device))
                return obj
            return obj

        # Create output directory
        out_dir = Path(self.config["output"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── LR Scheduler: warmup + cosine annealing ──
        total_steps = epochs * len(train_loader)
        if max_steps is not None:
            total_steps = min(total_steps, max_steps)
        print("total_steps: ", total_steps)
        warmup_steps = self.config["training"].get("warmup_steps", 500)
        warmup_steps = min(warmup_steps, total_steps // 2)  # cap at 50% of total
        print_rank0(f"LR schedule: warmup={warmup_steps} steps, total={total_steps} steps "
                     f"(~{warmup_steps / max(1, len(train_loader)):.1f} epochs)")

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

        global_step = 0
        for epoch in range(epochs):
            # Set epoch for distributed sampler (required for proper shuffling)
            if self.multi_gpu and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Track training losses for this epoch
            train_losses = []

            # Only show progress bar on rank 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=not is_main_process())
            for batch_dict in pbar:
                input_batch = batch_dict["input"]
                input_batch = move_to_device(input_batch, self.device)

                # Forward pass
                # outputs_list is SAM3Output, we need to pass the whole thing to loss_wrapper
                outputs_list = self.model(input_batch)

                # Prepare targets for loss
                # input_batch.find_targets is a list of BatchedFindTarget (one per stage)
                find_targets = [
                    self._unwrapped_model.back_convert(target)
                    for target in input_batch.find_targets
                ]
                attach_vpref_valid_pixels(find_targets, batch_dict, self.device)

                # Move targets to device
                for targets in find_targets:
                    for k, v in targets.items():
                        if isinstance(v, torch.Tensor):
                            targets[k] = v.to(self.device)

                # Add matcher indices to outputs (required by Sam3LossWrapper)
                # Use SAM3Output.iteration_mode to properly iterate over outputs
                with SAM3Output.iteration_mode(
                    outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
                ) as outputs_iter:
                    for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                        # stage_targets is a single target dict, replicate for all steps
                        stage_targets_list = [stage_targets] * len(stage_outputs)
                        for outputs, targets in zip(stage_outputs, stage_targets_list):
                            # Compute indices for main output
                            outputs["indices"] = self.matcher(outputs, targets)

                            # Also add indices to auxiliary outputs if they exist
                            if "aux_outputs" in outputs:
                                for aux_out in outputs["aux_outputs"]:
                                    aux_out["indices"] = self.matcher(aux_out, targets)

                # Compute loss using Sam3LossWrapper
                # This handles num_boxes calculation and proper weighting
                loss_dict = self.loss_wrapper(outputs_list, find_targets)

                # Extract total loss
                total_loss = loss_dict[CORE_LOSS_KEY]
                if not torch.isfinite(total_loss).all():
                    raise FloatingPointError("Nonfinite SAM3 training loss")

                # Backward
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                global_step += 1

                # Track training loss
                train_losses.append(total_loss.item())
                pbar.set_postfix({"loss": total_loss.item()})
                if max_steps is not None and global_step >= max_steps:
                    break

            # Calculate average training loss for this epoch
            avg_train_loss = (
                sum(train_losses) / len(train_losses) if train_losses else 0.0
            )
            if is_main_process():
                with (out_dir / "train_stats.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(
                        epoch=epoch + 1, global_step=global_step, train_loss=avg_train_loss,
                        learning_rate=self.optimizer.param_groups[0]["lr"],
                        has_validation=has_validation,
                    )) + "\n")

            # Validation (only compute loss - no metrics, like SAM3)
            if has_validation and val_loader is not None:
                self.model.eval()
                val_losses = []

                with torch.no_grad():
                    val_pbar = tqdm(
                        val_loader, desc="Validation", disable=not is_main_process()
                    )

                    for batch_dict in val_pbar:
                        input_batch = batch_dict["input"]
                        input_batch = move_to_device(input_batch, self.device)

                        # Forward pass
                        outputs_list = self.model(input_batch)

                        # Prepare targets
                        find_targets = [
                            self._unwrapped_model.back_convert(target)
                            for target in input_batch.find_targets
                        ]
                        attach_vpref_valid_pixels(find_targets, batch_dict, self.device)

                        # Move targets to device
                        for targets in find_targets:
                            for k, v in targets.items():
                                if isinstance(v, torch.Tensor):
                                    targets[k] = v.to(self.device)

                        # Add matcher indices to outputs (required by Sam3LossWrapper)
                        with SAM3Output.iteration_mode(
                            outputs_list,
                            iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE,
                        ) as outputs_iter:
                            for stage_outputs, stage_targets in zip(
                                outputs_iter, find_targets
                            ):
                                stage_targets_list = [stage_targets] * len(
                                    stage_outputs
                                )
                                for outputs, targets in zip(
                                    stage_outputs, stage_targets_list
                                ):
                                    outputs["indices"] = self.matcher(outputs, targets)
                                    if "aux_outputs" in outputs:
                                        for aux_out in outputs["aux_outputs"]:
                                            aux_out["indices"] = self.matcher(
                                                aux_out, targets
                                            )

                        # Compute loss using Sam3LossWrapper
                        loss_dict = self.loss_wrapper(outputs_list, find_targets)
                        total_loss = loss_dict[CORE_LOSS_KEY]
                        if not torch.isfinite(total_loss).all():
                            raise FloatingPointError("Nonfinite SAM3 validation loss")

                        val_losses.append(total_loss.item())
                        val_pbar.set_postfix({"val_loss": total_loss.item()})

                avg_val_loss = sum(val_losses) / len(val_losses)

                # Synchronize val_loss across all processes for consistent best model selection
                if self.multi_gpu:
                    val_loss_tensor = torch.tensor([avg_val_loss], device=self.device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                    avg_val_loss = val_loss_tensor.item()

                print_rank0(
                    f"\nEpoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                )

                # Save models based on validation loss (only on rank 0)
                if is_main_process():
                    # Get underlying model from DDP wrapper
                    model_to_save = self.model.module if self.multi_gpu else self.model
                    save_lora_weights(
                        model_to_save, str(out_dir / "last_lora_weights.pt")
                    )
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        save_lora_weights(
                            model_to_save, str(out_dir / "best_lora_weights.pt")
                        )
                        print(f"✓ New best model saved (val_loss: {avg_val_loss:.6f})")

                    # Log to file
                    with open(out_dir / "val_stats.json", "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "epoch": epoch + 1,
                                    "global_step": global_step,
                                    "train_loss": avg_train_loss,
                                    "val_loss": avg_val_loss,
                                    "validation_queries": len(val_ds),
                                    "uses_unreviewed_proposals": getattr(val_ds, "uses_unreviewed_proposals", False),
                                }
                            )
                            + "\n"
                        )

                torch.cuda.empty_cache()

                # Back to training mode
                self.model.train()
            else:
                # No validation - just save model each epoch (only on rank 0)
                if is_main_process():
                    model_to_save = self.model.module if self.multi_gpu else self.model
                    save_lora_weights(
                        model_to_save, str(out_dir / "last_lora_weights.pt")
                    )
                print_rank0(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.6f} "
                            f"(step {global_step}; no validation)")

            if max_steps is not None and global_step >= max_steps:
                print_rank0(f"Stopped at the requested limit of {global_step} optimizer steps")
                break

        # Synchronize before final save
        if self.multi_gpu:
            dist.barrier()

        # Final save (only on rank 0)
        if is_main_process():
            if has_validation:
                print(f"\n{'=' * 80}")
                print("✅ Training complete!")
                print(f"{'=' * 80}")
                print(f"Best validation loss: {best_val_loss:.6f}")
                print(f"\nModels saved to {out_dir}:")
                print("  - best_lora_weights.pt (best validation loss)")
                print("  - last_lora_weights.pt (last epoch)")
                if dataset_type == "vpref_v03":
                    print("  - val_stats.json (JSONL validation loss by epoch)")
                    print("  - selected_val_samples.jsonl / validation_data.json (validation provenance)")
                    print("Validation loss is used for checkpoint selection; query IoU/Dice require separate evaluation.")
                else:
                    print("\n📊 To compute full metrics (mAP, cgF1) with NMS:")
                    print("   python validate_sam3_lora.py \\")
                    print("     --config <config_path> \\")
                    print(f"     --weights {out_dir}/best_lora_weights.pt \\")
                    print("     --val_data_dir <data_dir>/valid")
                print(f"{'=' * 80}")
            else:
                print(f"\n{'=' * 80}")
                print("✅ Training complete!")
                print(f"{'=' * 80}")
                print(f"\nModels saved to {out_dir}:")
                print("  - last_lora_weights.pt (last epoch)")
                print("  - train_stats.jsonl (training loss; no validation-based model selection)")
                print(f"{'=' * 80}")

        # Cleanup distributed training
        if self.multi_gpu:
            cleanup_distributed()

