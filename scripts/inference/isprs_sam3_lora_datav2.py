#!/usr/bin/env python3
"""Query-level union-mask evaluation of VPRef v0.3 train or original-val queries.

From the repository root:
  python scripts/inference/isprs_sam3_lora_datav2.py --check-data
  python scripts/inference/isprs_sam3_lora_datav2.py --device 0
  python scripts/inference/isprs_sam3_lora_datav2.py --run-dir outputs/vpref_v03_debug --split val --device 0
  python scripts/inference/isprs_sam3_lora_datav2.py --run-dir outputs/vpref_v07 --split test --device 0 --selection-mode top1
  python scripts/inference/isprs_sam3_lora_datav2.py --run-dir outputs/vpref_v08 --split val --device 0
  CUDA_VISIBLE_DEVICES=1 python scripts/inference/isprs_sam3_lora_datav2.py --run-dir outputs/vpref_v07 --split test --device 0
  CUDA_VISIBLE_DEVICES=1 python scripts/inference/isprs_sam3_lora_datav2.py --run-dir outputs/vpref_v08 --split test --device 0
  CUDA_VISIBLE_DEVICES=1 python scripts/inference/isprs_sam3_lora_datav2.py --run-dir outputs/vpref_v08-r2 --split test --device 0

Defaults to the saved Vaihingen/480px run and last_lora_weights.pt. These are
training-partition fitting diagnostics by default. --split val selects the saved
training.validation manifest for development evaluation, not independent testing.
The original legacy evaluator remains in isprs_sam3_lora.py.
"""
import argparse
import contextlib
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from PIL import Image as PILImage, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml

from data.vpref_v03_dataset import VPRefV03Dataset
from sam3.model.model_misc import SAM3Output
from sam3.train.data.collator import collate_fn_api

DEFAULT_RUN = ROOT / "outputs/sam3_lora_vpref_v03_vaihingen_union_480"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def collate_without_targets(points):
    """Only image/text enter inference; remove all annotation geometry and masks."""
    for point in points:
        for image in point.images:
            image.objects = []
        for query in point.find_queries:
            if query.input_bbox is not None or query.input_points is not None:
                raise ValueError("This evaluation requires text-only prompts")
            query.object_ids_output = []
            query.semantic_target = None
            query.is_exhaustive = False
            query.is_pixel_exhaustive = False
    return collate_fn_api(points, dict_key="input", with_seg_masks=False)


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            setattr(value, name, move_to_device(getattr(value, name), device))
    return value


def binary_metrics(prediction, target, ignore_mask=None):
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    if prediction.shape != target.shape:
        raise ValueError("Prediction/target dimensions differ")
    if ignore_mask is not None:
        ignore_mask = np.asarray(ignore_mask, dtype=bool)
        if ignore_mask.shape != target.shape:
            raise ValueError("Ignore/target dimensions differ")
        prediction = prediction & ~ignore_mask
        target = target & ~ignore_mask
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    union = tp + fp + fn
    return dict(intersection_px=tp, union_px=union, false_positive_px=fp, false_negative_px=fn,
                predicted_area_px=tp + fp, target_area_px=tp + fn,
                iou=tp / union if union else 1.0,
                dice=2 * tp / (2 * tp + fp + fn) if union else 1.0,
                precision=tp / (tp + fp) if tp + fp else float(not target.any()),
                recall=tp / (tp + fn) if tp + fn else 1.0)


def summarize(records):
    if not records:
        raise ValueError("No evaluated queries")
    intersection = sum(r["intersection_px"] for r in records)
    union = sum(r["union_px"] for r in records)
    return dict(queries=len(records), images=len({r["image_id"] for r in records}),
                mean_query_iou=float(np.mean([r["iou"] for r in records])),
                overall_iou=intersection / union if union else 1.0,
                mean_query_dice=float(np.mean([r["dice"] for r in records])),
                mean_query_precision=float(np.mean([r["precision"] for r in records])),
                mean_query_recall=float(np.mean([r["recall"] for r in records])),
                precision_at_iou={str(t): sum(r["iou"] >= t for r in records) / len(records)
                                  for t in (0.5, 0.7, 0.9)},
                queries_without_selected_masks=sum(r["selected_model_masks"] == 0 for r in records))


def prediction_union(outputs, 
                     batch_index, 
                     size, 
                     score_threshold=0.3, 
                     mask_threshold=0.5,
                     use_presence=True, 
                     presence_in_logits=False, 
                     selection_mode="union"):
    
    """Score filter, resize logits before sigmoid, and union all retained masks.

    No NMS, largest-component filter, GT-dependent selection or top-1 fallback.
    Empty predictions remain empty and still contribute to evaluation metrics.
    """
    scores = outputs["pred_logits"][batch_index].float().reshape(-1).sigmoid()
    masks = outputs["pred_masks"][batch_index]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or len(masks) != len(scores):
        raise ValueError("Expected matching [Q,1] logits and [Q,H,W] mask logits")
    presence = None
    if use_presence and not presence_in_logits:
        if "presence_logit_dec" not in outputs:
            raise ValueError("Presence scoring requested but presence_logit_dec is missing")
        value = outputs["presence_logit_dec"][batch_index].float().reshape(-1)
        if value.numel() != 1:
            raise ValueError("Expected one presence logit per text query")
        presence = float(value.sigmoid().item())
        scores = scores * presence
    if not torch.isfinite(scores).all() or not torch.isfinite(masks).all():
        raise FloatingPointError("Nonfinite model prediction")
    kept = torch.where(scores > score_threshold)[0]

    if selection_mode == "top1":
        if len(kept):
            kept = kept[scores[kept].argmax()].reshape(1)
    elif selection_mode != "union":
        raise ValueError(f"Unknown mask selection mode: {selection_mode}")
    
    union = torch.zeros(size, dtype=torch.bool)
    # Bound host memory even if all 200 model slots pass the score threshold.
    if len(kept):
        for selected in kept.split(16):
            resized = F.interpolate(masks[selected].detach().float().cpu()[:, None],
                                    size=size, mode="bilinear", align_corners=False)[:, 0]
            union |= (resized.sigmoid() > mask_threshold).any(0)
    details = dict(selected_model_masks=len(kept), raw_model_masks=len(scores),
                   selected_model_indices=kept.detach().cpu().tolist(),
                   selected_scores=scores[kept].detach().cpu().tolist(),
                   maximum_score=float(scores.max().item()) if len(scores) else None,
                   presence_probability=presence)
    return union.numpy(), details


def adapter_state(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(state, dict) or not state
            or any(not key.endswith((".lora_A", ".lora_B")) for key in state)
            or any(not isinstance(t, torch.Tensor) or t.ndim != 2 or not torch.isfinite(t).all()
                   for t in state.values())):
        raise ValueError("Expected a finite LoRA-only checkpoint from save_lora_weights")
    return state


def load_adapter_strict(model, path):
    state = adapter_state(path)
    expected = {key: value for key, value in model.state_dict().items()
                if key.endswith((".lora_A", ".lora_B"))}
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise ValueError(f"LoRA module mismatch: missing={missing[:5]}, unexpected={extra[:5]}")
    for key, value in state.items():
        if value.shape != expected[key].shape:
            raise ValueError(f"LoRA tensor shape mismatch: {key}")
    model.load_state_dict(state, strict=False)  # Base weights were loaded separately.
    return len(state)


def setup(args):
    """
    Load the dataset, select queries, and verify LoRA checkpoint presence.
    """
    run = args.run_dir.expanduser().resolve()
    config_path = (args.config or run / "config.yaml").expanduser().resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    training = config["training"]
    if training.get("dataset") != "vpref_v03" or training.get("target_mode", "union") != "union":
        raise ValueError("This evaluator expects a vpref_v03 union-mask training config")
    split = getattr(args, "split", "train")
    selection = (training if split == "train" else
                 (training.get("validation") or {}) if split == "val" else
                 (training.get("test") or {}))
    
    if split == "val" and not (args.manifest or selection.get("manifest")):
        raise ValueError("--split val requires training.validation.manifest or an explicit --manifest")
    task_families = selection.get("task_families", training.get("task_families"))
   
    if getattr(args, "all_task_families", False):
        task_families = None
    elif getattr(args, "task_families", None) is not None:
        task_families = args.task_families
    
    dataset = VPRefV03Dataset(
        data_dir=args.data_dir or selection.get("data_dir", training["data_dir"]),
        manifest=args.manifest or selection.get("manifest", f"{split}.jsonl"),
        domain=args.domain or training.get("domain", "Vaihingen"),
        resolution=args.resolution or training.get("resolution", 480),
        target_mode="union", 
        rotation_degrees=[0],
        allow_unreviewed_proposals=selection.get("allow_unreviewed_proposals",
            training.get("allow_unreviewed_proposals", False) if split == "test" else False),
        split=split,
        task_families=task_families,
    )

    records = dataset.samples[:args.num_samples] if args.num_samples else dataset.samples
    weights = (args.weights or run / "last_lora_weights.pt").expanduser().resolve()
    if not args.use_base_model and not weights.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {weights}")
    membership = None
    training_data = run / "training_data.json"
    if training_data.is_file():
        saved = json.loads(training_data.read_text())
        membership = (saved["manifest_sha256"] == sha(dataset.manifest_path)
                      and all(r["sample_id"] in saved["sample_ids"] for r in records))
    val_membership = None
    validation_data = run / "validation_data.json"
    
    if validation_data.is_file():
        saved_val = json.loads(validation_data.read_text())
        val_membership = (saved_val["manifest_sha256"] == sha(dataset.manifest_path)
                          and all(r["sample_id"] in saved_val["sample_ids"] for r in records))
    role = {"train": "training_partition_fit_diagnostic", "val": "original_val_development_evaluation",
            "test": "original_test_evaluation"}[split]
    
    if dataset.uses_unreviewed_proposals:
        role = {"train": "unreviewed_training_proposal_fit_diagnostic", 
                "val": "unreviewed_original_val_development_evaluation",
                "test": "unreviewed_original_test_evaluation"}[split]
    test_isolation = None
    if split == "test":
        test_isolation = True
        for saved_path in (training_data, validation_data):
            if not saved_path.is_file():
                test_isolation = False
                continue
            selection_record = json.loads(saved_path.read_text())
            if not selection_record.get("source_scenes"):
                test_isolation = False
                continue
            if {r["source_scene"] for r in records} & set(selection_record["source_scenes"]):
                raise ValueError("Test source scenes overlap the saved training/validation selection")
            if {r["sample_id"] for r in records} & set(selection_record.get("sample_ids", [])):
                raise ValueError("Test queries overlap the saved training/validation selection")
    stats_file = run / "train_stats.jsonl"
    training_steps = None
    if stats_file.is_file():
        stats = [json.loads(line) for line in stats_file.read_text().splitlines() if line.strip()]
        training_steps = stats[-1].get("global_step") if stats else None
    metadata = dict(
        evaluation_role=role,
        uses_unreviewed_proposals=dataset.uses_unreviewed_proposals,
        recorded_training_optimizer_steps=training_steps,
        smoke_run_verification_present=(run / "smoke_verification.json").is_file(),
        independent_test=bool(split == "test" and test_isolation and not dataset.uses_unreviewed_proposals),
        held_out_source_test=split == "test", test_scene_isolation_verified=test_isolation,
        config_path=str(config_path), config_sha256=sha(config_path),
        run_dir=str(run), manifest_path=str(dataset.manifest_path),
        manifest_sha256=sha(dataset.manifest_path), data_dir=str(dataset.data_dir),
        domain=dataset.domain, resolution=dataset.resolution, target_mode="union", rotation_degrees=[0],
        queries=len(records), images=len({r["image_id"] for r in records}),
        task_families=list(dataset.task_families) if dataset.task_families else None,
        task_family_counts=dict(sorted(Counter(r.get("task_family", "unspecified") for r in records).items())),
        # sample_ids=[r["sample_id"] for r in records],
        all_queries_verified_in_saved_training_selection=membership,
        all_queries_verified_in_saved_validation_selection=val_membership,
        use_base_model=args.use_base_model,
        lora_weights=None if args.use_base_model else str(weights),
        lora_sha256=None if args.use_base_model else sha(weights),
        score_threshold=args.prob_threshold, mask_threshold=args.mask_threshold,
        selection_mode=getattr(args, "selection_mode", "union"),
        use_presence=not args.no_presence, nms=False,
        score_policy="sigmoid(logit) times presence, unless already joint in model",
        mask_policy=("bilinear resize logits, sigmoid, threshold; " +
                     ("highest-scoring passing mask" if getattr(args, "selection_mode", "union") == "top1"
                      else "union all passing masks")),
        metric_definitions={"mean_query_iou": "Mean foreground IoU over English queries",
                            "overall_iou": "Sum of intersections / sum of unions over queries"},
        source_partition=split, empty_diagnostics_included=False, ground_truth_prompts=False,
        pixel_ignore_enabled=dataset.has_ignore_masks,
        pixel_ignore_policy="Excluded from pixel metrics; never model input" if dataset.has_ignore_masks else None,
    )
    return config, dataset, records, weights, metadata


def data_check(dataset, records, weights, use_base_model):
    for i, row in enumerate(records):
        point = dataset[i]
        if point.find_queries[0].query_text != row["text"]:
            raise ValueError("Query/manifest mismatch")
        batch = collate_without_targets([point])["input"]
        if hasattr(batch, "vpref_valid_pixels"):
            raise ValueError("Ignore annotations leaked into model input")
        target = batch.find_targets[0]
        if (target.num_boxes.tolist() != [0] or target.segments is not None
                or target.boxes.numel() or target.semantic_segments.numel()
                or batch.find_inputs[0].input_boxes.numel()
                or batch.find_inputs[0].input_points.numel()):
            raise ValueError("Ground truth leaked into inference batch")
        if tuple(batch.img_batch.shape) != (1, 3, dataset.resolution, dataset.resolution):
            raise ValueError("Unexpected image batch dimensions")
    count = 0 if use_base_model else len(adapter_state(weights))
    return dict(status="passed", checked_queries=len(records),
                checked_images=len({r["image_id"] for r in records}),
                checkpoint_tensors_checked=count, ground_truth_prompts=False,
                model_forward_performed=False, pixel_ignore_enabled=dataset.has_ignore_masks)


def preview(image, target, prediction, target_label="Ground truth", ignore_mask=None):
    """
    preview the image with target and prediction overlays, including ignored pixels if present
    for folder: /home/icclab/Documents/lqw/sam3_finetune_lora/outputs/vpref_v08-r2/vpref_v03_val_lora/previews
    """
    raw = np.asarray(image.convert("RGB"))
    panels = [raw]
    for m, color in ((target, [0, 220, 50]), (prediction, [255, 60, 60])):
        panel = raw.copy()
        panel[m] = (0.5 * panel[m] + 0.5 * np.asarray(color)).astype(np.uint8)
        panels.append(panel)
    error = raw.copy()
    for m, color in ((target & prediction, [0, 220, 50]),
                     (prediction & ~target, [255, 50, 50]),
                     (target & ~prediction, [60, 120, 255])):
        error[m] = (0.25 * error[m] + 0.75 * np.asarray(color)).astype(np.uint8)
    if ignore_mask is not None:
        ignored = np.asarray(ignore_mask, dtype=bool)
        for panel in panels[1:] + [error]:
            panel[ignored] = (0.25 * raw[ignored] + 0.75 * np.asarray([255, 150, 20])).astype(np.uint8)
    panels.append(error)
    height, width = raw.shape[:2]
    canvas = PILImage.new("RGB", (4 * width, height + 28), "white")
    draw = ImageDraw.Draw(canvas)

    titles = ["Image", target_label, "Prediction", "TP green / FP red / FN blue" +
              (" / ignored orange" if ignore_mask is not None and np.any(ignore_mask) else "")]
    
    for i, (panel, title) in enumerate(zip(panels, titles)):
        canvas.paste(PILImage.fromarray(panel), (i * width, 28))
        draw.text((i * width + 6, 7), title, fill="black")
    return canvas


def write_results(out, results, protocol):
    """
    Publish per-query results; missing predictions remain in the denominator.
    """

    (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    
    fields = ["sample_id", "image_id", "text", "iou", "dice", "precision", "recall",
              "selected_model_masks", "maximum_score", "predicted_area_px", "target_area_px",
              "false_positive_px", "false_negative_px", "task_family"]
    
    with (out / "s.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    metrics = summarize(results)
    summary = dict(status="completed", evaluation_role=protocol["evaluation_role"],
                   independent_test=protocol.get("independent_test", False), metrics=metrics,
                   held_out_source_test=protocol.get("held_out_source_test", False),
                   test_scene_isolation_verified=protocol.get("test_scene_isolation_verified"),
                   task_families=protocol.get("task_families"),
                   task_family_counts=protocol.get("task_family_counts"),
                   pixel_ignore_enabled=protocol.get("pixel_ignore_enabled", False),
                   source_partition=protocol.get("source_partition", "train"),
                   uses_unreviewed_proposals=protocol.get("uses_unreviewed_proposals", False),
                   recorded_training_optimizer_steps=protocol.get("recorded_training_optimizer_steps"),
                   all_queries_verified_in_saved_training_selection=protocol[
                       "all_queries_verified_in_saved_training_selection"],
                   all_queries_verified_in_saved_validation_selection=protocol.get(
                       "all_queries_verified_in_saved_validation_selection"))
    dump(out / "summary.json", summary)
    title = ("VPRef v0.3 original-val development evaluation"
             if protocol.get("source_partition") == "val" else "VPRef v0.3 training-set fit evaluation")
    description = ("These queries come from the original val partition and are used for development."
                   if protocol.get("source_partition") == "val"
                   else "These queries come from the original train partition.")
    if protocol.get("source_partition") == "test":
        title = "VPRef original-test evaluation"
        description = "These queries come from the original held-out test partition."
    target_label = "Proposal target" if protocol.get("uses_unreviewed_proposals") else "Ground-truth mask"
    page = [f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{title}</title>',
            '<style>body{font:16px system-ui;max-width:1600px;margin:24px auto;padding:0 16px}'
            'article{margin:28px 0;padding:12px;border:1px solid #bbb}img{width:100%}</style>',
            f'<h1>{title}</h1>',
            f'<p>{description} '+('Independent test labels have not been certified.'
                if not protocol.get('independent_test') else 'Source-scene isolation verified.')+'</p>',
            f'<p>Queries: {len(results)}; mean query IoU: {metrics["mean_query_iou"]:.4f}; '
            f'overall IoU: {metrics["overall_iou"]:.4f}</p>',
            '<p><a href="summary.json">Summary</a> · <a href="results.csv">Per-query metrics</a> · '
            '<a href="evaluation_protocol.json">Protocol</a></p>']
    if protocol.get("uses_unreviewed_proposals"):
        page.append('<p>Reference masks are unreviewed geometric proposals, not accepted gold annotations.</p>')
    if protocol.get("recorded_training_optimizer_steps") is not None:
        page.append(f'<p>Recorded training optimizer steps: {protocol["recorded_training_optimizer_steps"]}.</p>')
    for r in results:
        page.append(f'<article><h2>{html.escape(r["sample_id"])} · {html.escape(r["image_id"])}</h2>'
                    f'<p>{html.escape(r["text"])}</p><p>IoU={r["iou"]:.4f}; Dice={r["dice"]:.4f}; '
                    f'retained model masks={r["selected_model_masks"]}</p>'
                    f'<a href="{r["preview_path"]}"><img src="{r["preview_path"]}"></a>'
                    f'<p><a href="{r["prediction_path"]}">Prediction mask</a> · '
                    f'<a href="{r["target_path"]}">{target_label}</a></p></article>')
    (out / "index.html").write_text("".join(page) + "</html>")
    return summary


def evaluate(args):
    config, dataset, records, weights, protocol = setup(args)

    if args.check_data:
        report = dict(protocol, data_check=data_check(dataset, records, weights, args.use_base_model))
        print(json.dumps(report, indent=2))
        return report
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. --check-data validates the checkpoint and data on CPU; "
                           "model evaluation requires an accessible CUDA GPU.")
    
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    from sam3.model_builder import build_sam3_image_model
    from lora_layers import LoRAConfig, apply_lora_to_model
    from tools.utils import patch_dynamic_rope

    model_config = config.get("model", {})
    model = build_sam3_image_model(
        device=device.type, compile=False, eval_mode=False, load_from_HF=False,
        checkpoint_path=model_config.get("checkpoint_path", "/home/icclab/Documents/lqw/sam3/weights/sam3.pt"),
        bpe_path=model_config.get("bpe_path", "/home/icclab/Documents/lqw/sam3/assets/bpe_simple_vocab_16e6.txt.gz"))
    patch_dynamic_rope(model)
    if not args.use_base_model:
        model = apply_lora_to_model(model, LoRAConfig(**config["lora"]))
        protocol["loaded_adapter_tensors"] = load_adapter_strict(model, weights)
    model.to(device).eval()
    model.num_interactive_steps_val = 0  # No validation prompts sampled from ground truth.
    protocol["presence_already_in_logits"] = bool(getattr(model, "supervise_joint_box_scores", False))
    protocol["device"] = str(device)
    protocol["amp"] = args.amp
    protocol["created_utc"] = datetime.now(timezone.utc).isoformat()
    protocol["script_sha256"] = sha(__file__)
    purpose = dataset.split if dataset.split in ("val", "test") else "fit"
    name = f"vpref_v03_{purpose}_base" if args.use_base_model else f"vpref_v03_{purpose}_lora"

    out = (args.output_dir or args.run_dir / name).expanduser().resolve()
    if out.exists():
        out = out.with_name(out.name + "_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True)
    
    for folder in ("prediction_masks", "target_masks", "previews"):
        (out / folder).mkdir()
    if getattr(args, "save_raw_predictions", False):
        (out / "raw_predictions").mkdir()
        protocol["raw_prediction_encoding"] = "unmodified float32 logits in compressed npz"

    dump(out / "evaluation_protocol.json", protocol)
    dump(out / "run_status.json", dict(status="running"))
    loader = DataLoader(Subset(dataset, range(len(records))), 
                        batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate_without_targets, pin_memory=True)
    
    results, offset = [], 0
    with torch.inference_mode():
        for wrapped in tqdm(loader, desc=f"VPRef {dataset.split} evaluation"):
            batch = move_to_device(wrapped["input"], device)
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if args.amp else contextlib.nullcontext()

            with amp:
                outputs = model(batch)
            
            with SAM3Output.iteration_mode(outputs, SAM3Output.IterMode.ALL_STEPS_PER_STAGE) as stages:
                final = list(stages)[-1][-1]
            count = final["pred_logits"].shape[0]

            if count != batch.img_batch.shape[0]:
                raise ValueError("Prediction/query batch count mismatch")
            
            for i in range(count):
                row = records[offset + i]
                with PILImage.open(dataset._asset(row["mask_path"])) as im:
                    target = np.asarray(im) > 0
                ignored = dataset.read_ignore_mask(row)
                
                prediction, details = prediction_union(
                    final, i, target.shape, args.prob_threshold, args.mask_threshold,
                    not args.no_presence, protocol["presence_already_in_logits"],
                    getattr(args, "selection_mode", "union"))
                qid = row["sample_id"]
                
                if getattr(args, "save_raw_predictions", False):
                    arrays = {key: final[key][i:i+1].detach().float().cpu().numpy()
                              for key in ("pred_logits", "pred_masks", "presence_logit_dec") if key in final}
                    np.savez_compressed(out / "raw_predictions" / f"{qid}.npz", **arrays)
                
                # # save the prediction and target masks as PNGs for visual inspection; these are not used in the HTML preview
                paths = dict(prediction_path=f"prediction_masks/{qid}.png",
                             target_path=f"target_masks/{qid}.png", preview_path=f"previews/{qid}.png")
                # PILImage.fromarray(prediction.astype(np.uint8) * 255).save(out / paths["prediction_path"])
                # PILImage.fromarray(target.astype(np.uint8) * 255).save(out / paths["target_path"])
                
                # # preview the image with target and prediction overlays, including ignored pixels if present
                # with PILImage.open(dataset._asset(row["image_path"])) as image:
                #     preview(image, 
                #             target, 
                #             prediction,
                #             "Proposal target (unreviewed)" if dataset.uses_unreviewed_proposals
                #             else "Ground truth", ignore_mask=ignored).save(out / paths["preview_path"])
                    
                results.append(dict(sample_id=qid, 
                                    image_id=row["image_id"],
                                    text=row["text"],
                                    task_family=row.get("task_family"),
                                    domain=row["domain"], 
                                    patch_id=row["patch_id"],
                                    ignored_pixels=int(ignored.sum()),
                                    target_region_ids=row["target_region_ids"],
                                    **paths, 
                                    **details, 
                                    **binary_metrics(prediction, target, ignored)))
            offset += count
            del wrapped, batch, outputs, final

    if offset != len(records):
        raise ValueError("Not all selected queries were evaluated")

    summary = write_results(out, results, protocol)
    dump(out / "run_status.json", dict(status="completed", evaluated_queries=offset))
    
    print(json.dumps(summary, indent=2))
    print(f"Results: {out}\nReview: {out / 'index.html'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", "--file_path", dest="run_dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train",
                        help="Evaluate train, original val, or the merged original test manifest")
    parser.add_argument("--config", type=Path, help="Defaults to the saved run's config.yaml")
    parser.add_argument("--weights", type=Path, help="Defaults to last_lora_weights.pt in the run directory")
    parser.add_argument("--data-dir", "--vai-data-dir", dest="data_dir", type=Path)
    parser.add_argument("--manifest", help="Package-relative manifest; defaults to the selected split's config")
    families = parser.add_mutually_exclusive_group()
    families.add_argument("--task-families", nargs="+", metavar="FAMILY",
                          help="Select one or more task families, overriding the config")
    families.add_argument("--all-task-families", action="store_true",
                          help="Use all families in the manifest, overriding the config filter")
    parser.add_argument("--domain", choices=["Vaihingen", "Potsdam"])
    parser.add_argument("--resolution", type=int, help="Defaults to saved training resolution (480)")
    parser.add_argument("--device", type=int, default=0, help="Logical CUDA index; respects CUDA_VISIBLE_DEVICES")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, help="Evaluate exactly the first N selected queries")
    parser.add_argument("--prob-threshold", "--score-threshold", dest="prob_threshold", type=float, default=0.3)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--selection-mode", choices=["union", "top1"], default="union",
                        help="Union all passing masks, or retain only the highest-scoring passing mask")
    parser.add_argument("--save-raw-predictions", action="store_true",
                        help="Cache float32 logits for CPU postprocessing comparisons without another forward pass")
    parser.add_argument("--no-presence", action="store_true", help="Do not multiply an additional presence probability")
    parser.add_argument("--amp", action="store_true", help="Use optional bfloat16 inference; default is float32")
    parser.add_argument("--use-base-model", action="store_true", help="Evaluate pretrained SAM3 without LoRA")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-data", action="store_true", help="Check data, text-only collation and LoRA file on CPU")
    args = parser.parse_args()

    if (args.batch_size < 1 or args.num_workers < 0 or args.device < 0
            or (args.num_samples is not None and args.num_samples < 1)
            or (args.resolution is not None and args.resolution < 1)):
        parser.error("Invalid batch size, workers, device, sample limit or resolution")

    if not 0 <= args.prob_threshold <= 1 or not 0 <= args.mask_threshold <= 1:
        parser.error("Probability thresholds must be between 0 and 1")
    evaluate(args)


if __name__ == "__main__":
    main()



