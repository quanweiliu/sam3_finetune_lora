#!/usr/bin/env python3
"""
Validation script for SAM3 LoRA model
Loads saved weights and runs validation with detailed debugging
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [1]))
print('using GPU %s' % ','.join(map(str, [1])))

import sys
# Add project root and training directory for local imports
_SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(_SCRIPT_DIR, "../.."))
sys.path.append(os.path.join(_SCRIPT_DIR, "../training"))

import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
import contextlib
import gc

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
from sam3.model.box_ops import box_xywh_to_xyxy
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights, count_parameters

from torchvision.transforms import v2

# Import evaluation modules
from sam3.eval.cgf1_eval import CGF1Evaluator, COCOCustom
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_utils
from sam3.train.masks_ops import rle_encode

# Import SAM3's NMS
from sam3.perflib.nms import nms_masks

from data.vaihingen_dataset import VaiRefDataset
from data.COCODataset import COCOSegmentDataset, DirectCOCODataset
from tools.utils import setup_distributed, cleanup_distributed, patch_dynamic_rope
from tools.utils import merge_overlapping_masks, get_world_size, get_rank, print_rank0


def apply_sam3_nms(pred_logits, pred_masks, pred_boxes, prob_threshold=0.3, nms_iou_threshold=0.7, max_detections=100):
    """
    Apply SAM3's standard NMS pipeline to filter predictions.

    Args:
        pred_logits: [N, 1] logits
        pred_masks: [N, H, W] mask logits
        pred_boxes: [N, 4] boxes in normalized format
        prob_threshold: Score threshold for filtering (default: 0.3, SAM3 uses 0.5)
        nms_iou_threshold: IoU threshold for NMS (default: 0.7, SAM3 uses 0.5-0.7)
        max_detections: Maximum detections to keep (default: 100)

    Returns:
        Tuple of (filtered_masks, filtered_scores, filtered_boxes)
    """
    if len(pred_logits) == 0:
        return pred_masks[:0], pred_logits[:0].squeeze(-1), pred_boxes[:0]

    # Convert logits to probabilities
    pred_probs = torch.sigmoid(pred_logits).squeeze(-1)  # [N]

    # Convert mask logits to binary masks (sigmoid + threshold)
    pred_masks_sigmoid = torch.sigmoid(pred_masks)  # [N, H, W]
    pred_masks_binary = pred_masks_sigmoid > 0.5  # [N, H, W]

    # Apply SAM3's NMS
    # nms_masks expects: pred_probs [N], pred_masks [N, H, W], prob_threshold, iou_threshold
    # Returns: keep mask [N] of booleans
    keep_mask = nms_masks(
        pred_probs=pred_probs,
        pred_masks=pred_masks_binary.float(),  # NMS expects float masks
        prob_threshold=prob_threshold,
        iou_threshold=nms_iou_threshold
    )

    # Filter predictions
    filtered_masks = pred_masks_sigmoid[keep_mask]  # Keep sigmoid masks for later
    filtered_scores = pred_probs[keep_mask]
    filtered_boxes = pred_boxes[keep_mask]

    # Top-K selection by score
    if max_detections > 0 and len(filtered_scores) > max_detections:
        top_k_scores, top_k_indices = torch.topk(filtered_scores, k=max_detections, largest=True)
        filtered_masks = filtered_masks[top_k_indices]
        filtered_scores = top_k_scores
        filtered_boxes = filtered_boxes[top_k_indices]

    return filtered_masks, filtered_scores, filtered_boxes


def convert_predictions_to_coco_format(predictions_list, 
                                       image_ids, 
                                       resolution=288,
                                       prob_threshold=0.3, 
                                       nms_iou_threshold=0.7, 
                                       max_detections=100,
                                       merge_cracks=False, 
                                       merge_iou_threshold=0.15):
    """
    Convert model predictions to COCO format using SAM3's NMS pipeline.

    Args:
        predictions_list: List of predictions per image
        image_ids: List of image IDs
        resolution: Resolution for box scaling (default: 288)
        prob_threshold: Score threshold (default: 0.3, SAM3 uses 0.5)
        nms_iou_threshold: NMS IoU threshold (default: 0.7)
        max_detections: Max detections per image (default: 100)
        merge_cracks: If True, merge overlapping segments instead of NMS suppression (default: False)
        merge_iou_threshold: IoU threshold for merging (default: 0.15, lower = more aggressive)
    """
    coco_predictions = []
    pred_id = 0

    if merge_cracks:
        print(f"\n[INFO] Converting {len(predictions_list)} predictions to COCO format...")
        print(f"[INFO] Using CRACK MERGING mode: prob_threshold={prob_threshold}, merge_iou={merge_iou_threshold}, max_dets={max_detections}")
        print(f"[INFO] This will MERGE overlapping crack segments instead of suppressing them")
    else:
        print(f"\n[INFO] Converting {len(predictions_list)} predictions to COCO format...")
        print(f"[INFO] Using SAM3 NMS: prob_threshold={prob_threshold}, nms_iou={nms_iou_threshold}, max_dets={max_detections}")

    for img_id, preds in tqdm(zip(image_ids, predictions_list), total=len(predictions_list), desc="Converting predictions"):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        logits = preds['pred_logits']  # [N, 1]
        boxes = preds['pred_boxes']    # [N, 4]
        masks = preds['pred_masks']    # [N, H, W]

        if merge_cracks:
            # Step 1: Filter by score threshold
            pred_probs = torch.sigmoid(logits).squeeze(-1)  # [N]
            valid_mask = pred_probs > prob_threshold

            filtered_masks = masks[valid_mask]
            filtered_scores = pred_probs[valid_mask]
            filtered_boxes = boxes[valid_mask]

            if len(filtered_masks) > 0:
                # Step 2: Convert masks to binary
                pred_masks_sigmoid = torch.sigmoid(filtered_masks)
                pred_masks_binary = (pred_masks_sigmoid > 0.5)

                # Step 3: MERGE overlapping crack segments
                merged_masks, merged_scores, merged_boxes = merge_overlapping_masks(
                    pred_masks_binary.cpu(),
                    filtered_scores.cpu(),
                    filtered_boxes.cpu(),
                    iou_threshold=merge_iou_threshold
                )

                # Step 4: Top-K selection by score
                if max_detections > 0 and len(merged_scores) > max_detections:
                    top_k_scores, top_k_indices = torch.topk(merged_scores, k=max_detections, largest=True)
                    merged_masks = merged_masks[top_k_indices]
                    merged_scores = top_k_scores
                    merged_boxes = merged_boxes[top_k_indices]

                # Return merged results (already binary)
                filtered_masks = merged_masks.float()  # Already binary, just convert to float
                filtered_scores = merged_scores
                filtered_boxes = merged_boxes
            else:
                filtered_masks = torch.tensor([])
                filtered_scores = torch.tensor([])
                filtered_boxes = torch.tensor([])
        else:
            # Apply SAM3's NMS pipeline (standard suppression)
            filtered_masks, filtered_scores, filtered_boxes = apply_sam3_nms(
                pred_logits=logits,
                pred_masks=masks,
                pred_boxes=boxes,
                prob_threshold=prob_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections=max_detections
            )

        if len(filtered_masks) > 0:
            # Convert filtered masks to binary for RLE encoding
            binary_masks = (filtered_masks > 0.5).cpu()
            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, filtered_scores.cpu().tolist(), filtered_boxes.cpu().tolist())):
                cx, cy, w, h = box
                x = (cx - w/2) * resolution
                y = (cy - h/2) * resolution
                w = w * resolution
                h = h * resolution

                pred_dict = {
                    'image_id': int(img_id),
                    'category_id': 1,
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                }

                coco_predictions.append(pred_dict)
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset(dataset, image_ids=None, mask_resolution=288):
    """
    Create COCO ground truth dictionary from dataset.

    OPTIMIZATION: Downsample GT masks to 288×288 to match prediction resolution.
    """
    print(f"\n[INFO] Creating COCO ground truth (downsampling to {mask_resolution}×{mask_resolution})...")

    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    for idx in tqdm(list(indices), desc="Creating GT"):
        # Check if idx is within valid range
        if idx >= len(dataset):
            print(f"[WARNING] Skipping index {idx} - out of range (dataset length: {len(dataset)})")
            continue
            
        coco_gt['images'].append({
            'id': int(idx),
            'width': mask_resolution,
            'height': mask_resolution,
            'is_instance_exhaustive': True
        })

        datapoint = dataset[idx]

        for obj in datapoint.images[0].objects:
            # Scale boxes to mask_resolution
            box = obj.bbox * mask_resolution
            x1, y1, x2, y2 = box.tolist()
            x, y, w, h = x1, y1, x2-x1, y2-y1

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            if obj.segment is not None:
                # Downsample mask from 1008×1008 to mask_resolution×mask_resolution
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                downsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(mask_resolution, mask_resolution),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = downsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    print(f"[INFO] Created {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations")

    return coco_gt


def convert_predictions_to_coco_format_original_res(
    predictions_list, 
    image_ids, 
    dataset, 
    model_resolution=288, 
    score_threshold=0.0, 
    merge_overlaps=True, 
    iou_threshold=0.3, 
    debug=False
    ):
    """
    Convert model predictions to COCO format at ORIGINAL image resolution.

    This matches the inference approach (infer_sam.py) where:
    1. Masks are upsampled from 288x288 to original image size
    2. Boxes are scaled to original image size
    3. Evaluation happens at original resolution

    Args:
        predictions_list: List of predictions per image
        image_ids: List of image IDs (indices into dataset)
        dataset: Dataset to get original image sizes
        model_resolution: Model output resolution (default: 288)
        score_threshold: Confidence threshold
        merge_overlaps: Whether to merge overlapping predictions
        iou_threshold: IoU threshold for merging
        debug: Print debug info
    """
    coco_predictions = []
    pred_id = 0

    if debug:
        print(f"\n[DEBUG] Converting {len(predictions_list)} predictions to COCO format (ORIGINAL RESOLUTION)...")
        if merge_overlaps:
            print(f"[DEBUG] Overlapping segment merging ENABLED (IoU threshold={iou_threshold})")

    for img_id, preds in zip(image_ids, predictions_list):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        # Get original image size from dataset
        datapoint = dataset[img_id]
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        logits = preds['pred_logits']
        boxes = preds['pred_boxes']
        masks = preds['pred_masks']  # [N, 288, 288]

        scores = torch.sigmoid(logits).squeeze(-1)

        # Filter by score threshold
        valid_mask = scores > score_threshold
        num_before = len(scores)
        scores = scores[valid_mask]
        boxes = boxes[valid_mask]
        masks = masks[valid_mask]

        if debug and img_id == image_ids[0]:
            print(f"[DEBUG] Image {img_id}: {num_before} queries -> {len(scores)} after filtering (threshold={score_threshold})")
            if len(scores) > 0:
                print(f"[DEBUG]   Original size: {orig_w}x{orig_h}")
                print(f"[DEBUG]   Filtered scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

        if len(masks) == 0:
            continue

        # Upsample masks from 288x288 to original resolution (like infer_sam.py)
        masks_sigmoid = torch.sigmoid(masks)  # [N, 288, 288]
        masks_upsampled = torch.nn.functional.interpolate(
            masks_sigmoid.unsqueeze(1).float(),  # [N, 1, 288, 288]
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [N, orig_h, orig_w]

        binary_masks = (masks_upsampled > 0.5).cpu()

        # Merge overlapping predictions
        if merge_overlaps and len(binary_masks) > 0:
            num_before_merge = len(binary_masks)
            binary_masks, scores, boxes = merge_overlapping_masks(
                binary_masks, scores.cpu(), boxes.cpu(), iou_threshold=iou_threshold
            )
            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Merged {num_before_merge} predictions -> {len(binary_masks)} (IoU threshold={iou_threshold})")

        if len(binary_masks) > 0:
            mask_areas = binary_masks.flatten(1).sum(1)

            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Upsampled mask shape: {binary_masks.shape}")
                print(f"[DEBUG]   Mask areas: min={mask_areas.min():.0f}, max={mask_areas.max():.0f}, mean={mask_areas.float().mean():.0f}")

            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, scores.cpu().tolist(), boxes.cpu().tolist())):
                # Convert box from normalized [0,1] to original image coordinates
                cx, cy, w_norm, h_norm = box
                x = (cx - w_norm/2) * orig_w
                y = (cy - h_norm/2) * orig_h
                w = w_norm * orig_w
                h = h_norm * orig_h

                # Clamp coordinates to image bounds
                x = max(0, min(x, orig_w))
                y = max(0, min(y, orig_h))
                w = max(0, min(w, orig_w - x))
                h = max(0, min(h, orig_h - y))

                # Skip if box is too small after clamping
                if w < 1 or h < 1:
                    continue

                pred_dict = {
                    'image_id': int(img_id),
                    'category_id': 1,
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                }

                if debug and img_id == image_ids[0] and idx == 0:
                    print(f"[DEBUG]   First prediction bbox (at {orig_w}x{orig_h}): {pred_dict['bbox']}")

                coco_predictions.append(pred_dict)
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset_original_res(dataset, image_ids=None, debug=False):
    """
    Create COCO ground truth dictionary from dataset at ORIGINAL resolution.

    This matches the inference approach (infer_sam.py) where GT is kept
    at original image size for evaluation.

    Args:
        dataset: Dataset with images and annotations
        image_ids: List of image IDs to include (None = all)
        debug: Print debug info
    """
    if debug:
        print(f"\n[DEBUG] Creating COCO ground truth (ORIGINAL RESOLUTION)...")

    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    for idx in indices:
        datapoint = dataset[idx]

        # Get original image size
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        coco_gt['images'].append({
            'id': int(idx),
            'width': orig_w,
            'height': orig_h,
            'is_instance_exhaustive': True
        })

        for obj in datapoint.images[0].objects:
            # Scale boxes from normalized [0,1] to original size
            box = obj.bbox  # Already in [0,1] normalized coordinates
            x1, y1, x2, y2 = box.tolist()

            # Convert to original image coordinates
            x1_orig = x1 * orig_w
            y1_orig = y1 * orig_h
            x2_orig = x2 * orig_w
            y2_orig = y2 * orig_h

            # Convert to COCO format [x, y, w, h]
            x, y = x1_orig, y1_orig
            w = x2_orig - x1_orig
            h = y2_orig - y1_orig

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            if obj.segment is not None:
                # Upsample mask from 1008x1008 to original size
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                upsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = upsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    if debug:
        print(f"[DEBUG] Created {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations")
        if len(coco_gt['annotations']) > 0:
            sample_gt = coco_gt['annotations'][0]
            sample_img = coco_gt['images'][0]
            print(f"[DEBUG] Sample GT: image_id={sample_gt['image_id']}, bbox={sample_gt['bbox']}, image_size={sample_img['width']}x{sample_img['height']}")

    return coco_gt


def move_to_device(obj, device):
    """Recursively move objects to device"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            val = getattr(obj, field)
            setattr(obj, field, move_to_device(val, device))
        return obj
    return obj


def compute_miou(coco_gt_dict, coco_predictions):
    """
    Compute Mean IoU (mIoU) for binary segmentation.

    Builds per-image binary masks from RLE annotations, then computes
    pixel-level confusion matrix accumulated across all images.

    For binary (crack + background):
        crack_IoU = TP / (TP + FP + FN)
        bg_IoU    = TN / (TN + FP + FN)
        mIoU      = (crack_IoU + bg_IoU) / 2

    Args:
        coco_gt_dict: COCO ground truth dict with 'images' and 'annotations'
        coco_predictions: List of COCO prediction dicts

    Returns:
        dict with keys: miou, iou_object, iou_background, total_tp, total_fp, total_fn, total_tn
    """

    # Group GT annotations by image_id
    gt_by_image = {}
    for ann in coco_gt_dict['annotations']:
        img_id = ann['image_id']
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
        gt_by_image[img_id].append(ann)

    # Group predictions by image_id
    pred_by_image = {}
    for pred in coco_predictions:
        img_id = pred['image_id']
        if img_id not in pred_by_image:
            pred_by_image[img_id] = []
        pred_by_image[img_id].append(pred)

    # Image dimensions lookup
    img_sizes = {img['id']: (img['height'], img['width']) for img in coco_gt_dict['images']}

    # Accumulate confusion matrix
    total_tp = 0  # object pixels correctly predicted
    total_fp = 0  # background pixels predicted as object
    total_fn = 0  # object pixels predicted as background
    total_tn = 0  # background pixels correctly predicted

    images_processed = 0

    for img_id, (h, w) in tqdm(img_sizes.items(), desc="Computing MIoU"):
        # Build GT binary mask (union of all object annotations)
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in gt_by_image.get(img_id, []):
            if 'segmentation' in ann:
                try:
                    decoded = mask_utils.decode(ann['segmentation'])
                    gt_mask = np.logical_or(gt_mask, decoded).astype(np.uint8)
                except Exception:
                    pass

        # Build prediction binary mask (union of all predicted objects)
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        for pred in pred_by_image.get(img_id, []):
            if 'segmentation' in pred:
                try:
                    decoded = mask_utils.decode(pred['segmentation'])
                    pred_mask = np.logical_or(pred_mask, decoded).astype(np.uint8)
                except Exception:
                    pass

        # Per-pixel metrics
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        fp = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
        fn = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()
        tn = np.logical_and(np.logical_not(pred_mask), np.logical_not(gt_mask)).sum()

        total_tp += intersection
        total_fp += fp
        total_fn += fn
        total_tn += tn
        images_processed += 1

    # Per-class IoU
    iou_object = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
    iou_bg = total_tn / (total_tn + total_fp + total_fn) if (total_tn + total_fp + total_fn) > 0 else 0.0
    miou = (iou_object + iou_bg) / 2.0

    print(f"\n[MIoU] Processed {images_processed} images")
    print(f"[MIoU] Confusion matrix: TP={total_tp}, FP={total_fp}, FN={total_fn}, TN={total_tn}")

    return {
        'miou': miou,
        'iou_object': iou_object,
        'iou_background': iou_bg,
        'total_tp': int(total_tp),
        'total_fp': int(total_fp),
        'total_fn': int(total_fn),
        'total_tn': int(total_tn),
    }


def validate(config_path, weights_path, val_data_dir, num_samples=None,
             prob_threshold=0.3, nms_iou=0.7, merge_cracks=False, merge_iou=0.15,
             use_base_model=False, dataset_type="coco",
             vai_data_dir=None, vai_split="valid", vai_variant="standard"):
    """Run validation with full metrics (mAP, cgF1) and SAM3 NMS

    Args:
        config_path: Path to config file (for LoRA settings only). Not required if use_base_model=True.
        weights_path: Path to LoRA weights. Not required if use_base_model=True.
        val_data_dir: Direct path to validation data directory containing _annotations.coco.json
                      (e.g., /workspace/data2/valid)
        num_samples: Optional limit for number of samples (for debugging)
        use_base_model: If True, use original SAM3 model without LoRA (default: False)

    Example (with LoRA):
        validate(
            config_path="configs/full_lora_config.yaml",
            weights_path="outputs/sam3_lora_full/best_lora_weights.pt",
            val_data_dir="/workspace/data2/valid"
        )

    Example (base SAM3 model):
        validate(
            config_path=None,
            weights_path=None,
            val_data_dir="/workspace/data2/valid",
            use_base_model=True
        )
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    print("\nBuilding SAM3 model...")
    model = build_sam3_image_model(
        device=device.type,
        compile=False,
        checkpoint_path="/home/icclab/Documents/lqw/sam3/weights/sam3.pt",  # 设置直接加载权重
        load_from_HF=False,
        bpe_path="/home/icclab/Documents/lqw/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=False
    )

    # Patch RoPE for dynamic resolution support (enables non-1008 inputs)
    n_patched = patch_dynamic_rope(model)
    print(f"Dynamic RoPE patch applied to {n_patched} attention layers")
    
    # Load config for batch_size and other settings
    if use_base_model:
        # Use original SAM3 model without LoRA
        print("Using original SAM3 model (no LoRA)")
        stats = count_parameters(model)
        print(f"Total params: {stats['total_parameters']:,}")
        # Use default batch_size for base model
        batch_size = 1
    else:
        # Apply LoRA and load weights
        if config_path is None or weights_path is None:
            raise ValueError("config_path and weights_path are required when use_base_model=False")

        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Apply LoRA
        print("Applying LoRA configuration...")
        lora_cfg = config["lora"]
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
        model = apply_lora_to_model(model, lora_config)

        # Load weights
        print(f"\nLoading LoRA weights from {weights_path}...")
        load_lora_weights(model, weights_path)

        stats = count_parameters(model)
        print(f"Trainable params: {stats['trainable_parameters']:,} ({stats['trainable_percentage']:.2f}%)")

        # Get batch_size from config
        batch_size = config["training"]["batch_size"]

    model.to(device)
    model.eval()

    # Load validation data
    if dataset_type == "vaihingen":
        if vai_data_dir is None:
            raise ValueError("vai_data_dir is required when dataset_type='vaihingen'")
        print(f"\nLoading Vaihingen validation data from {vai_data_dir}...")
        val_ds = VaiRefDataset(
            data_dir=vai_data_dir,
            split=vai_split,
            variant=vai_variant,
            resolution=480
        )

        # for input_batch in val_ds:

        #     img = input_batch.images[0]
        #     q = input_batch.find_queries[0]
        #     obj = img.objects[0]

        #     # print(f"\n--- Sample {i} ---")
        #     print(f"  Image tensor: {img.data.shape}")
        #     print(f"  Query: {q.query_text[:60]}...")
        #     print(f"  Mask: {obj.segment.shape}, sum={obj.segment.sum().item():.0f} px")
        #     print(f"  Bbox (CxCyWH): {obj.bbox.tolist()}")
        #     print(f"  Original size: {q.inference_metadata.original_size}")
    
    else:
        print(f"\nLoading validation data from {val_data_dir}...")

        # Load COCO annotations directly
        from pathlib import Path
        ann_file = Path(val_data_dir) / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

        # Create a simple dataset class that loads from the directory directly
        val_ds = DirectCOCODataset(val_data_dir)

    if num_samples:
        print(f"\n[INFO] Limiting validation to {num_samples} samples for debugging")

    def collate_fn(batch):
        return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,  # Enable parallel data loading
        pin_memory=True  # Faster GPU transfer
    )

    # Create matcher for loss computation
    matcher = BinaryHungarianMatcherV2(
        cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True
    )

    # Run validation
    print("\n" + "="*80)
    print("RUNNING VALIDATION")
    print("="*80)

    all_image_ids = []
    all_coco_predictions = []
    CHUNK_SIZE = 100  # Process 100 images per chunk to limit memory
    first_batch_scores = None  # For debug when no predictions survive NMS

    # Use automatic mixed precision for faster inference
    use_amp = device.type == 'cuda'

    chunk_predictions = []
    chunk_image_ids = []

    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(tqdm(val_loader, desc="Validation")):
            if num_samples and batch_idx * batch_size >= num_samples:
                break

            input_batch = batch_dict["input"]
            input_batch = move_to_device(input_batch, device)
            # Forward pass with optional AMP
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs_list = model(input_batch)
            else:
                outputs_list = model(input_batch)

            # Extract predictions
            with SAM3Output.iteration_mode(
                outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
            ) as outputs_iter:
                final_stage = list(outputs_iter)[-1]
                final_outputs = final_stage[-1]

                batch_size_actual = final_outputs['pred_logits'].shape[0]

                for i in range(batch_size_actual):
                    img_id = batch_idx * batch_size + i
                    all_image_ids.append(img_id)
                    chunk_image_ids.append(img_id)
                    chunk_predictions.append({
                        'pred_logits': final_outputs['pred_logits'][i].detach().cpu(),
                        'pred_boxes': final_outputs['pred_boxes'][i].detach().cpu(),
                        'pred_masks': final_outputs['pred_masks'][i].detach().cpu()
                    })

                # Capture first batch scores for debug
                if batch_idx == 0:
                    first_batch_scores = torch.sigmoid(final_outputs['pred_logits'].detach().cpu())

            # Free GPU memory from this batch
            del input_batch, outputs_list, final_outputs, batch_dict

            # Flush chunk to compact COCO format to avoid accumulating raw tensors
            if len(chunk_predictions) >= CHUNK_SIZE:
                chunk_coco = convert_predictions_to_coco_format_original_res(
                    chunk_predictions, 
                    chunk_image_ids,
                    val_ds,
                    score_threshold=prob_threshold,
                    iou_threshold=merge_iou,
                    )
                # chunk_coco = convert_predictions_to_coco_format(
                #     chunk_predictions, chunk_image_ids,
                #     resolution=288,
                #     prob_threshold=prob_threshold,
                #     nms_iou_threshold=nms_iou,
                #     max_detections=100,
                #     merge_cracks=merge_cracks,
                #     merge_iou_threshold=merge_iou
                # )
                all_coco_predictions.extend(chunk_coco)
                del chunk_predictions, chunk_image_ids, chunk_coco
                chunk_predictions = []
                chunk_image_ids = []
                torch.cuda.empty_cache()
                gc.collect()

    # Flush remaining predictions
    if chunk_predictions:
        chunk_coco = convert_predictions_to_coco_format_original_res(
            chunk_predictions, 
            chunk_image_ids,
            val_ds,
            score_threshold=prob_threshold,
            iou_threshold=merge_iou,
            )
        # chunk_coco = convert_predictions_to_coco_format(
        #     chunk_predictions, chunk_image_ids,
        #     resolution=288,
        #     prob_threshold=prob_threshold,
        #     nms_iou_threshold=nms_iou,
        #     max_detections=100,
        #     merge_cracks=merge_cracks,
        #     merge_iou_threshold=merge_iou
        # )
        all_coco_predictions.extend(chunk_coco)
        del chunk_predictions, chunk_image_ids, chunk_coco
        torch.cuda.empty_cache()
        gc.collect()

    coco_predictions = all_coco_predictions
    del all_coco_predictions
    print(f"\n[INFO] Collected {len(coco_predictions)} COCO-format predictions from {len(all_image_ids)} images")

    # If no predictions survived thresholding, analyze first batch scores for debugging
    if len(coco_predictions) == 0 and first_batch_scores is not None:
        print(f"\n[DEBUG] No predictions after NMS — analyzing first batch scores...")
        scores = first_batch_scores.squeeze(-1)
        print(f"[DEBUG] Raw pred_logits shape: {first_batch_scores.shape}")
        print(f"[DEBUG] Sigmoid scores: min={scores.min().item():.6f}, "
              f"max={scores.max().item():.6f}, "
              f"mean={scores.mean().item():.6f}")
        print(f"[DEBUG] Median score: {scores.median().item():.6f}")
        print(f"[DEBUG] Fraction > 0.1: {(scores > 0.1).float().mean().item():.3f}")
        print(f"[DEBUG] Fraction > 0.3: {(scores > 0.3).float().mean().item():.3f}")
        print(f"[INFO] Try: --prob-threshold 0.05 or 0.1")
        del first_batch_scores

    # Compute metrics
    print("\n" + "="*80)
    print("COMPUTING METRICS")
    print("="*80)

    # Create COCO ground truth (downsampled to 288×288)
    print(f"\n[INFO] Creating ground truth from validation dataset...")
    coco_gt_dict = create_coco_gt_from_dataset_original_res(val_ds)
    # coco_gt_dict = create_coco_gt_from_dataset(
    #     val_ds,
    #     image_ids=all_image_ids,
    #     mask_resolution=288
    # )

    # Filter predictions to only include images that exist in the ground truth
    gt_image_ids = set(img['id'] for img in coco_gt_dict['images'])
    filtered_predictions = [p for p in coco_predictions if p['image_id'] in gt_image_ids]
    if len(filtered_predictions) < len(coco_predictions):
        print(f"[INFO] Skipped {len(coco_predictions) - len(filtered_predictions)} predictions with no GT")
    coco_predictions = filtered_predictions

    if merge_cracks:
        print(f"\n[INFO] Total predictions after CRACK MERGING: {len(coco_predictions)}")
    else:
        print(f"\n[INFO] Total predictions after SAM3 NMS filtering: {len(coco_predictions)}")

    if len(coco_predictions) > 0:
        print("here is true, we went into coco_predictions")
        # Save temporary files for COCO evaluation
        import tempfile
        import os

        # Create temp directory for evaluation files
        temp_dir = tempfile.mkdtemp(prefix="sam3_eval_")
        gt_file = os.path.join(temp_dir, "gt.json")
        pred_file = os.path.join(temp_dir, "pred.json")

        with open(gt_file, 'w') as f:
            json.dump(coco_gt_dict, f)
        with open(pred_file, 'w') as f:
            json.dump(coco_predictions, f)

        # Compute mAP
        print("\n" + "="*80)
        print("COCO mAP EVALUATION")
        print("="*80)

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                coco_gt = COCO(str(gt_file))
                coco_dt = coco_gt.loadRes(str(pred_file))
                coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
                coco_eval.params.useCats = False
                coco_eval.evaluate()
                coco_eval.accumulate()

        # Print mAP results
        coco_eval.summarize()

        map_segm = coco_eval.stats[0]
        map50_segm = coco_eval.stats[1]
        map75_segm = coco_eval.stats[2]

        # Compute cgF1
        print("\n" + "="*80)
        print("cgF1 EVALUATION")
        print("="*80)

        cgf1_evaluator = CGF1Evaluator(
            gt_path=str(gt_file),
            iou_type='segm',
            verbose=True
        )
        cgf1_results = cgf1_evaluator.evaluate(str(pred_file))

        cgf1 = cgf1_results.get('cgF1_eval_segm_cgF1', 0.0)
        cgf1_50 = cgf1_results.get('cgF1_eval_segm_cgF1@0.5', 0.0)
        cgf1_75 = cgf1_results.get('cgF1_eval_segm_cgF1@0.75', 0.0)

        # Compute MIoU (pixel-level Mean IoU)
        print("\n" + "="*80)
        print("MIoU EVALUATION (Pixel-Level)")
        print("="*80)

        miou_results = compute_miou(coco_gt_dict, coco_predictions)

        # Print summary
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"mAP (IoU 0.50:0.95): {map_segm:.4f}")
        print(f"mAP@50: {map50_segm:.4f}")
        print(f"mAP@75: {map75_segm:.4f}")
        print(f"cgF1 (IoU 0.50:0.95): {cgf1:.4f}")
        print(f"cgF1@50: {cgf1_50:.4f}")
        print(f"cgF1@75: {cgf1_75:.4f}")
        print(f"---")
        print(f"mIoU: {miou_results['miou']:.4f}")
        print(f"IoU (object): {miou_results['iou_object']:.4f}")
        print(f"IoU (background): {miou_results['iou_background']:.4f}")
        print("="*80)

        # Cleanup temporary files
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

    else:
        print("\n[ERROR] No predictions generated! Cannot compute metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone validation script for SAM3 LoRA model with full metrics (mAP, cgF1) and SAM3 NMS"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (for LoRA settings). Not required if --use-base-model is set."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to LoRA weights file. Not required if --use-base-model is set."
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        # required=True,
        default=None,
        help="Direct path to validation data directory containing _annotations.coco.json (e.g., /workspace/data2/valid)"
    )
    parser.add_argument(
        "--use-base-model",
        action="store_true",
        help="Use original SAM3 model without LoRA (for baseline comparison)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit validation to N samples (for debugging)"
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.3,
        help="Probability threshold for filtering predictions (default: 0.3)"
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold (default: 0.7)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Enable aggressive merging of overlapping segments (recommended for crack detection)"
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.15,
        help="IoU threshold for merging overlapping predictions (default: 0.15, lower = more aggressive)"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="coco",
        choices=["coco", "vaihingen"],
        help="Dataset type: 'coco' (COCO format) or 'vaihingen' (VaihingenRef)"
    )
    parser.add_argument(
        "--vai-data-dir",
        type=str,
        default=None,
        help="Path to VaihingenRef dataset root (required when --dataset-type=vaihingen)"
    )
    parser.add_argument(
        "--vai-split",
        type=str,
        default="valid",
        choices=["train", "valid", "test"],
        help="Vaihingen dataset split (default: valid)"
    )
    parser.add_argument(
        "--vai-variant",
        type=str,
        default="standard",
        choices=["concept", "simple", "standard", "complex"],
        help="Vaihingen text variant (default: standard)"
    )
    args = parser.parse_args()

    # Validate argument combinations
    if not args.use_base_model:
        if args.config is None or args.weights is None:
            parser.error("--config and --weights are required when not using --use-base-model")

    validate(
        config_path=args.config,
        weights_path=args.weights,
        val_data_dir=args.val_data_dir,
        num_samples=args.num_samples,
        prob_threshold=args.prob_threshold,
        nms_iou=args.nms_iou,
        merge_cracks=args.merge,
        merge_iou=args.merge_iou,
        use_base_model=args.use_base_model,
        dataset_type=args.dataset_type,
        vai_data_dir=args.vai_data_dir,
        vai_split=args.vai_split,
        vai_variant=args.vai_variant,
    )



# python scripts/inference/validate_sam3_lora.py --config configs/vaihingen_lora_config.yaml --weights /home/icclab/Documents/lqw/sam3/weights/sam3.pt --dataset-type vaihingen --vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/VaihingenRef --vai-split valid --vai-variant concept --prob-threshold 0.1
# python scripts/inference/validate_sam3_lora.py --config configs/vaihingen_lora_config.yaml --weights outputs/sam3_lora_vaihingen/best_lora_weights.pt --dataset-type vaihingen --vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/VaihingenRef --vai-split valid --vai-variant concept
# python scripts/inference/validate_sam3_lora.py --config configs/vaihingen_lora_config.yaml --weights outputs/sam3_lora_vaihingen_simple/best_lora_weights.pt --dataset-type vaihingen --vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/VaihingenRef --vai-split valid --vai-variant simple --prob-threshold 0.1
# python scripts/inference/validate_sam3_lora.py --config configs/vaihingen_lora_config.yaml --weights outputs/sam3_lora_vaihingen_stand/best_lora_weights.pt --dataset-type vaihingen --vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/VaihingenRef --vai-split valid --vai-variant standard