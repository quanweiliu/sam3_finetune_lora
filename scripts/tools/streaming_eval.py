"""Pixel IoU evaluation without COCO encoding or a second dataset pass."""

import numpy as np
import torch
import torch.nn.functional as F


def target_union_original_res(segments, size):
    """Match the legacy GT policy: bilinear resize each mask, then threshold."""
    target = np.zeros(size, dtype=bool)
    for segment in segments:
        if segment is None:
            continue
        resized = F.interpolate(
            segment.detach().cpu()[None, None].float(), size=size,
            mode="bilinear", align_corners=False,
        )[0, 0]
        target |= (resized > 0.5).numpy()
    return target


def collate_for_iou(points):
    """Prepare pixel targets in loader workers, keeping them outside model input."""
    from sam3.train.data.collator import collate_fn_api

    targets = []
    for point in points:
        for query in point.find_queries:
            if query.query_processing_order != 0:
                raise ValueError("Streaming IoU expects single-stage image queries")
            size = tuple(query.inference_metadata.original_size)
            objects = point.images[query.image_id].objects
            targets.append(target_union_original_res(
                [objects[idx].segment for idx in query.object_ids_output], size,
            ))
            query.object_ids_output = []
            query.semantic_target = None
            query.is_exhaustive = False
            query.is_pixel_exhaustive = False
        for image in point.images:
            image.objects = []
        point.raw_images = None
    wrapped = collate_fn_api(points, dict_key="input", with_seg_masks=False)
    wrapped["pixel_targets"] = targets
    return wrapped


def _valid_boxes(boxes, size):
    """Use the same pixel-size cutoff and clipping as the COCO converter."""
    height, width = size
    valid = []
    for cx, cy, bw, bh in boxes.detach().cpu().tolist():
        x = max(0, min((cx - bw / 2) * width, width))
        y = max(0, min((cy - bh / 2) * height, height))
        w = max(0, min(bw * width, width - x))
        h = max(0, min(bh * height, height - y))
        valid.append(w >= 1 and h >= 1)
    return torch.tensor(valid, dtype=torch.bool)


def prediction_union_original_res(logits, boxes, masks, size, score_threshold=0.3,
                                  merge_overlaps=False, iou_threshold=0.15,
                                  mask_chunk_size=16):
    """Match legacy masks while storing only one prediction union per query.

    Preserve sigmoid-before-resize, score precision, and box clipping. Merging
    preserves the union unless an invalid seed box causes a whole group to be
    discarded; only that case requires the original greedy merging operation.
    """
    if mask_chunk_size < 1:
        raise ValueError("mask_chunk_size must be positive")
    scores = logits.detach().cpu().sigmoid().squeeze(-1)
    selected = torch.where(scores > score_threshold)[0]
    union = np.zeros(size, dtype=bool)
    if not len(selected):
        return union

    selected_boxes = boxes.detach().cpu()[selected]
    valid_boxes = _valid_boxes(selected_boxes, size)
    needs_merging = merge_overlaps and not bool(valid_boxes.all())
    if not merge_overlaps:
        selected = selected[valid_boxes]

    # Transfer only the selected low-resolution masks, once per query.
    selected_masks = masks.detach()[selected.to(masks.device)].cpu()
    binary_chunks = []
    for mask_chunk in selected_masks.split(mask_chunk_size):
        if not len(mask_chunk):
            continue
        # Keep the original CPU sigmoid precision, including float16 outputs.
        probabilities = mask_chunk.sigmoid()
        resized = F.interpolate(
            probabilities[:, None].float(), size=size,
            mode="bilinear", align_corners=False,
        )[:, 0]
        binary = resized > 0.5
        if needs_merging:
            binary_chunks.append(binary)
        else:
            union |= binary.numpy().any(axis=0)

    if needs_merging:
        from .utils import merge_overlapping_masks

        binary, _, merged_boxes = merge_overlapping_masks(
            torch.cat(binary_chunks), scores[selected], selected_boxes,
            iou_threshold=iou_threshold,
        )
        binary = binary[_valid_boxes(merged_boxes, size)]
        if len(binary):
            union |= binary.numpy().any(axis=0)
    return union


class StreamingIoU:
    """Accumulate the same mIoU/oIoU and confusion counts as compute_miou."""

    def __init__(self):
        self.count = 0
        self.iou_sum = 0.0
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, prediction, target):
        prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
        if prediction.shape != target.shape:
            raise ValueError("Prediction/target dimensions differ")
        tp = int(np.count_nonzero(prediction & target))
        fp = int(np.count_nonzero(prediction)) - tp
        fn = int(np.count_nonzero(target)) - tp
        union = tp + fp + fn
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += prediction.size - union
        self.iou_sum += tp / union if union else 0.0
        self.count += 1

    def compute(self):
        if not self.count:
            raise ValueError("No evaluated queries")
        union = self.tp + self.fp + self.fn
        background_union = self.tn + self.fp + self.fn
        return dict(
            miou=self.iou_sum / self.count,
            oiou=self.tp / union if union else 0.0,
            iou_object=self.tp / union if union else 0.0,
            iou_background=self.tn / background_union if background_union else 0.0,
            total_tp=self.tp, total_fp=self.fp, total_fn=self.fn, total_tn=self.tn,
        )
