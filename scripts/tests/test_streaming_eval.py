"""Compare streaming pixel evaluation with the COCO path, without a checkpoint."""

import contextlib
import io
import logging
import os
import unittest
from unittest.mock import patch

import numpy as np
import pycocotools.mask as mask_utils
import torch

from sam3.model.model_misc import SAM3Output
from sam3.train.data.sam3_image_dataset import (
    Datapoint, FindQueryLoaded, Image, InferenceMetadata, Object,
)
from scripts.tools.streaming_eval import (
    StreamingIoU, collate_for_iou, prediction_union_original_res,
    target_union_original_res,
)
from scripts.tools.utils import (
    compute_miou, convert_predictions_to_coco_format_original_res,
    create_coco_gt_from_dataset_original_res,
)


def make_point(index=0):
    mask = torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.bool)
    obj = Object(bbox=torch.tensor([0.5, 0.5, 1.0, 1.0]), area=2, object_id=0, segment=mask)
    query = FindQueryLoaded(
        query_text="building", image_id=0, object_ids_output=[0], is_exhaustive=True,
        inference_metadata=InferenceMetadata(
            coco_image_id=index, original_image_id=index, original_category_id=0,
            original_size=(2, 3), object_id=0, frame_index=0,
        ),
    )
    return Datapoint(
        images=[Image(data=torch.zeros(3, 2, 3), objects=[obj], size=(2, 3))],
        find_queries=[query],
    )


class SinglePointDataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return make_point(index)


class CountingDataset:
    def __init__(self):
        self.reads = []

    def __len__(self):
        return 5

    def __getitem__(self, index):
        self.reads.append(index)
        return make_point(index)


class PerfectModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, batch):
        if batch.find_targets[0].boxes.numel() or batch.find_targets[0].segments is not None:
            raise AssertionError("Ground truth entered model input")
        count = len(batch.find_inputs[0].img_ids)
        self.batch_sizes.append(count)
        result = SAM3Output(iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE)
        result.append([dict(
            pred_logits=torch.full((count, 1, 1), 8.0),
            pred_boxes=torch.tensor([[[0.5, 0.5, 1.0, 1.0]]]).repeat(count, 1, 1),
            pred_masks=torch.tensor([[[[8.0, -8.0, -8.0], [8.0, -8.0, -8.0]]]]).repeat(count, 1, 1, 1),
        )])
        return result


class StreamingEvaluationTests(unittest.TestCase):
    def test_union_matches_coco_masks_with_clipped_and_invalid_boxes(self):
        generator = torch.Generator().manual_seed(19)
        for dtype in (torch.float32, torch.float16):
            prediction = dict(
                pred_logits=torch.randn((35, 1), generator=generator).to(dtype),
                pred_boxes=torch.tensor([[0.5, 0.5, 1.0, 1.0]] * 35),
                pred_masks=torch.randn((35, 5, 7), generator=generator).to(dtype),
                original_size=(17, 31),
            )
            prediction['pred_boxes'][0] = torch.tensor([1.1, 0.5, 0.1, 1.0])
            prediction['pred_boxes'][1] = torch.tensor([0.5, 0.5, 0.0, 0.0])
            prediction['pred_logits'][1] = 8.0
            for merge in (False, True):
                for threshold in (0.0, 0.3, 1.0):
                    with self.subTest(dtype=dtype, merge=merge, threshold=threshold):
                        coco = convert_predictions_to_coco_format_original_res(
                            [prediction], [0], None, score_threshold=threshold,
                            merge_overlaps=merge, iou_threshold=0.15,
                        )
                        expected = np.zeros((17, 31), dtype=bool)
                        for item in coco:
                            expected |= mask_utils.decode(item['segmentation']).astype(bool)
                        actual = prediction_union_original_res(
                            prediction['pred_logits'], prediction['pred_boxes'],
                            prediction['pred_masks'], (17, 31), threshold, merge,
                        )
                        np.testing.assert_array_equal(actual, expected)

    def test_merging_can_skip_pairwise_work_when_all_boxes_are_valid(self):
        with patch('scripts.tools.utils.merge_overlapping_masks', side_effect=AssertionError):
            union = prediction_union_original_res(
                torch.ones((20, 1)), torch.tensor([[0.5, 0.5, 1.0, 1.0]] * 20),
                torch.ones((20, 2, 3)), (4, 6), merge_overlaps=True,
            )
        self.assertTrue(union.all())

    def test_ground_truth_and_metric_totals_match_coco_path(self):
        point = make_point()
        gt = create_coco_gt_from_dataset_original_res(SinglePointDataset())
        expected_target = mask_utils.decode(gt['annotations'][0]['segmentation']).astype(bool)
        actual_target = target_union_original_res([point.images[0].objects[0].segment], (2, 3))
        np.testing.assert_array_equal(actual_target, expected_target)
        metrics = StreamingIoU()
        metrics.update(np.zeros((2, 3), dtype=bool), actual_target)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            expected_metrics = compute_miou(gt, [])
        for key, expected in expected_metrics.items():
            self.assertAlmostEqual(metrics.compute()[key], expected)

    def test_collation_keeps_targets_outside_model_input(self):
        point = make_point()
        expected = point.images[0].objects[0].segment.numpy().copy()
        wrapped = collate_for_iou([point])
        np.testing.assert_array_equal(wrapped['pixel_targets'][0], expected)
        target = wrapped['input'].find_targets[0]
        self.assertEqual(target.num_boxes.tolist(), [0])
        self.assertEqual(target.boxes.numel(), 0)
        self.assertIsNone(target.segments)
        self.assertIsNone(wrapped['input'].raw_images)

    def test_empty_prediction_and_empty_target_keep_legacy_zero_iou(self):
        metrics = StreamingIoU()
        metrics.update(np.zeros((2, 3), bool), np.zeros((2, 3), bool))
        self.assertEqual(metrics.compute()['miou'], 0.0)
        self.assertEqual(metrics.compute()['total_tn'], 6)

    def test_runner_reads_each_selected_sample_once_and_handles_partial_batch(self):
        # Restore the script's CUDA_VISIBLE_DEVICES setting after importing it.
        with patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
            from scripts.inference.isprs_sam3_lora import validate_streaming
        dataset, model = CountingDataset(), PerfectModel()
        with contextlib.redirect_stderr(io.StringIO()):
            results = validate_streaming(
                model, dataset, torch.device('cpu'), logging.getLogger(__name__),
                batch_size=2, num_workers=0, num_samples=3,
                prob_threshold=0.3, merge_cracks=True, merge_iou=0.15,
            )
        self.assertEqual(dataset.reads, [0, 1, 2])
        self.assertEqual(model.batch_sizes, [2, 1])
        self.assertEqual(results['miou'], 1.0)
        self.assertEqual(results['oiou'], 1.0)
        self.assertEqual(results['total_tp'], 6)


if __name__ == '__main__':
    unittest.main()
