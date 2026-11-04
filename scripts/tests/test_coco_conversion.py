"""CPU regression checks; no model checkpoint or dataset files are required.

Run from the project root with:
    python -m unittest discover -s scripts/tests -p test_coco_conversion.py -v
"""

import unittest
from types import SimpleNamespace

import numpy as np
import pycocotools.mask as mask_utils
import torch

from sam3.train.masks_ops import rle_encode
from scripts.tools.utils import (
    convert_predictions_to_coco_format_original_res,
    merge_overlapping_masks,
)


class MetadataDataset:
    def __init__(self):
        self.reads = []

    def __getitem__(self, image_id):
        self.reads.append(image_id)
        return SimpleNamespace(
            find_queries=[
                SimpleNamespace(
                    inference_metadata=SimpleNamespace(original_size=(4, 6))
                )
            ]
        )


class CocoConversionTests(unittest.TestCase):
    @staticmethod
    def predictions():
        # Two confident duplicates, plus a low-confidence background mask.
        return {
            "pred_logits": torch.tensor([[2.0], [1.0], [-2.0]]),
            "pred_boxes": torch.tensor([[0.5, 0.5, 1.0, 1.0]] * 3),
            "pred_masks": torch.tensor(
                [[[12.0, -12.0], [12.0, -12.0]]] * 2
                + [[[-12.0, -12.0], [-12.0, -12.0]]]
            ),
            "original_size": (4, 6),
        }

    def test_cached_size_and_native_rle_preserve_non_square_mask(self):
        dataset = MetadataDataset()
        predictions = convert_predictions_to_coco_format_original_res(
            [self.predictions()], [17], dataset,
            score_threshold=0.5, merge_overlaps=False,
        )
        self.assertEqual(dataset.reads, [])
        self.assertEqual(len(predictions), 2)
        expected_mask = torch.zeros((1, 4, 6), dtype=torch.bool)
        expected_mask[:, :, :3] = True
        expected_rle = rle_encode(expected_mask)[0]
        for prediction in predictions:
            self.assertEqual(prediction["segmentation"], expected_rle)
            self.assertEqual(prediction["bbox"], [0.0, 0.0, 6.0, 4.0])
            self.assertEqual(prediction["image_id"], 17)
            np.testing.assert_array_equal(
                mask_utils.decode(prediction["segmentation"]), expected_mask[0].numpy()
            )

    def test_merge_option_controls_duplicate_predictions(self):
        predictions = convert_predictions_to_coco_format_original_res(
            [self.predictions()], [0], MetadataDataset(),
            score_threshold=0.5, merge_overlaps=True,
        )
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["score"], torch.sigmoid(torch.tensor(2.0)).item())

    def test_older_callers_can_read_size_from_dataset(self):
        dataset = MetadataDataset()
        prediction = self.predictions()
        del prediction["original_size"]
        result = convert_predictions_to_coco_format_original_res(
            [prediction], [3], dataset, score_threshold=0.5,
        )
        self.assertEqual(dataset.reads, [3])
        self.assertEqual(result[0]["segmentation"]["size"], [4, 6])

    def test_empty_and_all_filtered_predictions(self):
        dataset = MetadataDataset()
        self.assertEqual(
            convert_predictions_to_coco_format_original_res([], [], dataset), []
        )
        self.assertEqual(
            convert_predictions_to_coco_format_original_res(
                [self.predictions()], [0], dataset, score_threshold=1.0,
            ),
            [],
        )
        self.assertEqual(dataset.reads, [])

    def test_greedy_merge_uses_updated_mask_and_keeps_seed_box(self):
        # A overlaps B; C only qualifies after A and B have been merged.
        masks = torch.tensor([
            [[0, 0, 1, 1, 0, 0]],  # C
            [[1, 1, 0, 0, 0, 0]],  # A
            [[0, 1, 1, 0, 0, 0]],  # B
        ], dtype=torch.bool)
        original_masks = masks.clone()
        scores = torch.tensor([0.7, 0.9, 0.8])
        boxes = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        merged, merged_scores, merged_boxes = merge_overlapping_masks(
            masks, scores, boxes, iou_threshold=0.2,
        )
        self.assertTrue(torch.equal(merged, masks.any(dim=0, keepdim=True)))
        self.assertTrue(torch.equal(merged_scores, scores[1:2]))
        self.assertTrue(torch.equal(merged_boxes, boxes[1:2]))
        self.assertTrue(torch.equal(masks, original_masks))

    def test_strict_iou_threshold_and_empty_masks(self):
        masks = torch.tensor([[[1, 1, 0]], [[0, 1, 1]]], dtype=torch.bool)
        scores = torch.tensor([0.9, 0.8])
        boxes = torch.ones((2, 4))
        merged, _, _ = merge_overlapping_masks(masks, scores, boxes, 1 / 3)
        self.assertTrue(torch.equal(merged, masks))
        empty = torch.zeros_like(masks)
        merged, _, _ = merge_overlapping_masks(empty, scores, boxes, 0.0)
        self.assertTrue(torch.equal(merged, empty))
        merged, _, _ = merge_overlapping_masks(empty[:0], scores[:0], boxes[:0])
        self.assertEqual(merged.shape, (0, 1, 3))


if __name__ == "__main__":
    unittest.main()
