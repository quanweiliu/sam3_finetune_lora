"""VPRef union-mask JSONL queries in SAM3's Datapoint format.

v0.3 through v0.8 datasets share this loader and the v0.3 union-mask schema.
Content versions are recorded separately; a new content version needs no new loader.

One record is one English query and one union-mask target, even when that mask
contains several connected regions. Region IDs and landmark masks are annotation
metadata, not input prompts. The source package is never modified.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset

from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image,
    InferenceMetadata,
    Object,
)

from torch.utils.data import DataLoader, Dataset

class VPRefV03Dataset(Dataset):
    def __init__(
        self,
        data_dir,
        manifest="train.jsonl",
        domain="Vaihingen",
        resolution=480,
        target_mode="union",
        rotation_degrees=(0,),
        allow_unreviewed_proposals=False,
        split="train",
        task_families=None,
    ):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.domain = domain

        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val' or 'test'")
        
        self.split = split
        self.resolution = int(resolution)
        self.rotation_degrees = tuple(rotation_degrees)
        # if not isinstance(allow_unreviewed_proposals, bool):
        #     raise ValueError("allow_unreviewed_proposals must be a boolean")
        
        self.allow_unreviewed_proposals = allow_unreviewed_proposals
        self.uses_unreviewed_proposals = False

        if target_mode != "union":
            raise ValueError("This first VPRef adapter supports target_mode='union' only")
        # if self.resolution <= 0:
        #     raise ValueError("resolution must be positive")
        if not self.rotation_degrees or any(d not in (0, 90, 180, 270)
                                            for d in self.rotation_degrees):
            raise ValueError("rotation_degrees must contain only 0, 90, 180, 270")
        # if domain not in (None, "Vaihingen", "Potsdam"):
        #     raise ValueError(f"Unknown domain: {domain}")
        
        self.manifest_path = self._asset(manifest)
        records = [json.loads(line) for line in self.manifest_path.read_text().splitlines()
                   if line.strip()]
        self.samples = [row for row in records if domain is None or row["domain"] == domain]

        if not self.samples:
            raise ValueError(f"No queries for domain={domain!r} in {self.manifest_path}")
        self.available_task_families = tuple(sorted({row["task_family"] for row in self.samples
                                                   if isinstance(row.get("task_family"), str)}))
        
        if isinstance(task_families, str):
            task_families = [task_families]
        if task_families is not None:
            if (not isinstance(task_families, (list, tuple))
                    or any(not isinstance(name, str) or not name.strip() for name in task_families)):
                raise ValueError("task_families must be a name, a list of names, or null/[] for all")
            task_families = tuple(sorted(set(task_families))) or None
        
        self.task_families = task_families
        if self.task_families:
            missing = set(self.task_families) - set(self.available_task_families)
            if missing:
                raise ValueError(f"Requested task_families not available in {self.manifest_path}: "
                                 f"{sorted(missing)}. Available: {list(self.available_task_families)}")
            self.samples = [row for row in self.samples if row.get("task_family") in self.task_families]
        self.task_family_counts = dict(sorted(Counter(row.get("task_family", "unspecified")
                                                     for row in self.samples).items()))
        self.has_ignore_masks = any(row.get("ignore_mask_path") for row in self.samples)
        ids = [row["sample_id"] for row in self.samples]
        
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate sample IDs in manifest")
        self.image_ids = {key: i for i, key in enumerate(sorted(
            {row["image_id"] for row in self.samples}))}
        
        for row in self.samples:
            if row.get("schema_version") != "vpref-v0.3-union-mask-1.0":
                raise ValueError(f"Unsupported schema: {row['sample_id']}")
            role = {"train": "training", "val": "validation", "test": "testing"}[split]
            eligibility = "eligible_for_training" if split == "train" else "eligible_for_evaluation"
            accepted = (row.get("review_status") == "accepted"
                        and row.get(eligibility) is True
                        and row.get("usage") == f"union_mask_{role}_pilot")
            proposal = (allow_unreviewed_proposals
                        and row.get("review_status") == "pending_human_review"
                        and row.get("eligible_for_training") is False
                        and row.get("eligible_for_evaluation") is False
                        and row.get("answer_role") == "geometric_proposal_only_not_gold"
                        and row.get("usage") == f"unreviewed_union_{role}_experiment")

            # print(f"Row split: {row.get('split')}, Expected split: {split}")
            if (row.get("split") != split or not (accepted or proposal)
                    or not row.get("target_region_ids")):
                raise ValueError(f"Not an accepted nonempty {role} query: {row['sample_id']}")
            
            self.uses_unreviewed_proposals |= bool(proposal)
            
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"Missing English: {row['sample_id']}")
            self._asset(row["image_path"])
            self._asset(row["mask_path"])
            
            # if row.get("dataset_version") == "0.7" and not row.get("ignore_mask_path"):
            #     raise ValueError(f"v0.7 requires an ignore mask: {row['sample_id']}")
            
            if row.get("ignore_mask_path"):
                self._asset(row["ignore_mask_path"])

            variants = {v["rotation_degrees_ccw"]: v
                        for v in row.get("rotation_text_variants", [])}
            
            for degrees in self.rotation_degrees:
                if degrees == 0:
                    continue  # Original English is authoritative for the unrotated image.
                variant = variants.get(degrees)
                if (variant is None or not variant.get("text")
                        or variant.get("target_region_ids") != row["target_region_ids"]):
                    raise ValueError(f"Missing synchronized {degrees}° English: {row['sample_id']}")

    def _asset(self, relative):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Expected package-relative path: {relative}")
        path = (self.data_dir / relative).resolve()
        if not path.is_relative_to(self.data_dir) or not path.is_file():
            raise FileNotFoundError(f"Missing or external package asset: {path}")
        return path

    def __len__(self):
        return len(self.samples)

    def read_ignore_mask(self, row):
        """Native-resolution void pixels, never an input prompt or background label."""
        shape = (row["height"], row["width"])
        if not row.get("ignore_mask_path"):
            return np.zeros(shape, dtype=bool)
        with PILImage.open(self._asset(row["ignore_mask_path"])) as im:
            mask = np.asarray(im).copy()
        if mask.shape != shape or not set(np.unique(mask)).issubset({0, 255}):
            raise ValueError(f"Expected a native-size binary ignore mask: {row['sample_id']}")
        return mask > 0

    def __getitem__(self, index):
        row = self.samples[index]
        
        with PILImage.open(self._asset(row["image_path"])) as im:
            image = im.convert("RGB")
        
        with PILImage.open(self._asset(row["mask_path"])) as im:
            array = np.asarray(im).copy()
        
        if array.ndim != 2 or not set(np.unique(array)).issubset({0, 255}):
            raise ValueError(f"Expected 0/255 single-channel mask: {row['sample_id']}")
        orig_w, orig_h = image.size
        
        # if array.shape != (orig_h, orig_w) or (orig_w, orig_h) != (row["width"], row["height"]):
        #     raise ValueError(f"Image/mask/manifest dimensions differ: {row['sample_id']}")
        ignored = self.read_ignore_mask(row)
        
        if np.any(ignored & (array > 0)):
            raise ValueError(f"Positive target overlaps ignored pixels: {row['sample_id']}")
        degrees = self.rotation_degrees[torch.randint(len(self.rotation_degrees), ()).item()]
        text = row["text"]
        
        if degrees:
            # np.rot90 is counterclockwise, matching the packaged language variants.
            image = PILImage.fromarray(np.rot90(np.asarray(image), degrees // 90).copy())
            array = np.rot90(array, degrees // 90).copy()
            ignored = np.rot90(ignored, degrees // 90).copy()
            text = next(v["text"] for v in row["rotation_text_variants"] if v["rotation_degrees_ccw"] == degrees)

        resized = image.resize((self.resolution, self.resolution), PILImage.Resampling.BILINEAR)
        image_tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1)
        image_tensor = image_tensor.to(torch.float32).div_(127.5).sub_(1.0)
        resized_mask = PILImage.fromarray(array).resize(
            (self.resolution, self.resolution), PILImage.Resampling.NEAREST)
        segment = torch.from_numpy(np.asarray(resized_mask).copy() > 0)
        bbox, area = self._bbox_and_area(segment)
        obj = Object(bbox=bbox, area=area, object_id=0, segment=segment)

        query = FindQueryLoaded(
            query_text=text,
            image_id=0,
            object_ids_output=[0],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=self.image_ids[row["image_id"]],
                original_image_id=self.image_ids[row["image_id"]],
                original_category_id=0,
                original_size=(orig_h, orig_w),
                object_id=index,
                frame_index=0,
            ),
        )

        point = Datapoint(
            images=[Image(data=image_tensor, 
                          objects=[obj],
                          size=(self.resolution, self.resolution))],
            find_queries=[query], 
            raw_images=[image],
        )

        if row.get("ignore_mask_path"):
            resized_ignore = PILImage.fromarray(ignored.astype(np.uint8)).resize(
                (self.resolution, self.resolution), PILImage.Resampling.NEAREST)
            # Kept outside the SAM3 dataclass fields; the training collator extracts
            # this annotation separately and never feeds it through model inputs.
            point.vpref_valid_pixels = torch.from_numpy(np.asarray(resized_ignore).copy() == 0)
        return point

    @staticmethod
    def _bbox_and_area(segment):
        points = torch.nonzero(segment, as_tuple=False)
        if not len(points):
            raise ValueError("A positive training mask became empty")
        height, width = segment.shape
        lo = points.amin(0).to(torch.float32)
        hi = points.amax(0).to(torch.float32) + 1  # Exclusive maximum includes the last pixel.
        x1, y1, x2, y2 = lo[1], lo[0], hi[1], hi[0]
        bbox = torch.stack(((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                            (x2 - x1) / width, (y2 - y1) / height))
        return bbox, float(len(points))


def collate_vpref(points):
    from sam3.train.data.collator import collate_fn_api
    batch = collate_fn_api(points, dict_key="input", with_seg_masks=True)
    if any(hasattr(point, "vpref_valid_pixels") for point in points):
        batch["vpref_valid_pixels"] = torch.stack([
            getattr(point, "vpref_valid_pixels", torch.ones_like(point.images[0].objects[0].segment))
            for point in points])
    return batch

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/icclab/Documents/lqw/DatasetMMF/VPRef_Vaihingen_v08", help="Path to the VPRef package directory")
    parser.add_argument("--manifest", default="train.jsonl", help="JSONL manifest file name")
    parser.add_argument("--domain", default="Vaihingen", help="Domain to filter by")
    parser.add_argument("--resolution", type=int, default=480, help="Resize images to this resolution")
    parser.add_argument("--target_mode", default="union", help="Target mode (only 'union' supported)")
    parser.add_argument("--rotation_degrees", type=int, nargs="+", default=[0], help="Allowed rotation degrees")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train", help="Dataset split")
    parser.add_argument("--task_families", default=[], nargs="+", help="Filter by task families")
    args = parser.parse_args()

    dataset = VPRefV03Dataset(
        data_dir=args.data_dir,
        manifest="train.jsonl",
        domain=args.domain,
        resolution=args.resolution,
        target_mode=args.target_mode,
        rotation_degrees=args.rotation_degrees,
        allow_unreviewed_proposals=True,
        split=args.split,
        task_families=args.task_families,
    )

    val_ds = VPRefV03Dataset(
        data_dir=args.data_dir,
        manifest="val.jsonl",
        domain=args.domain,
        resolution=args.resolution,
        target_mode=args.target_mode,
        rotation_degrees=args.rotation_degrees,
        allow_unreviewed_proposals=True,
        split="val",
        task_families=args.task_families,
    )
    print(f"Loaded {len(dataset)} samples from {args.split} split.")

    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=None,  # Only shuffle if not using sampler
        sampler=None,
        collate_fn=collate_vpref,
        num_workers=0,
        pin_memory=True,
    )

    # for key in ("sample_id", "image_id", "source_scene"):
    #     if any(not row.get(key) for ds in (dataset, val_ds) for row in ds.samples):
    #         raise ValueError(f"Cannot check train/val isolation: missing {key}")
    #     left = {(r["domain"], r[key]) for r in dataset.samples}
    #     right = {(r["domain"], r[key]) for r in val_ds.samples}
    #     print("left", left)
    #     print("right", right)
    #     if left & right:
    #         raise ValueError(f"Train/val {key} overlap: {sorted(left & right)}")
        


    for batch_dict in train_loader:

        # 这里面的属性都来源于 collate_fn_api
        input_batch = batch_dict["input"]
        # print(input_batch) 
        # inside img_batch; 
        # find_text_batch;
        # find_inputs;
        # find_targets;
        # find_metadatas;
        print(input_batch.img_batch.shape)  # torch.Size([2, 3, 480, 480])
        break

    # img_batch=[]
    # find_text_batch=[]
    # find_inputs=[]
    # find_targets=[]
    # find_metadatas=[]
    # boxes
    # segments
    # semantic_segments=tensor([], dtype=torch.bool), 
    # is_valid_segment=tensor([True, True]), 
    # is_exhaustive=tensor([True, True]), 
    # object_ids=tensor([0, 0]), 
    # object_ids_padded=tensor([[0],[0]]))], 
    # find_metadatas=[BatchedInferenceMetadata(coco_image_id=tensor([0, 0]), 
    #                                          original_image_id=tensor([0, 0]), 
    #                                          original_category_id=tensor([0, 0], dtype=torch.int32), 
    #                                          original_size=tensor([[480, 480], [480, 480]]), 
    #                                          object_id=tensor([0, 1]), 
    #                                          frame_index=tensor([0, 0]), is_conditioning_only=[False, False])], 
    # raw_images=[]


    # for i in range(1):

    #     for row in dataset.samples:
    #         print(row.keys())

    #     point = dataset[i]
    #     print(f"Sample {i}:")
    #     print(f"  Image size: {point.images[0].data.shape}")
    #     print(f"  Segment size: {point.images[0].objects[0].segment.shape}")
    #     print(f"  Query text: {point.find_queries[0].query_text}")

