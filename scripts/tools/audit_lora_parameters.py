"""Export exact LoRA module counts from an adapter-only checkpoint.

Run from the repository root using the Ref environment. --verify-model checks
the current native training architecture without loading the base weights or
allocating the full model. It does not run inference or validate weight values.
"""

import argparse
from collections import defaultdict
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def component(name):
    if name.startswith(("backbone.", "transformer.")):
        return ".".join(name.split(".")[:2])
    return name.split(".")[0]


def read_adapters(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    pairs = defaultdict(dict)
    for name, tensor in state.items():
        module, separator, suffix = name.rpartition(".lora.")
        if not separator or suffix not in ("lora_A", "lora_B"):
            raise ValueError(f"Expected an adapter-only checkpoint: {name}")
        pairs[module][suffix] = tensor
    if not pairs:
        raise ValueError("No LoRA adapters found")
    rows = []
    for name, pair in pairs.items():
        if set(pair) != {"lora_A", "lora_B"}:
            raise ValueError(f"Incomplete adapter: {name}")
        a, b = pair["lora_A"], pair["lora_B"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError(f"Invalid adapter shapes: {name}")
        rows.append(dict(component=component(name), module=name,
                         in_features=a.shape[0], out_features=b.shape[1],
                         rank=a.shape[1], parameters=a.numel() + b.numel()))
    return rows, {name: tuple(tensor.shape) for name, tensor in state.items()}


def verify_model(config, shapes, bpe_path):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.position_encoding import PositionEmbeddingSine
    from sam3.model.decoder import TransformerDecoder
    from lora_layers import LoRAConfig, apply_lora_to_model, count_parameters

    pos_init = PositionEmbeddingSine.__init__
    coords = TransformerDecoder._get_coords
    linspace = torch.linspace

    def no_position_cache(self, *args, **kwargs):
        kwargs["precompute_resolution"] = None
        pos_init(self, *args, **kwargs)

    def cpu_coords(height, width, device):
        return coords(height, width, device="cpu")

    def cpu_linspace(*args, **kwargs):
        kwargs["device"] = "cpu"
        return linspace(*args, **kwargs)

    # These temporary patches affect only positional/coordinate caches and
    # scalar drop-path schedule construction, never parameter names or shapes.
    with patch.object(PositionEmbeddingSine, "__init__", no_position_cache), \
         patch.object(TransformerDecoder, "_get_coords", staticmethod(cpu_coords)), \
         patch.object(torch, "linspace", cpu_linspace), torch.device("meta"):
        model = build_sam3_image_model(
            device="cpu", eval_mode=False, load_from_HF=False,
            bpe_path=str(bpe_path), compile=False,
        )
        base_count = count_parameters(model)["total_parameters"]
        with contextlib.redirect_stdout(io.StringIO()):
            apply_lora_to_model(model, LoRAConfig(**config["lora"]))

    live = {name: tuple(p.shape) for name, p in model.named_parameters()
            if p.requires_grad}
    if live != shapes:
        raise ValueError("Checkpoint names/shapes differ from current trainable parameters")
    if any(".lora." not in name for name in live):
        raise ValueError("Non-LoRA trainable parameters found")
    return dict(base_parameters=base_count, **count_parameters(model),
                checkpoint_matches_current_trainable_parameters=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-model", action="store_true")
    parser.add_argument("--bpe-path", type=Path,
                        default=ROOT.parent / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    rows, shapes = read_adapters(args.checkpoint)
    if any(row["rank"] != config["lora"]["rank"] for row in rows):
        raise ValueError("Checkpoint ranks do not match the configuration")
    summary = defaultdict(lambda: dict(adapted_linear_layers=0, parameters=0))
    for row in rows:
        summary[row["component"]]["adapted_linear_layers"] += 1
        summary[row["component"]]["parameters"] += row["parameters"]
    report = dict(checkpoint=str(args.checkpoint), config=str(args.config),
                  lora=config["lora"], components=dict(summary),
                  adapted_linear_layers=len(rows),
                  adapter_parameters=sum(row["parameters"] for row in rows))
    if args.verify_model:
        report["model_verification"] = verify_model(config, shapes, args.bpe_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "lora_modules.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "lora_parameter_counts.json").write_text(
        json.dumps(report, indent=2) + "\n")
    for name, stats in summary.items():
        print(f"{name}: {stats['adapted_linear_layers']} layers, {stats['parameters']:,} parameters")
    print(f"Total: {len(rows)} layers, {report['adapter_parameters']:,} parameters")
    if args.verify_model:
        print(json.dumps(report["model_verification"], indent=2))


if __name__ == "__main__":
    main()
