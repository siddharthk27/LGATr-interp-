#!/usr/bin/env python
"""
Interpretation pipeline for L-GATr-slim top tagging model.
Extracts per-layer attention score distributions and heatmaps.

Usage
-----
cd tagger-quantization
python run_interpretation_slim.py \
    --checkpoint runs/toptagging/<run>/models/model_run0.pt \
    --config runs/toptagging/<run>/.hydra/config.yaml \
    --data-path data/toptagging_full.npz \
    --output-dir interpretation_results_slim
"""

import sys
import os

# Ensure tagger-quantization package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import argparse
from pathlib import Path
from torch_geometric.loader import DataLoader

from experiments.tagging.interpret_slim import LGATrSlimInterpreter
from experiments.tagging.dataset import TopTaggingDataset


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("L-GATr-slim Interpretation Pipeline")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("\n[1/3] Loading model...")
    interpreter = LGATrSlimInterpreter(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
    )

    # ------------------------------------------------------------------
    # 2. Load test data  (same npz format as standard L-GATr)
    # ------------------------------------------------------------------
    print("\n[2/3] Loading test data...")
    test_dataset = TopTaggingDataset()
    test_dataset.load_data(args.data_path, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    batch = next(iter(test_loader))

    # ------------------------------------------------------------------
    # 3. Extract attention and produce plots
    # ------------------------------------------------------------------
    print("\n[3/3] Extracting attention maps...")
    results = interpreter.extract_attention_for_batch(batch, num_jets=args.num_jets)

    attention_dir = output_dir / "attention"
    attention_dir.mkdir(exist_ok=True)

    for jet_idx in range(len(results["attention_maps"])):
        attention_maps = results["attention_maps"][jet_idx]   # list[np.ndarray per layer]
        fourmomenta   = results["fourmomenta"][jet_idx]        # (n_particles, 4)
        label         = "Top" if results["labels"][jet_idx] else "QCD"

        # --- Attention score distribution (all layers, all heads) ---
        interpreter.plot_attention_distribution(
            attention_maps,
            save_path=attention_dir / f"jet{jet_idx}_{label}_distribution.png",
        )

        # --- Heatmaps for every head in the last layer ---
        final_layer_idx = len(attention_maps) - 1
        final_attn = attention_maps[final_layer_idx]   # (num_heads, N, N)
        for head_idx in range(final_attn.shape[0]):
            interpreter.visualize_attention_heatmap(
                final_attn,
                layer_idx=final_layer_idx,
                head_idx=head_idx,
                save_path=attention_dir
                / f"jet{jet_idx}_{label}_heatmap_layer{final_layer_idx}_head{head_idx}.png",
                title=(
                    f"L-GATr-slim  Jet {jet_idx} ({label}), "
                    f"Layer {final_layer_idx}, Head {head_idx}"
                ),
            )

        # --- Heatmaps for every head in the first layer ---
        first_attn = attention_maps[0]
        for head_idx in range(first_attn.shape[0]):
            interpreter.visualize_attention_heatmap(
                first_attn,
                layer_idx=0,
                head_idx=head_idx,
                save_path=attention_dir
                / f"jet{jet_idx}_{label}_heatmap_layer0_head{head_idx}.png",
                title=(
                    f"L-GATr-slim  Jet {jet_idx} ({label}), "
                    f"Layer 0, Head {head_idx}"
                ),
            )

        # --- η-φ connection plot: final layer, head 0 ---
        interpreter.visualize_attention_on_eta_phi(
            fourmomenta,
            final_attn,
            head_idx=0,
            threshold=args.attn_threshold,
            save_path=attention_dir / f"jet{jet_idx}_{label}_etaphi.png",
        )

    print(f"\n✓ Done! Results saved to {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpret trained L-GATr-slim top tagging model"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file, must contain key 'model')",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to resolved Hydra config (e.g. runs/.../..hydra/config.yaml)",
    )
    parser.add_argument(
        "--data-path", type=str, default="data/toptagging_full.npz",
        help="Path to toptagging npz dataset",
    )
    parser.add_argument(
        "--output-dir", type=str, default="interpretation_results_slim",
        help="Directory to save output figures",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="DataLoader batch size (only the first batch is used for interpretation)",
    )
    parser.add_argument(
        "--num-jets", type=int, default=5,
        help="Number of jets to analyse",
    )
    parser.add_argument(
        "--attn-threshold", type=float, default=0.1,
        help="Minimum attention weight to draw a line in the η-φ plot",
    )

    args = parser.parse_args()
    main(args)
