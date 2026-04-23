#!/usr/bin/env python3
"""
Grade decomposition ablation for LGATr-slim on TopTagging.

LGATr-slim's hidden representations have two types of channels at each layer:
  h_v  (1, n_tokens, 32, 4)  —  32 channels of grade-1 Lorentz 4-vectors
  h_s  (1, n_tokens, 96)     —  96 Lorentz-invariant scalar channels

Unlike the full LGATr which has all 5 grades (0–4) in h_mv, slim is
CONSTRAINED to grade-1 vectors + grade-0 scalars. This experiment asks:

  1. Which of the two channel types is load-bearing?
       zero_v: zero ALL h_v (only h_s drives predictions)
       zero_s: zero ALL h_s (only h_v drives predictions)

  2. Within h_v, which Minkowski component matters?
     The 4 components of each grade-1 vector (t, x, y, z) in Minkowski space:
       zero_t:          zero h_v[..., 0]     — energy/timelike
       zero_z:          zero h_v[..., 3]     — beam-axis/longitudinal
       zero_transverse: zero h_v[..., 1:3]   — transverse plane (px, py)
       zero_spatial:    zero h_v[..., 1:]    — all 3-momentum components

Note on frame-dependence: individual Minkowski components (t,x,y,z) are
frame-dependent. All jets are in the lab frame (or consistent boosted frame),
so the results are empirically meaningful even though not Lorentz-invariant.

Comparison with full LGATr:
  Full LGATr grade-2 (bivectors) ΔAUC = 0.363 — hugely important
  Slim has no grade-2 at all; this experiment quantifies what slim substitutes.

Outputs (--output-dir, default results/grade_decomp_slim/):
  grade_decomp_slim.npz       all AUC values
  grade_decomp_slim_full.png  bar chart: ΔAUC per ablation (full-network)
  grade_decomp_slim_layer.png heatmap: ΔAUC per (ablation × layer) for zero_v / zero_s
  grade_decomp_slim_summary.txt

Usage:
  cd /home/jay_agarwal_2022/tagger-quantization
  CUDA_VISIBLE_DEVICES=1 nohup python3.10 -u run_grade_decomp_slim.py \\
      --checkpoint runs/lgatr_slim_toptag/slim_run1/models/model_run0.pt \\
      --config     runs/lgatr_slim_toptag/slim_run1/config.yaml \\
      --data-path  data/toptagging_full.npz \\
      --mode       all \\
      --n-jets     10000 \\
      > results/grade_decomp_slim.log 2>&1 &
"""

import argparse
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")

from experiments.tagging.dataset import TopTaggingDataset
from experiments.tagging.embedding import embed_tagging_data

NUM_LAYERS = 13   # linear_in (L0) + 12 blocks (L1..L12)

# ── Ablation definitions ──────────────────────────────────────────────────────
# Each ablation is: (name, label, description, which_tensor, slice_or_None)
#   which_tensor: "v" → zero in h_v last dim, "s" → zero all of h_s,
#                 "v_all" → zero all of h_v
ABLATIONS = [
    # name              label                    description
    ("zero_v",         "Zero h_v (all vectors)", "h_v = 0, h_s intact"),
    ("zero_s",         "Zero h_s (all scalars)", "h_s = 0, h_v intact"),
    ("zero_t",         "Zero e_t (energy)",       "h_v[...,0] = 0"),
    ("zero_z",         "Zero e_z (beam axis)",    "h_v[...,3] = 0"),
    ("zero_transverse","Zero e_x,e_y (transverse)","h_v[...,1:3] = 0"),
    ("zero_spatial",   "Zero e_x,e_y,e_z (3-mom)","h_v[...,1:] = 0"),
]
ABL_NAMES  = [a[0] for a in ABLATIONS]
ABL_LABELS = [a[1] for a in ABLATIONS]
ABL_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

# Ablations to include in layer-resolved (the most informative ones)
LAYER_ABL_INDICES = [0, 1]   # zero_v, zero_s


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path, config_path, device):
    from hydra.utils import instantiate
    cfg = OmegaConf.load(config_path)
    cfg.model.net.compile = False   # hooks require un-compiled model
    model = instantiate(cfg.model)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]
    if any(k.startswith("net.module.") for k in state):
        state = {k.replace("net.module.", "net.", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded model from {checkpoint_path}")
    return model, cfg


# ── Hook factories ────────────────────────────────────────────────────────────

def make_hook(abl_name):
    """
    Returns a forward hook for LGATrSlimBlock and linear_in.
    Both output (h_v, h_s) where:
      h_v: (1, N_tokens, 32, 4) — grade-1 4-vector channels
      h_s: (1, N_tokens, 96)    — scalar channels
    """
    if abl_name == "zero_v":
        def hook(module, inp, out):
            h_v, h_s = out
            return (torch.zeros_like(h_v), h_s)
    elif abl_name == "zero_s":
        def hook(module, inp, out):
            h_v, h_s = out
            return (h_v, torch.zeros_like(h_s))
    elif abl_name == "zero_t":
        def hook(module, inp, out):
            h_v, h_s = out
            h_v = h_v.clone(); h_v[..., 0] = 0.0
            return (h_v, h_s)
    elif abl_name == "zero_z":
        def hook(module, inp, out):
            h_v, h_s = out
            h_v = h_v.clone(); h_v[..., 3] = 0.0
            return (h_v, h_s)
    elif abl_name == "zero_transverse":
        def hook(module, inp, out):
            h_v, h_s = out
            h_v = h_v.clone(); h_v[..., 1:3] = 0.0
            return (h_v, h_s)
    elif abl_name == "zero_spatial":
        def hook(module, inp, out):
            h_v, h_s = out
            h_v = h_v.clone(); h_v[..., 1:] = 0.0
            return (h_v, h_s)
    else:
        raise ValueError(f"Unknown ablation: {abl_name}")
    return hook


def register_all_layer_hooks(model, abl_name):
    hook = make_hook(abl_name)
    handles = [model.net.linear_in.register_forward_hook(hook)]
    for blk in model.net.blocks:
        handles.append(blk.register_forward_hook(hook))
    return handles


def register_single_layer_hook(model, abl_name, layer_idx):
    hook = make_hook(abl_name)
    if layer_idx == 0:
        return [model.net.linear_in.register_forward_hook(hook)]
    else:
        return [model.net.blocks[layer_idx - 1].register_forward_hook(hook)]


# ── Inference helper ──────────────────────────────────────────────────────────

def run_inference(model, loader, cfg, n_jets, device):
    """
    Run model over n_jets jets.  Returns (labels, scores) as np.arrays.
    Assumes any ablation hooks are already registered externally.
    """
    all_logits, all_labels = [], []
    n_done = 0

    with torch.no_grad():
        for batch in loader:
            if n_done >= n_jets:
                break
            batch_dev = batch.to(device)
            fourmomenta = batch_dev.x
            scalars = (
                batch_dev.scalars
                if (hasattr(batch_dev, "scalars") and batch_dev.scalars is not None)
                else torch.zeros(fourmomenta.shape[0], 0, device=device)
            )
            emb = embed_tagging_data(fourmomenta, scalars, batch_dev.ptr, cfg.data)
            result = model(emb)
            logits = result[0]   # LGATrSlimWrapper returns (logits, tracker, frames)

            B    = logits.shape[0]
            take = min(B, n_jets - n_done)
            all_logits.extend(logits[:take, 0].cpu().tolist())
            all_labels.extend(batch.label[:take].cpu().tolist())
            n_done += take

    return np.array(all_labels, dtype=np.float32), np.array(all_logits, dtype=np.float32)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_full_network(baseline_auc, abl_aucs, n_jets, save_path):
    delta = baseline_auc - abl_aucs

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: absolute AUC
    ax = axes[0]
    x = np.arange(len(ABLATIONS))
    ax.bar(x, abl_aucs, color=ABL_COLORS, edgecolor="k", alpha=0.85, width=0.6)
    ax.axhline(baseline_auc, color="black", linestyle="--", linewidth=1.5,
               label=f"Baseline {baseline_auc:.4f}")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, label="Random (0.5)")
    for i, val in enumerate(abl_aucs):
        ax.text(i, val + 0.003, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ABL_LABELS, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("AUC (ablated)")
    ax.set_ylim(max(0.4, abl_aucs.min() - 0.05), baseline_auc + 0.01)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("AUC after ablation (all layers)", fontsize=11)

    # Right: ΔAUC
    ax = axes[1]
    ax.bar(x, delta, color=ABL_COLORS, edgecolor="k", alpha=0.85, width=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    for i, val in enumerate(delta):
        ax.text(i, val + 0.001, f"{val:+.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ABL_LABELS, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("ΔAUC  (baseline − ablated)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("AUC drop when ablation applied (all layers)", fontsize=11)

    fig.suptitle(
        f"LGATr-slim Channel Ablation  —  TopTagging  ({n_jets} jets)\n"
        f"Baseline AUC = {baseline_auc:.4f}  |  "
        f"h_v = grade-1 4-vectors (32 ch)   h_s = scalars (96 ch)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved full-network plot → {save_path}")


def plot_layer_resolved(baseline_auc, layer_aucs, n_jets, save_path):
    """
    layer_aucs: (len(LAYER_ABL_INDICES), 13)
    """
    delta = baseline_auc - layer_aucs
    row_labels = [ABL_LABELS[i] for i in LAYER_ABL_INDICES]
    layer_labels = ["L0"] + [f"L{i}" for i in range(1, NUM_LAYERS)]

    fig, ax = plt.subplots(figsize=(14, max(3, 2 * len(LAYER_ABL_INDICES))))
    sns.heatmap(
        delta, ax=ax, cmap="Reds", vmin=0, vmax=max(delta.max(), 1e-4),
        annot=True, fmt=".4f", annot_kws={"size": 8},
        xticklabels=layer_labels, yticklabels=row_labels,
        linewidths=0.3,
    )
    ax.set_xlabel("Layer (ablation at this layer's output)", fontsize=11)
    ax.set_ylabel("Ablation", fontsize=11)
    ax.set_title(
        f"LGATr-slim Layer-Resolved Ablation  —  ΔAUC  ({n_jets} jets)\n"
        f"Baseline AUC = {baseline_auc:.4f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved layer-resolved plot → {save_path}")


def write_summary(baseline_auc, abl_aucs, layer_aucs, n_jets, save_path):
    lines = []
    lines.append(f"LGATr-slim Channel Ablation Summary  —  TopTagging ({n_jets} jets)")
    lines.append("=" * 70)
    lines.append(f"\nBaseline AUC (no ablation): {baseline_auc:.5f}")
    lines.append(f"  (Full LGATr baseline for reference: 0.98634)\n")

    lines.append("Full-network ablation (applied at ALL layers)")
    lines.append("-" * 60)
    lines.append(f"{'Ablation':<28}  {'AUC ablated':>12}  {'ΔAUC':>8}")
    for i, (name, label, desc) in enumerate(ABLATIONS):
        lines.append(f"  {label:<26}  {abl_aucs[i]:>12.5f}  {baseline_auc - abl_aucs[i]:>+8.5f}")

    lines.append("\n  For reference — Full LGATr grade ablations:")
    lines.append("    G2 bivector zeroed: AUC=0.62329  ΔAUC=+0.36305  (largest drop)")
    lines.append("    G1 vector zeroed:   AUC=0.89971  ΔAUC=+0.08663")

    if layer_aucs is not None:
        lines.append("\nLayer-resolved ablation (zero_v and zero_s, one layer at a time)")
        lines.append("-" * 60)
        header = f"{'Ablation':<28}" + "".join(f" {f'L{i}':>7}" for i in range(NUM_LAYERS))
        lines.append(header)
        for ri, abl_idx in enumerate(LAYER_ABL_INDICES):
            row = f"  {ABL_LABELS[abl_idx]:<26}"
            for l in range(NUM_LAYERS):
                row += f" {baseline_auc - layer_aucs[ri, l]:>+7.4f}"
            lines.append(row)

    text = "\n".join(lines)
    save_path.write_text(text)
    print(f"\nSaved summary → {save_path}")
    print(text)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="LGATr-slim channel ablation on TopTagging")
    ap.add_argument("--checkpoint", default="runs/lgatr_slim_toptag/slim_run1/models/model_run0.pt")
    ap.add_argument("--config",     default="runs/lgatr_slim_toptag/slim_run1/config.yaml")
    ap.add_argument("--data-path",  default="data/toptagging_full.npz")
    ap.add_argument("--output-dir", default="results/grade_decomp_slim")
    ap.add_argument("--mode", choices=["full", "layer", "all"], default="all")
    ap.add_argument("--n-jets",     type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--save-scores", action="store_true",
                    help="Save per-jet logits for each ablation to "
                         "grade_decomp_slim_scores.npz (needed for bootstrap significance testing)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}  |  n_jets={args.n_jets}  |  batch_size={args.batch_size}")

    model, cfg = load_model(args.checkpoint, args.config, device)

    ds = TopTaggingDataset()
    ds.load_data(args.data_path, mode="test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    t_total = time.time()

    # ── Baseline ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nRun 0  —  Baseline (no ablation)")
    t0 = time.time()
    labels, scores = run_inference(model, loader, cfg, args.n_jets, device)
    baseline_auc = float(roc_auc_score(labels, scores))
    print(f"  Baseline AUC = {baseline_auc:.5f}  [{time.time()-t0:.0f}s]")

    abl_aucs   = None
    layer_aucs = None
    scores_to_save = {"labels": labels, "baseline": scores}  # populated if --save-scores

    # ── Full-network ablation ─────────────────────────────────────────────────
    if args.mode in ("full", "all"):
        print(f"\n{'='*60}\nFull-network ablation  ({len(ABLATIONS)} ablations × all layers)")
        abl_aucs = np.zeros(len(ABLATIONS), dtype=np.float32)

        for i, (abl_name, abl_label, abl_desc) in enumerate(ABLATIONS):
            t0 = time.time()
            handles = register_all_layer_hooks(model, abl_name)
            labels_a, scores_a = run_inference(model, loader, cfg, args.n_jets, device)
            for h in handles:
                h.remove()
            auc_a = float(roc_auc_score(labels_a, scores_a))
            abl_aucs[i] = auc_a
            if args.save_scores:
                scores_to_save[abl_name] = scores_a
            print(
                f"  [{i+1}/{len(ABLATIONS)}]  {abl_label:<30}: "
                f"AUC={auc_a:.5f}  ΔAUC={baseline_auc - auc_a:+.5f}  "
                f"[{time.time()-t0:.0f}s]"
            )

        if args.save_scores:
            scores_path = output_dir / "grade_decomp_slim_scores.npz"
            np.savez(scores_path, **scores_to_save)
            print(f"Saved per-jet scores → {scores_path}")

        np.savez(
            output_dir / "grade_decomp_slim.npz",
            baseline_auc=np.float32(baseline_auc),
            abl_aucs=abl_aucs,
            abl_names=np.array(ABL_NAMES),
        )
        plot_full_network(baseline_auc, abl_aucs, args.n_jets,
                          output_dir / "grade_decomp_slim_full.png")

    # ── Layer-resolved (zero_v and zero_s only) ────────────────────────────────
    if args.mode in ("layer", "all"):
        n_abl = len(LAYER_ABL_INDICES)
        total_runs = n_abl * NUM_LAYERS
        print(f"\n{'='*60}\nLayer-resolved ablation  ({n_abl} ablations × 13 layers = {total_runs} runs)")
        layer_aucs = np.zeros((n_abl, NUM_LAYERS), dtype=np.float32)
        run_idx = 0

        for ri, abl_idx in enumerate(LAYER_ABL_INDICES):
            abl_name, abl_label, _ = ABLATIONS[abl_idx]
            for l in range(NUM_LAYERS):
                t0 = time.time()
                handles = register_single_layer_hook(model, abl_name, l)
                labels_al, scores_al = run_inference(model, loader, cfg, args.n_jets, device)
                for h in handles:
                    h.remove()
                auc_al = float(roc_auc_score(labels_al, scores_al))
                layer_aucs[ri, l] = auc_al
                run_idx += 1
                layer_label = f"L{l}" if l > 0 else "L0(linear_in)"
                print(
                    f"  [{run_idx:3d}/{total_runs}]  {abl_label:<30} @ {layer_label:<15}: "
                    f"AUC={auc_al:.5f}  ΔAUC={baseline_auc - auc_al:+.5f}  "
                    f"[{time.time()-t0:.0f}s]"
                )

        np.savez(
            output_dir / "grade_decomp_slim_layer.npz",
            baseline_auc=np.float32(baseline_auc),
            layer_aucs=layer_aucs,
            abl_names=np.array([ABL_NAMES[i] for i in LAYER_ABL_INDICES]),
        )
        plot_layer_resolved(baseline_auc, layer_aucs, args.n_jets,
                            output_dir / "grade_decomp_slim_layer.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    write_summary(baseline_auc, abl_aucs, layer_aucs, args.n_jets,
                  output_dir / "grade_decomp_slim_summary.txt")
    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s")
    print(f"All outputs in {output_dir}")


if __name__ == "__main__":
    main()
