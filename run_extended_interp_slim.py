#!/usr/bin/env python3
"""
Extended interpretation for LGATr-slim on TopTagging test set.

Computes three analyses from per-jet forward passes:
  1. CKA similarity matrix (13×13, layers 0..12) — linear CKA between global-token
     scalar representations at every layer.
  2. Spurion attention — mean attention weight from physical particles to each
     special token [Global, Llike+z, Llike-z, Time-spur], compared Top vs QCD.
  3. pT-rank attention — mean attention weight received by each pT rank (0=highest),
     compared Top vs QCD.

Token layout inside model.net (LGATrSlimWrapper inserts global token at 0):
  [global(0), llike+z(1), llike-z(2), time-spur(3), phys_0(4), phys_1(5), ...]
  → TOKEN_OFFSET=4, N_SPECIAL=4

Output (--output-dir, default results/extended_interp/):
  cka_slim.npy / cka_slim.png
  spurion_attn_slim.npz / spurion_attn_slim.png
  ptrank_attn_slim.npz  / ptrank_attn_slim.png

Usage:
  cd /home/jay_agarwal_2022/tagger-quantization
  CUDA_VISIBLE_DEVICES=1 nohup python3.10 -u run_extended_interp_slim.py \\
    --checkpoint runs/lgatr_slim_toptag/slim_run1/models/model_run0.pt \\
    --config runs/lgatr_slim_toptag/slim_run1/config.yaml \\
    --data-path data/toptagging_full.npz \\
    > results/extended_interp_slim.log 2>&1 &
"""

import argparse
import sys
import os
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif"})

NUM_LAYERS    = 13   # linear_in (L0) + 12 blocks (L1..L12)
NUM_HEADS     = 8
K             = 50   # max physical particles per jet
TOKEN_OFFSET  = 4    # global(0) + llike+z(1) + llike-z(2) + time(3)
N_SPECIAL     = 4
SPECIAL_NAMES = ["Global", "Llike+z", "Llike-z", "Time-spur"]


# ── CKA ──────────────────────────────────────────────────────────────────────

def _centered_gram(X):
    K_mat = X @ X.T
    rm = K_mat.mean(axis=1, keepdims=True)
    cm = K_mat.mean(axis=0, keepdims=True)
    gm = K_mat.mean()
    return K_mat - rm - cm + gm


def compute_cka_matrix(hs_accum):
    print(f"  Pre-computing centred Gram matrices ({NUM_LAYERS} layers)...")
    X  = [np.stack(hs_accum[i]).astype(np.float64) for i in range(NUM_LAYERS)]
    Kc = [_centered_gram(xi) for xi in X]
    diag = [np.sum(k * k) for k in Kc]
    cka = np.zeros((NUM_LAYERS, NUM_LAYERS))
    for i in range(NUM_LAYERS):
        for j in range(i, NUM_LAYERS):
            val = np.sum(Kc[i] * Kc[j]) / (np.sqrt(diag[i] * diag[j]) + 1e-10)
            cka[i, j] = cka[j, i] = val
    return cka


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path, config_path, device):
    from hydra.utils import instantiate
    cfg = OmegaConf.load(config_path)

    cfg.model.net.compile = False  # hooks don't work on compiled models
    model = instantiate(cfg.model)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]
    if any(k.startswith("net.module.") for k in state):
        state = {k.replace("net.module.", "net.", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded model from {checkpoint_path}")
    return model, cfg


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_cka(cka_matrix, title, save_path):
    layer_labels = ["L0"] + [f"L{i}" for i in range(1, NUM_LAYERS)]
    fig, ax = plt.subplots(figsize=(11, 10))
    sns.heatmap(
        cka_matrix, ax=ax, cmap="viridis", vmin=0, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 6},
        xticklabels=layer_labels, yticklabels=layer_labels,
    )
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved CKA plot → {save_path}")


def plot_spurion(top_arr, qcd_arr, names, title, save_path):
    n_s = len(names)
    fig, axes = plt.subplots(1, n_s, figsize=(4.5 * n_s, 5), sharey=False)
    colors = {"Top": "tab:blue", "QCD": "tab:orange"}
    for si, (name, ax) in enumerate(zip(names, axes)):
        for cls, arr in [("Top", top_arr), ("QCD", qcd_arr)]:
            if len(arr) == 0:
                continue
            ax.hist(arr[:, si], bins=60, alpha=0.6, density=True,
                    label=cls, color=colors[cls])
        ax.set_title(f"→ {name}", fontsize=12)
        ax.set_xlabel("Mean attention to token", fontsize=11)
        ax.set_ylabel("Density" if si == 0 else "", fontsize=11)
        ax.legend(fontsize=10)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved spurion plot → {save_path}")


def plot_ptrank(top_list, qcd_list, title, save_path):
    def _mean_per_rank(arr_list):
        if not arr_list:
            return np.array([]), np.array([])
        max_k = max(len(a) for a in arr_list)
        s = np.zeros(max_k)
        c = np.zeros(max_k, dtype=int)
        for a in arr_list:
            s[:len(a)] += a
            c[:len(a)] += 1
        return np.arange(max_k), np.where(c > 0, s / c, 0.0)

    ranks_t, mean_t = _mean_per_rank(top_list)
    ranks_q, mean_q = _mean_per_rank(qcd_list)

    fig, ax = plt.subplots(figsize=(10, 5))
    if len(ranks_t):
        ax.plot(ranks_t, mean_t, label="Top", color="tab:blue",   linewidth=2)
    if len(ranks_q):
        ax.plot(ranks_q, mean_q, label="QCD", color="tab:orange", linewidth=2)
    ax.set_xlabel("pT rank (0 = highest pT)", fontsize=13)
    ax.set_ylabel("Mean attention weight received", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved pT-rank plot → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"n_jets={args.n_jets}  n_cka={args.n_cka}  K={K}  TOKEN_OFFSET={TOKEN_OFFSET}")

    model, cfg = load_model(args.checkpoint, args.config, device)

    from experiments.tagging.dataset import TopTaggingDataset
    from experiments.tagging.embedding import embed_tagging_data

    ds = TopTaggingDataset()
    ds.load_data(args.data_path, mode="test")
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    # ── Accumulators ─────────────────────────────────────────────────────────
    hs_cka  = [[] for _ in range(NUM_LAYERS)]
    n_cka   = 0

    spurion_top, spurion_qcd = [], []
    ptrank_top,  ptrank_qcd  = [], []

    n_processed = 0
    t0 = time.time()

    with torch.no_grad():
        for batch in loader:
            if n_processed >= args.n_jets:
                break

            ptr    = batch.ptr.cpu().numpy()
            labels = batch.label.cpu().numpy()
            n_this = min(len(ptr) - 1, args.n_jets - n_processed)

            for bi in range(n_this):
                start, end = int(ptr[bi]), int(ptr[bi + 1])
                n_phys     = end - start

                # ── Single-jet embedding → wrapper forward ────────────────────
                fourmomenta_t = batch.x[start:end].to(device)
                scalars_t     = torch.zeros(n_phys, 0, device=device)
                ptr_t         = torch.tensor([0, n_phys], dtype=torch.long, device=device)
                emb = embed_tagging_data(fourmomenta_t, scalars_t, ptr_t, cfg.data)

                # ── Hooks: block outputs (h_s) + final-block attention (Q, K) ─
                captured_hs = {}    # {layer_idx: (N_total, s_ch) cpu tensor}
                captured_qk = {}    # {"q", "k"} from final block's attention module

                def _make_hs_hook(li):
                    def _hook(module, inp, out):
                        h_s = out[1].detach().cpu()
                        while h_s.dim() > 2 and h_s.shape[0] == 1:
                            h_s = h_s.squeeze(0)
                        captured_hs[li] = h_s  # (N_total, s_ch)
                    return _hook

                def _attn_hook(module, inp, output):
                    # SelfAttention.forward(vectors, scalars, ...)
                    # inp[0]=vectors (1,N_total,v_ch,4), inp[1]=scalars (1,N_total,s_ch)
                    h_v, h_s = inp[0], inp[1]
                    with torch.no_grad():
                        qkv_v, qkv_s = module.linear_in(h_v, h_s)
                        q, k, _ = module._pre_reshape(qkv_v, qkv_s)
                    # q, k: (1, H, N_total, d_k)
                    captured_qk["q"] = q.detach().cpu()
                    captured_qk["k"] = k.detach().cpu()

                hooks = []
                hooks.append(model.net.linear_in.register_forward_hook(_make_hs_hook(0)))
                for li, blk in enumerate(model.net.blocks):
                    hooks.append(blk.register_forward_hook(_make_hs_hook(li + 1)))
                hooks.append(
                    model.net.blocks[-1].attention.register_forward_hook(_attn_hook)
                )

                _ = model(emb)

                for h in hooks:
                    h.remove()

                if not captured_qk:
                    continue

                # N_total = TOKEN_OFFSET + n_phys
                n_total = TOKEN_OFFSET + n_phys

                # ── CKA: global-token h_s at each layer ───────────────────────
                if n_cka < args.n_cka:
                    for li in range(NUM_LAYERS):
                        h_s = captured_hs.get(li)
                        if h_s is None or h_s.shape[0] < 1:
                            continue
                        # Global token is at index 0 (inserted by wrapper)
                        hs_cka[li].append(h_s[0].numpy().astype(np.float32))
                    n_cka += 1

                # ── Attention: phys queries, all-token keys ───────────────────
                q = captured_qk["q"].squeeze(0)   # (H, N_total, d_k)
                k = captured_qk["k"].squeeze(0)   # (H, N_total, d_k)

                k_eff = min(K, n_phys)
                if k_eff < 5:
                    n_processed += 1
                    continue

                # Physical query rows: TOKEN_OFFSET .. TOKEN_OFFSET+k_eff
                q_phys = q[:, TOKEN_OFFSET:TOKEN_OFFSET + k_eff, :]   # (H, k_eff, d_k)

                # All keys for this jet
                scale = q_phys.shape[-1] ** -0.5
                attn = torch.softmax(
                    q_phys @ k.transpose(-2, -1) * scale, dim=-1
                ).numpy()  # (H, k_eff, N_total)

                # Column layout:
                # 0..N_SPECIAL-1     → special tokens [global, llike+z, llike-z, time]
                # N_SPECIAL..        → physical particles (pT-ordered)

                # Spurion: mean attention from phys to each of the N_SPECIAL special tokens
                spur_vals = attn[:, :, :N_SPECIAL].mean(axis=(0, 1))   # (N_SPECIAL,)

                # pT-rank: mean attention from phys to phys (cols TOKEN_OFFSET..TOKEN_OFFSET+k_eff)
                ptrank_vals = attn[:, :, TOKEN_OFFSET:TOKEN_OFFSET + k_eff].mean(axis=(0, 1))
                # (k_eff,) — attention received per pT rank

                label_str = "Top" if labels[bi] else "QCD"
                if label_str == "Top":
                    spurion_top.append(spur_vals)
                    ptrank_top.append(ptrank_vals)
                else:
                    spurion_qcd.append(spur_vals)
                    ptrank_qcd.append(ptrank_vals)

                n_processed += 1

            if n_processed % 500 == 0 or n_processed == args.n_jets:
                elapsed = time.time() - t0
                print(f"  [{elapsed:6.0f}s]  {n_processed}/{args.n_jets} jets  "
                      f"cka={n_cka}  top={len(spurion_top)}  qcd={len(spurion_qcd)}")

    elapsed = time.time() - t0
    print(f"\nDone. jets={n_processed}  cka={n_cka}  elapsed={elapsed:.0f}s")

    # ── Save and plot CKA ─────────────────────────────────────────────────────
    if n_cka > 0:
        print(f"\nComputing {NUM_LAYERS}×{NUM_LAYERS} CKA matrix from {n_cka} jets...")
        t_cka = time.time()
        cka_matrix = compute_cka_matrix(hs_cka)
        print(f"CKA done in {time.time() - t_cka:.1f}s")
        np.save(output_dir / "cka_slim.npy", cka_matrix)
        plot_cka(
            cka_matrix,
            f"LGATr-slim  Linear CKA  (TopTagging, global-token h_s, {n_cka} jets)",
            output_dir / "cka_slim.png",
        )

    # ── Save and plot spurion attention ───────────────────────────────────────
    if spurion_top or spurion_qcd:
        top_arr = np.array(spurion_top) if spurion_top else np.zeros((0, N_SPECIAL))
        qcd_arr = np.array(spurion_qcd) if spurion_qcd else np.zeros((0, N_SPECIAL))
        np.savez(
            output_dir / "spurion_attn_slim.npz",
            top=top_arr, qcd=qcd_arr, names=np.array(SPECIAL_NAMES),
        )
        plot_spurion(
            top_arr, qcd_arr, SPECIAL_NAMES,
            f"LGATr-slim  Spurion Attention  (TopTagging, {len(spurion_top)+len(spurion_qcd)} jets)",
            output_dir / "spurion_attn_slim.png",
        )

    # ── Save and plot pT-rank attention ───────────────────────────────────────
    if ptrank_top or ptrank_qcd:
        plot_ptrank(
            ptrank_top, ptrank_qcd,
            f"LGATr-slim  pT-rank Attention  (TopTagging, {len(ptrank_top)+len(ptrank_qcd)} jets)",
            output_dir / "ptrank_attn_slim.png",
        )
        np.savez(
            output_dir / "ptrank_attn_slim.npz",
            top=np.array(ptrank_top, dtype=object),
            qcd=np.array(ptrank_qcd, dtype=object),
        )

    print(f"\nAll results in {output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="LGATr-slim extended interpretation: CKA, spurion, pT-rank"
    )
    ap.add_argument("--checkpoint", default="runs/lgatr_slim_toptag/slim_run1/models/model_run0.pt")
    ap.add_argument("--config",     default="runs/lgatr_slim_toptag/slim_run1/config.yaml")
    ap.add_argument("--data-path",  default="data/toptagging_full.npz")
    ap.add_argument("--output-dir", default="results/extended_interp")
    ap.add_argument("--n-jets",     type=int, default=10000,
                    help="Total jets for spurion/pT-rank analysis")
    ap.add_argument("--n-cka",      type=int, default=5000,
                    help="Jets to use for CKA (first n_cka jets encountered)")
    args = ap.parse_args()
    main(args)
