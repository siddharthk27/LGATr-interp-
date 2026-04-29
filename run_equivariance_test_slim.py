#!/usr/bin/env python3
"""
Per-layer equivariance test for LGATr-slim on TopTagging.

Same approach as run_equivariance_test.py (full LGATr):
  - Embed jet once (particles + spurions)
  - Apply Λ_4x4 directly to ALL 4-vectors in the embedding (so spurions transform too)
  - Run both original and transformed embedding through model with hooks
  - Check h_s invariance and h_v equivariance per layer

h_s : Lorentz-INVARIANT (scalar channels)   → h_s(Λx) == h_s(x)
h_v : Lorentz-EQUIVARIANT (4-vector channels) → h_v(Λx) == h_v(x) @ Λ_4x4

Usage:
  cd /home/jay_agarwal_2022/tagger-quantization
  CUDA_VISIBLE_DEVICES=0 /usr/bin/python3.10 -u run_equivariance_test_slim.py \\
      --n-jets 200 --n-transforms 5 > results/equivariance_slim.log 2>&1
"""

import sys
import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

sys.path.insert(0, ".")
sys.path.insert(0, "/home/jay_agarwal_2022/lorentz-gatr")  # for gatr module

from experiments.tagging.dataset import TopTaggingDataset
from experiments.tagging.embedding import embed_tagging_data

NUM_LAYERS = 13
EPS = 1e-8

GAMMAS = [1.5, 2.0, 5.0, 10.0, 50.0]


# ── Lorentz transform helpers ─────────────────────────────────────────────────

def _rho_from_u(u):
    from gatr.utils.clifford import sandwich, np_to_mv
    rho_16 = np.zeros((16, 16), dtype=np.float32)
    for i in range(16):
        b = np.zeros(16); b[i] = 1.0
        rho_16[i] = sandwich(u, np_to_mv(b)).value.astype(np.float32)
    return rho_16


def sample_lorentz_transform(rng=None):
    from gatr.utils.clifford import sample_pin_multivector, np_to_mv
    if rng is None:
        rng = np.random.default_rng()
    u = sample_pin_multivector(spin=True, rng=rng)
    rho_16 = _rho_from_u(u)
    Lambda_4x4 = rho_16[1:5, 1:5].copy()
    return u, rho_16, Lambda_4x4


def sample_pure_boost(gamma, rng=None):
    from gatr.utils.clifford import LAYOUT, np_to_mv
    if rng is None:
        rng = np.random.default_rng()
    eta = float(np.arccosh(float(gamma)))
    n = rng.standard_normal(3); n /= np.linalg.norm(n)
    ch, sh = np.cosh(eta / 2), np.sinh(eta / 2)
    blade_list = LAYOUT.bladeTupList
    u_val = np.zeros(16)
    u_val[0] = ch
    u_val[blade_list.index((1, 2))] = sh * n[0]
    u_val[blade_list.index((1, 3))] = sh * n[1]
    u_val[blade_list.index((1, 4))] = sh * n[2]
    u = np_to_mv(u_val)
    rho_16 = _rho_from_u(u)
    return u, rho_16, rho_16[1:5, 1:5].copy()


def rel_err(a, b, eps=EPS):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + eps))


# ── Model loading ─────────────────────────────────────────────────────────────

def load_slim(checkpoint_path, config_path, device):
    from hydra.utils import instantiate
    cfg = OmegaConf.load(config_path)
    cfg.model.net.compile = False
    cfg.model.attention_backend = "math"
    model = instantiate(cfg.model)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]
    if any(k.startswith("net.module.") for k in state):
        state = {k.replace("net.module.", "net.", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, cfg


# ── Embedding + forward ───────────────────────────────────────────────────────

def embed_slim_jet(p4_np, cfg, device):
    n = len(p4_np)
    x   = torch.tensor(p4_np, dtype=torch.float32, device=device)
    s   = torch.zeros(n, 0, dtype=torch.float32, device=device)
    ptr = torch.tensor([0, n], dtype=torch.long, device=device)
    return embed_tagging_data(x, s, ptr, cfg.data)


def clone_embedding(emb):
    """Clone tensor values because LGATrSlimWrapper mutates ptr/fourmomenta in-place."""
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in emb.items()}


def apply_lambda_to_embedding(emb, Lambda_t, device):
    """
    Apply Λ_4x4 to ALL fourmomenta (particles + spurions) in the slim embedding.
    emb["fourmomenta"] shape: (n_tokens, 4).
    Note: tagging_features (log_pt, deta, etc.) are lab-frame scalars; we leave them
    unchanged here, so h_s errors will reflect slim's non-invariant scalar inputs.
    """
    emb_boost = clone_embedding(emb)
    emb_boost["fourmomenta"] = emb_boost["fourmomenta"] @ Lambda_t  # (n_tokens, 4) @ (4,4)
    return emb_boost


def run_slim_emb(model, emb):
    """
    Run a pre-built embedding through slim with hooks.
    Returns hs_list, hv_list (each NUM_LAYERS entries), logit (float).
    """
    hs_list = [None] * NUM_LAYERS
    hv_list = [None] * NUM_LAYERS

    def _make_hook(li):
        def hook(module, inp, out):
            h_v, h_s = out
            hv_list[li] = h_v.squeeze(0).clone().detach().cpu().numpy()
            hs_list[li] = h_s.squeeze(0).clone().detach().cpu().numpy()
        return hook

    hooks = []
    hooks.append(model.net.linear_in.register_forward_hook(_make_hook(0)))
    for li, blk in enumerate(model.net.blocks):
        hooks.append(blk.register_forward_hook(_make_hook(li + 1)))

    with torch.no_grad():
        result = model(clone_embedding(emb))
    for h in hooks:
        h.remove()

    logit = float(result[0][0, 0].cpu())
    return hs_list, hv_list, logit


# ── Main equivariance test ────────────────────────────────────────────────────

def test_equivariance(model, cfg, ds, n_jets, n_transforms, device, rng):
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    err_s_all     = np.zeros((n_jets, n_transforms, NUM_LAYERS), dtype=np.float32)
    err_v_all     = np.zeros((n_jets, n_transforms, NUM_LAYERS), dtype=np.float32)
    err_naive_all = np.zeros((n_jets, n_transforms, NUM_LAYERS), dtype=np.float32)
    logit_errs    = np.zeros((n_jets, n_transforms), dtype=np.float32)

    transforms = [sample_lorentz_transform(rng) for _ in range(n_transforms)]
    lambda_tensors = [
        torch.tensor(lam, dtype=torch.float32, device=device)
        for (_, _, lam) in transforms
    ]

    jet_count = 0
    t0 = time.time()
    for batch in loader:
        if jet_count >= n_jets:
            break
        ptr = batch.ptr.cpu().numpy()
        p4  = batch.x[int(ptr[0]):int(ptr[1])].numpy()

        emb_orig = embed_slim_jet(p4, cfg, device)
        hs_orig, hv_orig, logit_orig = run_slim_emb(model, emb_orig)

        for ti, ((u, rho_16, Lambda_4x4), lam_t) in enumerate(zip(transforms, lambda_tensors)):
            emb_boost = apply_lambda_to_embedding(emb_orig, lam_t, device)
            hs_boost, hv_boost, logit_boost = run_slim_emb(model, emb_boost)

            logit_errs[jet_count, ti] = abs(logit_boost - logit_orig) / (abs(logit_orig) + EPS)

            for li in range(NUM_LAYERS):
                err_s_all[jet_count, ti, li]     = rel_err(hs_boost[li], hs_orig[li])
                # h_v equivariance: h_v(Λx) == h_v(x) @ Λ_4x4  (row-vector)
                hv_expected = (hv_orig[li] @ Lambda_4x4).astype(np.float32)
                err_v_all[jet_count, ti, li]     = rel_err(hv_boost[li], hv_expected)
                err_naive_all[jet_count, ti, li] = rel_err(hv_boost[li], hv_orig[li])

        jet_count += 1
        if jet_count % 50 == 0:
            print(f"  Slim: {jet_count}/{n_jets} jets  [{time.time()-t0:.0f}s]")

    return err_s_all, err_v_all, err_naive_all, logit_errs


def test_boost_sweep(model, cfg, ds, gammas, n_jets, n_transforms, device, rng):
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    jets = []
    for batch in loader:
        if len(jets) >= n_jets: break
        ptr = batch.ptr.cpu().numpy()
        jets.append(batch.x[int(ptr[0]):int(ptr[1])].numpy())

    n_g = len(gammas)
    sweep_mv    = np.zeros((n_g, n_jets, n_transforms), dtype=np.float32)
    sweep_s     = np.zeros((n_g, n_jets, n_transforms), dtype=np.float32)
    sweep_logit = np.zeros((n_g, n_jets, n_transforms), dtype=np.float32)

    t0 = time.time()
    for gi, gamma in enumerate(gammas):
        transforms  = [sample_pure_boost(gamma, rng) for _ in range(n_transforms)]
        lambda_tensors = [torch.tensor(lam, dtype=torch.float32, device=device)
                          for (_, _, lam) in transforms]

        for ji, p4 in enumerate(jets):
            emb_orig = embed_slim_jet(p4, cfg, device)
            hs_orig, hv_orig, logit_orig = run_slim_emb(model, emb_orig)

            for ti, ((u, rho_16, Lambda_4x4), lam_t) in enumerate(zip(transforms, lambda_tensors)):
                emb_boost = apply_lambda_to_embedding(emb_orig, lam_t, device)
                hs_boost, hv_boost, logit_boost = run_slim_emb(model, emb_boost)

                sweep_logit[gi, ji, ti] = abs(logit_boost - logit_orig) / (abs(logit_orig) + EPS)
                layer_v = [rel_err(hv_boost[li], (hv_orig[li] @ Lambda_4x4).astype(np.float32))
                           for li in range(NUM_LAYERS)]
                layer_s = [rel_err(hs_boost[li], hs_orig[li]) for li in range(NUM_LAYERS)]
                sweep_mv[gi, ji, ti] = np.mean(layer_v)
                sweep_s[gi, ji, ti]  = np.mean(layer_s)

        print(f"  γ={gamma:5.1f}: h_v={sweep_mv[gi].mean():.2e}  "
              f"h_s={sweep_s[gi].mean():.2e}  "
              f"logit={sweep_logit[gi].mean():.2e}  [{time.time()-t0:.0f}s]")

    return sweep_mv, sweep_s, sweep_logit


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_equivariance(err_s, err_v, label_v, title, save_path):
    n_jets, n_tr, n_layers = err_s.shape
    flat_s = err_s.reshape(-1, n_layers)
    flat_v = err_v.reshape(-1, n_layers)
    layers = np.arange(n_layers)
    layer_labels = ["L0"] + [f"L{i}" for i in range(1, n_layers)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(layers, flat_s.mean(0), "b-o", linewidth=2, markersize=5, label="h_s invariance")
    ax.fill_between(layers, np.maximum(flat_s.mean(0)-flat_s.std(0), 1e-9),
                    flat_s.mean(0)+flat_s.std(0), alpha=0.2, color="blue")
    ax.semilogy(layers, flat_v.mean(0), "r-s", linewidth=2, markersize=5, label=label_v)
    ax.fill_between(layers, np.maximum(flat_v.mean(0)-flat_v.std(0), 1e-9),
                    flat_v.mean(0)+flat_v.std(0), alpha=0.2, color="red")
    ax.axhline(1.2e-7, color="gray", linestyle=":", linewidth=1.0, label="float32 ε")
    ax.axhline(1e-3,   color="silver", linestyle="--", linewidth=0.8, label="1e-3")
    ax.set_xticks(layers); ax.set_xticklabels(layer_labels, fontsize=9)
    ax.set_xlabel("Layer", fontsize=12); ax.set_ylabel("Relative equivariance error (log scale)", fontsize=12)
    ax.set_title(title, fontsize=13); ax.legend(fontsize=10); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {save_path}")


def plot_boost_sweep(gammas, sweep_mv, sweep_s, sweep_logit, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(gammas, sweep_mv.reshape(len(gammas),-1).mean(1),    "r-o", lw=2, ms=7, label="h_v equivariance")
    ax.loglog(gammas, sweep_s.reshape(len(gammas),-1).mean(1),     "b-s", lw=2, ms=7, label="h_s invariance")
    ax.loglog(gammas, sweep_logit.reshape(len(gammas),-1).mean(1), "g-^", lw=2, ms=7, label="logit invariance")
    ax.axhline(1.2e-7, color="gray",   linestyle=":", lw=1.0, label="float32 ε")
    ax.axhline(1e-3,   color="silver", linestyle="--", lw=0.8, label="1e-3")
    ax.set_xlabel("Lorentz boost factor γ", fontsize=12)
    ax.set_ylabel("Mean relative equivariance error", fontsize=12)
    ax.set_title("LGATr-slim equivariance vs boost strength", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved boost sweep → {save_path}")


def write_summary(err_s, err_v, err_naive, logit_errs,
                  gammas, sweep_mv, sweep_s, sweep_logit,
                  n_jets, n_tr, save_path):
    layer_labels = ["L0(linear_in)"] + [f"L{i}" for i in range(1, NUM_LAYERS)]

    def _mean(arr):
        return arr.reshape(-1, NUM_LAYERS).mean(0)

    lines = [
        "Per-layer Lorentz Equivariance Test  —  LGATr-slim",
        f"N_jets={n_jets}  N_transforms={n_tr}",
        "=" * 75,
        "",
        "Relative errors: ||h(Λx) - T(Λ)·h(x)|| / ||h(x)||",
        "  h_s correct : scalar invariance (should be ~0)",
        "  h_v correct : 4-vector equivariance with correct Λ_4x4 (should be ~0)",
        "  h_v naive   : if we predicted no change (sanity baseline)",
        "",
        f"{'Layer':<18}  {'h_s correct':>13}  {'h_v correct':>13}  {'h_v naive':>11}",
    ]
    ms, mv, mn = _mean(err_s), _mean(err_v), _mean(err_naive)
    for li in range(NUM_LAYERS):
        lines.append(f"  {layer_labels[li]:<16}  {ms[li]:>13.2e}  {mv[li]:>13.2e}  {mn[li]:>11.2e}")

    lines += [
        "",
        f"Output-level logit invariance: |logit(Λx) - logit(x)| / |logit(x)|",
        f"  Mean over {n_jets} jets × {n_tr} transforms: {logit_errs.mean():.2e}",
        "",
        "Boost sweep — pure boosts, mean error across all layers:",
        f"  {'γ':>6}  {'h_v correct':>13}  {'h_s correct':>13}  {'logit':>8}",
    ]
    for gi, gamma in enumerate(gammas):
        lines.append(f"  {gamma:>6.1f}  {sweep_mv[gi].mean():>13.2e}  "
                     f"{sweep_s[gi].mean():>13.2e}  {sweep_logit[gi].mean():>8.2e}")

    text = "\n".join(lines)
    save_path.write_text(text)
    print(f"\nSaved summary → {save_path}")
    print(text)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="runs/lgatr_slim_toptag/slim_run1/models/model_run0.pt")
    ap.add_argument("--config",
                    default="runs/lgatr_slim_toptag/slim_run1/config.yaml")
    ap.add_argument("--data-path",    default="data/toptagging_full.npz")
    ap.add_argument("--output-dir",   default="results/equivariance_slim")
    ap.add_argument("--n-jets",       type=int, default=200)
    ap.add_argument("--n-transforms", type=int, default=5)
    ap.add_argument("--n-jets-sweep", type=int, default=50)
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    print(f"Device: {device}")
    print(f"n_jets={args.n_jets}  n_transforms={args.n_transforms}")

    ds = TopTaggingDataset()
    ds.load_data(args.data_path, mode="test")

    model, cfg = load_slim(args.checkpoint, args.config, device)

    t_total = time.time()

    print(f"\n{'='*60}\nPer-layer equivariance test ({args.n_jets} jets)...")
    err_s, err_v, err_naive, logit_errs = test_equivariance(
        model, cfg, ds, args.n_jets, args.n_transforms, device, rng
    )
    print(f"  Mean logit invariance error: {logit_errs.mean():.2e}")

    print(f"\n{'='*60}\nBoost sweep γ={GAMMAS} ({args.n_jets_sweep} jets each)...")
    sweep_mv, sweep_s, sweep_logit = test_boost_sweep(
        model, cfg, ds, GAMMAS, args.n_jets_sweep, args.n_transforms, device, rng
    )

    np.savez(output_dir / "equivariance_slim_errors.npz",
             err_s=err_s, err_v=err_v, err_naive=err_naive, logit_errs=logit_errs,
             gammas=np.array(GAMMAS), sweep_mv=sweep_mv, sweep_s=sweep_s, sweep_logit=sweep_logit)

    plot_equivariance(err_s, err_v, "h_v equivariance (correct Λ)",
                      f"LGATr-slim — Per-layer Lorentz Equivariance "
                      f"({args.n_jets} jets × {args.n_transforms} transforms)",
                      output_dir / "equivariance_slim.png")
    plot_equivariance(err_s, err_naive, "h_v naive baseline (no-change prediction)",
                      f"LGATr-slim — Naive Baseline ({args.n_jets} jets × {args.n_transforms} transforms)",
                      output_dir / "equivariance_slim_naive.png")
    plot_boost_sweep(GAMMAS, sweep_mv, sweep_s, sweep_logit,
                     output_dir / "equivariance_slim_boost_sweep.png")
    write_summary(err_s, err_v, err_naive, logit_errs,
                  GAMMAS, sweep_mv, sweep_s, sweep_logit,
                  args.n_jets, args.n_transforms,
                  output_dir / "equivariance_slim_summary.txt")

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s")
    print(f"All outputs in {output_dir}")


if __name__ == "__main__":
    main()
