#!/usr/bin/env python
"""
Extract per-layer scalar representations from LGATr-slim for linear probing.

Registers forward hooks on model.net.linear_in (layer 0) and each of the 12
LGATrSlimBlocks (layers 1-12), capturing the scalar output h_s after each layer.

Token layout inside model.net after LGATrSlimWrapper inserts the global token:
  index 0          : global token             → jet-level representation
  index 1          : lightlike spurion +z
  index 2          : lightlike spurion -z
  index 3          : time spurion
  index 4..4+n-1   : physical particles       → particle-level representations

Saves results/probe_data_slim.npz with identical key structure to probe_data_lgatr.npz.

Usage:
  cd /home/jay_agarwal_2022/tagger-quantization
  CUDA_VISIBLE_DEVICES=1 nohup python3.10 run_probes_slim.py \\
      --checkpoint runs/lgatr_slim_toptag/slim_run1/models/model_run0.pt \\
      --config     runs/lgatr_slim_toptag/slim_run1/config.yaml \\
      --data-path  data/toptagging_full.npz \\
      --output     results/probe_data_slim.npz \\
      --n-jets     10000 \\
      > results/probe_slim.log 2>&1 &
"""

import argparse
import sys
import time
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

sys.path.insert(0, ".")

from experiments.tagging.dataset import TopTaggingDataset
from experiments.tagging.embedding import embed_tagging_data

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# After the wrapper inserts the global token at index 0:
#   [global(0), llike+z(1), llike-z(2), time-spur(3), phys_0(4), ...]
TOKEN_OFFSET = 4
N_LAYERS     = 13   # 0 = after linear_in,  1..12 = after each LGATrSlimBlock
K_MAX        = 50   # max particles per jet to store

# ──────────────────────────────────────────────────────────────────────────────
# Physics label helpers  (identical to run_probes_lgatr.py)
# ──────────────────────────────────────────────────────────────────────────────

def _pt(p4):
    return np.sqrt(p4[:, 1]**2 + p4[:, 2]**2)

def _eta(p4):
    pt = _pt(p4)
    return np.arcsinh(p4[:, 3] / (pt + 1e-10))

def _phi(p4):
    return np.arctan2(p4[:, 2], p4[:, 1])

def jet_mass(p4):
    j  = p4.sum(axis=0)
    m2 = j[0]**2 - j[1]**2 - j[2]**2 - j[3]**2
    return float(np.sqrt(max(m2, 0.0)))

def dR_to_jet_axis(p4):
    j     = p4.sum(axis=0)
    pt_j  = np.sqrt(j[1]**2 + j[2]**2)
    eta_j = float(np.arcsinh(j[3] / (pt_j + 1e-10)))
    phi_j = float(np.arctan2(j[2], j[1]))
    deta  = _eta(p4) - eta_j
    dphi  = (_phi(p4) - phi_j + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(deta**2 + dphi**2).astype(np.float32)

def pt_rank_quartile(p4):
    n   = len(p4)
    pt  = _pt(p4)
    rnk = np.argsort(np.argsort(-pt))
    return (rnk * 4 // n).clip(0, 3).astype(np.int32)

def _exclusive_kt_merge(cp, ce, cphi, target_n):
    while len(cp) > target_n:
        n    = len(cp)
        pt_i = cp[:, None];  pt_j = cp[None, :]
        de   = ce[:, None] - ce[None, :]
        dp   = cphi[:, None] - cphi[None, :]
        dp   = (dp + np.pi) % (2 * np.pi) - np.pi
        d    = np.minimum(pt_i**2, pt_j**2) * (de**2 + dp**2)
        np.fill_diagonal(d, np.inf)

        flat = np.argmin(d);  i, j = flat // n, flat % n
        if i > j: i, j = j, i

        pt_new  = cp[i] + cp[j]
        wj      = cp[j] / (pt_new + 1e-10)
        eta_new = ce[i] + wj * (ce[j] - ce[i])
        dpij    = (cphi[j] - cphi[i] + np.pi) % (2 * np.pi) - np.pi
        phi_new = cphi[i] + wj * dpij

        mask        = np.ones(n, dtype=bool)
        mask[i]     = False;  mask[j] = False
        cp          = np.append(cp[mask],   pt_new)
        ce          = np.append(ce[mask],   eta_new)
        cphi        = np.append(cphi[mask], phi_new)
    return cp, ce, cphi

def _tau_from_axes(pt, eta, phi, axes, R=0.8):
    d0     = pt.sum() * R + 1e-10
    ax_eta = np.array([a[0] for a in axes])
    ax_phi = np.array([a[1] for a in axes])
    tau    = 0.0
    for i in range(len(pt)):
        de  = eta[i] - ax_eta
        dp  = (phi[i] - ax_phi + np.pi) % (2 * np.pi) - np.pi
        tau += pt[i] * np.sqrt(de**2 + dp**2).min()
    return tau / d0

def nsubjettiness(p4, R=0.8):
    pt  = _pt(p4).astype(float)
    eta = _eta(p4).astype(float)
    phi = _phi(p4).astype(float)

    axes_at = {}
    cp, ce, cphi = pt.copy(), eta.copy(), phi.copy()

    while len(cp) > 1:
        if len(cp) <= 3:
            axes_at[len(cp)] = list(zip(ce.copy(), cphi.copy()))
        cp, ce, cphi = _exclusive_kt_merge(cp, ce, cphi, len(cp) - 1)

    axes_at[1] = [(ce[0], cphi[0])]
    for n in (2, 3):
        if n not in axes_at:
            axes_at[n] = axes_at[n - 1]

    tau1 = _tau_from_axes(pt, eta, phi, axes_at[1], R)
    tau2 = _tau_from_axes(pt, eta, phi, axes_at[2], R)
    tau3 = _tau_from_axes(pt, eta, phi, axes_at[3], R)
    return float(tau1), float(tau2), float(tau3)

# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, config_path, device):
    from hydra.utils import instantiate
    cfg = OmegaConf.load(config_path)

    model = instantiate(cfg.model)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]
    # Strip DataParallel prefix if present
    if any(k.startswith("net.module.") for k in state):
        state = {k.replace("net.module.", "net.", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded model from {checkpoint_path}")
    return model, cfg

# ──────────────────────────────────────────────────────────────────────────────
# Representation extraction
# ──────────────────────────────────────────────────────────────────────────────

def _squeeze_leading(t):
    while t.dim() > 2 and t.shape[0] == 1:
        t = t.squeeze(0)
    return t


def extract(model, cfg, data_path, n_jets, device):
    # ── data ──────────────────────────────────────────────────────────────────
    ds = TopTaggingDataset()
    ds.load_data(data_path, mode="test")
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    # ── storage ───────────────────────────────────────────────────────────────
    jet_reps_buf   = [[] for _ in range(N_LAYERS)]
    part_reps_buf  = [[] for _ in range(N_LAYERS)]

    jet_labels     = []
    jet_mass_l     = []
    jet_mult_l     = []
    jet_tau1_l     = []
    jet_tau2_l     = []
    jet_tau3_l     = []

    part_jet_idx_l  = []
    part_pt_l       = []
    part_eta_l      = []
    part_phi_l      = []
    part_E_l        = []
    part_quartile_l = []
    part_dR_l       = []

    jet_count = 0
    t0 = time.time()

    for batch in loader:
        if jet_count >= n_jets:
            break

        ptr    = batch.ptr.cpu().numpy()
        n_this = min(len(ptr) - 1, n_jets - jet_count)

        for bi in range(n_this):
            start, end = int(ptr[bi]), int(ptr[bi + 1])
            n_phys_raw = end - start
            n_phys     = min(n_phys_raw, K_MAX)

            # raw 4-momenta for physics labels
            p4_full = batch.x[start:end].numpy()
            p4      = p4_full[:n_phys]

            # ── physics labels ─────────────────────────────────────────────
            label        = int(batch.label[bi].item())
            mass         = jet_mass(p4_full)
            mult         = n_phys_raw
            tau1, tau2, tau3 = nsubjettiness(p4)

            pt_p       = _pt(p4).astype(np.float32)
            eta_p      = _eta(p4).astype(np.float32)
            phi_p      = _phi(p4).astype(np.float32)
            E_p        = p4[:, 0].astype(np.float32)
            quartiles  = pt_rank_quartile(p4)
            dR_p       = dR_to_jet_axis(p4)

            jet_labels.append(label)
            jet_mass_l.append(mass)
            jet_mult_l.append(mult)
            jet_tau1_l.append(tau1)
            jet_tau2_l.append(tau2)
            jet_tau3_l.append(tau3)

            part_jet_idx_l.extend([jet_count] * n_phys)
            part_pt_l.extend(pt_p.tolist())
            part_eta_l.extend(eta_p.tolist())
            part_phi_l.extend(phi_p.tolist())
            part_E_l.extend(E_p.tolist())
            part_quartile_l.extend(quartiles.tolist())
            part_dR_l.extend(dR_p.tolist())

            # ── single-jet forward pass with hooks ────────────────────────
            # embed_tagging_data returns the dict that LGATrSlimWrapper.forward expects
            fourmomenta_t = torch.tensor(p4_full, dtype=torch.float32, device=device)
            scalars_t     = torch.zeros(n_phys_raw, 0, dtype=torch.float32, device=device)
            ptr_t         = torch.tensor([0, n_phys_raw], dtype=torch.long, device=device)
            emb = embed_tagging_data(fourmomenta_t, scalars_t, ptr_t, cfg.data)

            captured = [None] * N_LAYERS

            def _make_hook(layer_idx):
                def _hook(module, inp, out):
                    # out = (h_v, h_s) for both linear_in and LGATrSlimBlock
                    captured[layer_idx] = out[1].detach().cpu()
                return _hook

            hooks = []
            hooks.append(model.net.linear_in.register_forward_hook(_make_hook(0)))
            for li, blk in enumerate(model.net.blocks):
                hooks.append(blk.register_forward_hook(_make_hook(li + 1)))

            with torch.no_grad():
                _ = model(emb)

            for h in hooks:
                h.remove()

            # ── store representations ──────────────────────────────────────
            for li in range(N_LAYERS):
                h_s = _squeeze_leading(captured[li])   # (n_tokens, s_ch)

                # jet-level: global token at index 0
                jet_reps_buf[li].append(h_s[0].numpy().astype(np.float32))

                # particle-level: physical particles
                phys = h_s[TOKEN_OFFSET : TOKEN_OFFSET + n_phys].numpy()
                part_reps_buf[li].append(phys.astype(np.float32))

            jet_count += 1

        if jet_count % 500 == 0 or jet_count == n_jets:
            elapsed = time.time() - t0
            print(f"  [{elapsed:6.0f}s]  {jet_count}/{n_jets} jets processed")

    print(f"\nDone. Total jets: {jet_count}")

    # ── assemble ───────────────────────────────────────────────────────────────
    jet_reps  = np.stack(
        [np.stack(jet_reps_buf[li]) for li in range(N_LAYERS)]
    )
    part_reps = np.stack(
        [np.vstack(part_reps_buf[li]) for li in range(N_LAYERS)]
    )

    tau1 = np.array(jet_tau1_l, dtype=np.float32)
    tau2 = np.array(jet_tau2_l, dtype=np.float32)
    tau3 = np.array(jet_tau3_l, dtype=np.float32)

    return {
        "jet_reps"                 : jet_reps,
        "particle_reps"            : part_reps,
        "particle_jet_idx"         : np.array(part_jet_idx_l,  dtype=np.int32),
        "jet_labels"               : np.array(jet_labels,       dtype=np.int32),
        "jet_mass"                 : np.array(jet_mass_l,       dtype=np.float32),
        "jet_multiplicity"         : np.array(jet_mult_l,       dtype=np.int32),
        "jet_tau1"                 : tau1,
        "jet_tau2"                 : tau2,
        "jet_tau3"                 : tau3,
        "jet_tau21"                : tau2 / (tau1 + 1e-10),
        "particle_pt"              : np.array(part_pt_l,        dtype=np.float32),
        "particle_eta"             : np.array(part_eta_l,       dtype=np.float32),
        "particle_phi"             : np.array(part_phi_l,       dtype=np.float32),
        "particle_E"               : np.array(part_E_l,         dtype=np.float32),
        "particle_pt_rank_quartile": np.array(part_quartile_l,  dtype=np.int32),
        "particle_dR_to_axis"      : np.array(part_dR_l,        dtype=np.float32),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/lgatr_slim_toptag/slim_run1/models/model_run0.pt")
    ap.add_argument("--config",     default="runs/lgatr_slim_toptag/slim_run1/config.yaml")
    ap.add_argument("--data-path",  default="data/toptagging_full.npz")
    ap.add_argument("--output",     default="results/probe_data_slim.npz")
    ap.add_argument("--n-jets",     type=int, default=10000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg = load_model(args.checkpoint, args.config, device)

    print(f"\nExtracting representations ({args.n_jets} jets, {N_LAYERS} layers)...")
    data = extract(model, cfg, args.data_path, args.n_jets, device)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)

    print(f"\nSaved → {out}")
    print("Shapes:")
    for k, v in data.items():
        print(f"  {k:35s}: {str(v.shape):25s}  {v.dtype}")


if __name__ == "__main__":
    main()
