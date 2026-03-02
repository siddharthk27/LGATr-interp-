"""
Interpretation module for L-GATr-slim top tagging model.

Architecture differences vs standard L-GATr:
- Represents data as raw Lorentz 4-vectors (..., v_channels, 4)
  instead of full GA multivectors (..., channels, 16).
- Attention Q/K are formed by concatenating flattened vectors
  (with Lorentz metric sign flip on Q) and scalars.
- Block path: LGATrSlimWrapper.net.blocks[i].attention  (SelfAttention from lgatr_slim)
"""

import sys
import torch
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf

plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif"})


class LGATrSlimInterpreter:
    """Load and interpret a trained L-GATr-slim model for top tagging."""

    def __init__(self, checkpoint_path: str, config_path: str, taggerq_root: str | None = None):
        """
        Parameters
        ----------
        checkpoint_path : str
            Path to model checkpoint (.pt file).  The file must contain the key "model".
        config_path : str
            Path to the Hydra config file used during training (e.g. toptagging.yaml resolved).
        taggerq_root : str, optional
            Absolute path to the tagger-quantization repo root.
            Required if not already on sys.path.
        """
        if taggerq_root is not None and taggerq_root not in sys.path:
            sys.path.insert(0, taggerq_root)

        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = OmegaConf.load(config_path)
        self._load_model()

        # Per-layer storage populated by hooks during extract_attention_for_batch
        self._qk_store: list[dict] = []

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Instantiate and load the LGATrSlimWrapper from checkpoint."""
        from hydra.utils import instantiate

        self.model = instantiate(self.config.model)

        state_dict = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )["model"]
        # Strip DataParallel 'module.' prefix if present
        if any(k.startswith("net.module.") for k in state_dict):
            state_dict = {
                k.replace("net.module.", "net.", 1): v for k, v in state_dict.items()
            }
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded L-GATr-slim model from {self.checkpoint_path}")

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_single_jet(self, fourmomenta_np: np.ndarray) -> dict:
        """
        Build the embedding dict for a single jet.

        Parameters
        ----------
        fourmomenta_np : np.ndarray of shape (n_particles, 4)
            Raw 4-momenta (E, px, py, pz) in GeV.

        Returns
        -------
        dict  embedding suitable for LGATrSlimWrapper.forward
        """
        from experiments.tagging.embedding import embed_tagging_data

        fourmomenta = torch.tensor(
            fourmomenta_np, dtype=torch.float32, device=self.device
        )
        scalars = torch.zeros(
            len(fourmomenta), 0, dtype=torch.float32, device=self.device
        )
        ptr = torch.tensor(
            [0, len(fourmomenta)], dtype=torch.long, device=self.device
        )
        return embed_tagging_data(fourmomenta, scalars, ptr, self.config.data)

    # ------------------------------------------------------------------
    # Attention extraction
    # ------------------------------------------------------------------

    def _make_hook(self, layer_idx: int):
        """
        Return a forward hook for LGATrSlimBlock.attention (SelfAttention).

        The hook captures Q and K by re-running linear_in + _pre_reshape
        on the (already-normed) vectors and scalars that enter SelfAttention.

        Input signature of SelfAttention.forward:
            forward(self, vectors, scalars, **attn_kwargs)
        Hook input tuple: (vectors, scalars)  [positional args only]
        """

        def hook(module, inp, output):
            # inp[0] = vectors after RMSNorm  shape: (1, n_items, v_channels, 4)
            # inp[1] = scalars after RMSNorm  shape: (1, n_items, s_channels)
            h_v, h_s = inp[0], inp[1]

            with torch.no_grad():
                qkv_v, qkv_s = module.linear_in(h_v, h_s)
                # _pre_reshape applies internal RMSNorm for numerical stability,
                # then splits into q/k/v and constructs the actual dot-product keys.
                q, k, _v = module._pre_reshape(qkv_v, qkv_s)
                # q, k shapes: (1, num_heads, n_items, hidden_v_channels*4 + hidden_s_channels)

            self._qk_store.append(
                {
                    "q": q.detach().cpu(),
                    "k": k.detach().cpu(),
                    "layer_idx": layer_idx,
                }
            )

        return hook

    def _compute_attn_weights(self, q: torch.Tensor, k: torch.Tensor) -> np.ndarray:
        """
        Compute attention weights from Q and K.

        Parameters
        ----------
        q, k : Tensor of shape (1, num_heads, n_items, d)

        Returns
        -------
        np.ndarray of shape (num_heads, n_items, n_items)
        """
        scale = 1.0 / (q.shape[-1] ** 0.5)
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # (1, H, N, N)
        weights = torch.softmax(logits, dim=-1)
        weights = weights.squeeze(0)  # (H, N, N)
        return weights.numpy()

    def extract_attention_for_batch(self, batch, num_jets: int = 10) -> dict:
        """
        Extract per-layer attention weights for individual jets.

        Processes each jet in isolation (single-jet forward pass) to get
        clean per-jet attention patterns without cross-jet contamination.

        Parameters
        ----------
        batch : torch_geometric batch from TopTaggingDataset
        num_jets : int
            Maximum number of jets to process.

        Returns
        -------
        dict with keys:
            "attention_maps": List[List[np.ndarray]]
                Per-jet, per-layer attention of shape (num_heads, n_particles, n_particles).
            "fourmomenta":    List[np.ndarray]  raw 4-momenta per jet (after rescaling is undone)
            "labels":         List[int]
            "num_particles":  List[int]
        """
        results = {
            "attention_maps": [],
            "fourmomenta": [],
            "labels": [],
            "num_particles": [],
        }

        ptr = batch.ptr.cpu().numpy()
        n_jets = min(num_jets, len(ptr) - 1)

        for jet_idx in range(n_jets):
            start, end = ptr[jet_idx], ptr[jet_idx + 1]
            fourmomenta_np = batch.x[start:end].numpy()  # (n_particles, 4)

            if jet_idx == 0:
                print(f"  Jet 0: {len(fourmomenta_np)} particles")

            # Build embedding (rescaling happens inside LGATrSlimWrapper.forward)
            embedding = self._embed_single_jet(fourmomenta_np)

            # Register hooks on every SelfAttention module
            self._qk_store = []
            hooks = []
            for layer_idx, block in enumerate(self.model.net.blocks):
                hooks.append(
                    block.attention.register_forward_hook(self._make_hook(layer_idx))
                )

            with torch.no_grad():
                _ = self.model(embedding)

            for h in hooks:
                h.remove()

            # Convert Q/K pairs to attention weight matrices
            jet_attention_maps = []
            for qk in self._qk_store:
                attn = self._compute_attn_weights(qk["q"], qk["k"])
                # attn now covers all tokens including global token / spurions.
                # Slice to physical particles only for cleaner visualizations.
                n_phys = len(fourmomenta_np)
                attn_phys = attn[:, :n_phys, :n_phys]

                if jet_idx == 0 and qk["layer_idx"] == 0:
                    print(
                        f"  Layer 0 attn: shape={attn_phys.shape}, "
                        f"range=[{attn_phys.min():.5f}, {attn_phys.max():.5f}]"
                    )
                jet_attention_maps.append(attn_phys)

            results["attention_maps"].append(jet_attention_maps)
            results["fourmomenta"].append(fourmomenta_np)
            results["labels"].append(int(batch.label[jet_idx].item()))
            results["num_particles"].append(len(fourmomenta_np))

        self._qk_store = []
        return results

    # ------------------------------------------------------------------
    # Visualizations  (same interface as LGATrInterpreter)
    # ------------------------------------------------------------------

    def plot_attention_distribution(
        self, attention_maps: list, save_path=None
    ):
        """
        Histogram of off-diagonal attention scores (log-y scale).

        Parameters
        ----------
        attention_maps : list of np.ndarray, one per layer, shape (H, N, N)
        save_path : Path or str, optional
        """
        plt.figure(figsize=(8, 6))

        all_values = []
        for layer_attn in attention_maps:
            for head_attn in layer_attn:          # iterate over heads
                mask = ~np.eye(head_attn.shape[0], dtype=bool)
                all_values.extend(head_attn[mask].flatten().tolist())

        plt.hist(all_values, bins=100, density=True, alpha=0.7, edgecolor="black")
        plt.yscale("log")
        plt.xlabel("Attention Score", fontsize=14)
        plt.ylabel("Density (log scale)", fontsize=14)
        plt.title("L-GATr-slim Attention Score Distribution", fontsize=16)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved attention distribution to {save_path}")
        plt.close()

    def visualize_attention_heatmap(
        self,
        attention_map: np.ndarray,
        layer_idx: int,
        head_idx: int,
        save_path=None,
        title: str | None = None,
    ):
        """
        Plot a single-head attention heatmap.

        Parameters
        ----------
        attention_map : np.ndarray of shape (num_heads, n_particles, n_particles)
        layer_idx : int
        head_idx : int
        save_path : Path or str, optional
        title : str, optional
        """
        plt.figure(figsize=(10, 8))
        attn = attention_map[head_idx]  # (N, N)

        sns.heatmap(
            attn,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Attention Weight"},
        )

        if title is None:
            title = f"L-GATr-slim  Layer {layer_idx}, Head {head_idx}"
        plt.title(title, fontsize=14)
        plt.xlabel("Key Position", fontsize=12)
        plt.ylabel("Query Position", fontsize=12)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved attention heatmap to {save_path}")
        plt.close()

    def visualize_attention_on_eta_phi(
        self,
        fourmomenta: np.ndarray,
        attention_map: np.ndarray,
        head_idx: int = 0,
        threshold: float = 0.1,
        save_path=None,
    ):
        """
        Draw attention connections on the η-φ plane.

        Parameters
        ----------
        fourmomenta : np.ndarray (n_particles, 4)  (E, px, py, pz) in GeV
        attention_map : np.ndarray (num_heads, n_particles, n_particles)
        head_idx : int
        threshold : float  only draw connections with weight > threshold
        save_path : Path or str, optional
        """
        E, px, py, pz = (
            fourmomenta[:, 0],
            fourmomenta[:, 1],
            fourmomenta[:, 2],
            fourmomenta[:, 3],
        )
        pt = np.sqrt(px**2 + py**2)
        eta = np.arctanh(pz / np.sqrt(px**2 + py**2 + pz**2 + 1e-8))
        phi = np.arctan2(py, px)

        attn = attention_map[head_idx]

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            phi, eta, c=pt, s=100 * pt / (pt.max() + 1e-8),
            cmap="viridis", alpha=0.6, edgecolors="black", linewidth=1,
        )
        plt.colorbar(scatter, label="$p_T$ (GeV)")

        for i in range(len(fourmomenta)):
            for j in range(len(fourmomenta)):
                if i != j and attn[i, j] > threshold:
                    alpha = float(min(attn[i, j], 1.0))
                    plt.plot(
                        [phi[i], phi[j]], [eta[i], eta[j]],
                        "r-", alpha=alpha, linewidth=2 * attn[i, j],
                    )

        plt.xlabel("φ", fontsize=14)
        plt.ylabel("η", fontsize=14)
        plt.title(
            f"L-GATr-slim Attention  Head {head_idx}  threshold={threshold}",
            fontsize=16,
        )
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved η-φ attention plot to {save_path}")
        plt.close()
