import hydra
import torch
import math
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Server-friendly backend
import matplotlib.pyplot as plt
import seaborn as sns
from einops import rearrange
from omegaconf import OmegaConf

# Import Experiments
from experiments.tagging.experiment import TopTaggingExperiment
from experiments.tagging.dataset import TopTaggingDataset
from torch_geometric.data import DataLoader

# Import GATr internals
import gatr.layers.attention.attention
from gatr.primitives.invariants import _load_inner_product_factors

plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif" # Also fix the 'Charter' font warning
    })
    

# ==============================================================================
# 1. MONKEY PATCH (Capture Attention Weights)
# ==============================================================================
ATTENTION_LOG = []

def manual_attention_with_capture(query, key, value, attn_mask=None, dropout_p=0.0):
    scale = 1 / math.sqrt(query.size(-1))
    attn_weight = query @ key.transpose(-2, -1) * scale
    
    if attn_mask is not None:
        if hasattr(attn_mask, 'materialize'):
            attn_bias = attn_mask.materialize(attn_weight.shape)
        else:
            attn_bias = attn_mask
        attn_weight += attn_bias
    
    attn_weight = torch.softmax(attn_weight, dim=-1)
    
    # Capture weights (Detach to save memory)
    ATTENTION_LOG.append(attn_weight.detach().cpu())
    
    output = attn_weight @ value
    return output

def sdp_attention_patched(q_mv, k_mv, v_mv, q_s, k_s, v_s, attn_mask=None, return_weights=False):
    q = torch.cat([
        rearrange(q_mv * _load_inner_product_factors(device=q_mv.device, dtype=q_mv.dtype), "... c x -> ... (c x)"),
        q_s,
    ], -1)
    k = torch.cat([rearrange(k_mv, "... c x -> ... (c x)"), k_s], -1)
    v = torch.cat([rearrange(v_mv, "... c x -> ... (c x)"), v_s], -1)
    
    v_out = manual_attention_with_capture(q, k, v, attn_mask)
    
    num_channels_out = v_mv.shape[-2]
    v_out_mv = rearrange(v_out[..., : num_channels_out * 16], "... (c x) -> ...  c x", x=16)
    v_out_s = v_out[..., num_channels_out * 16 :]
    return v_out_mv, v_out_s

print("Applying attention monkey-patch...")
gatr.layers.attention.attention.sdp_attention = sdp_attention_patched
print("Patch applied.")

# ==============================================================================
# 2. PLOTTING FUNCTION
# ==============================================================================
def plot_attention(attention_matrix, title_prefix="Head", save_path="plot.png"):
    
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif" # Also fix the 'Charter' font warning
    })
    
    num_heads = attention_matrix.shape[0]
    cols = min(num_heads, 4)
    rows = math.ceil(num_heads / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    if rows == 1 and cols == 1: axes = [axes]
    axes = np.array(axes).flatten()

    for i in range(num_heads):
        sns.heatmap(attention_matrix[i], ax=axes[i], cmap='viridis', square=True, cbar=True)
        axes[i].set_title(f"{title_prefix} {i}")
        axes[i].set_xlabel("Key")
        axes[i].set_ylabel("Query")

    for i in range(num_heads, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================
@hydra.main(config_path="config", config_name="toptagging", version_base=None)
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- FIX 1: Override Config to match CHECKPOINT ---
    print("Forcing config overrides (12 blocks, 16 channels)...")
    cfg.model.net.num_blocks = 12
    cfg.model.net.hidden_mv_channels = 16
    cfg.model.net.hidden_s_channels = 32
    cfg.model.net.attention.num_heads = 8
    cfg.model.net.attention.head_scale = True
    cfg.model.net.attention.increase_hidden_channels = 2
    if 'attention' not in cfg.model.net:
        cfg.model.net.attention = {}
    cfg.model.net.attention.head_scale = True

    # --- A. Initialize Experiment ---
    print("Initializing TopTaggingExperiment...")
    exp = TopTaggingExperiment(cfg, rank=0, world_size=1) 
    
    # Set flags manually
    exp.warm_start = False      
    exp.device = device         
    exp.dtype = torch.float32   

    exp.init_physics()
    exp.init_geometric_algebra()
    exp.init_model()
    
    # --- B. Load Checkpoint with Key Renaming ---
    checkpoint_path = "results/models/topt_GATr_7327_run0_it200000.pt"
    
    if not os.path.exists(checkpoint_path):
        alt_path = os.path.join("lorentz-gatr", checkpoint_path) 
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        elif os.path.exists(os.path.basename(checkpoint_path)):
             checkpoint_path = os.path.basename(checkpoint_path)

    if os.path.exists(checkpoint_path):
        print(f"Loading weights from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = state["model"] if "model" in state else state
        
        # FIX 2: Rename Keys (Old Code -> New Code)
        new_state = {}
        for k, v in model_state.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_key = new_key.replace("attention.qkv_in_linear", "attention.qkv_module.in_linear")
            new_state[new_key] = v
            
        missing, unexpected = exp.model.load_state_dict(new_state, strict=False)
        if len(missing) > 0:
            print(f"WARNING: Missing keys: {len(missing)}")
        else:
            print("Weights loaded perfectly!")
            
    else:
        print(f"ERROR: Weights not found at {checkpoint_path}")
        return

    exp.model.eval()
    
    # --- C. Load Data (ORIGINAL DATASET) ---
    print("Loading ORIGINAL datasets (this may take time)...")
    # This calls the standard logic which checks config for 'full' or 'mini'
    exp.init_data()
    exp._init_dataloader()
    
    print("Data loaded. Preparing inference...")
    
    # --- D. Run Inference ---
    # Use validation or test loader
    loader = exp.val_loader
    iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration:
        print("Error: DataLoader is empty.")
        return

    # Move batch to device
    batch = batch.to(device)
    
    global ATTENTION_LOG
    ATTENTION_LOG = []
    
    print(f"Running inference on batch with {batch.num_graphs} jets...")
    with torch.no_grad():
        # FIX 3: Pass batch directly (No list wrapper)
        exp._get_ypred_and_label(batch)

    # --- E. Visualize ---
    if len(ATTENTION_LOG) > 0:
        raw_stack = torch.stack(ATTENTION_LOG)
        num_layers = raw_stack.shape[0]
        print(f"Captured {num_layers} layers of attention maps.")
        
        # Plot First and Last Layer
        layers_to_plot = [0, num_layers - 1]
        jet_idx = 0 
        
        print(f"\nGenerating plots for Jet {jet_idx}...")
        
        for layer_idx in layers_to_plot:
            filename = f"attention_layer_{layer_idx}.png"
            attn_matrix = raw_stack[layer_idx, jet_idx].numpy()
            plot_attention(attn_matrix, 
                           title_prefix=f"Layer {layer_idx} - Head", 
                           save_path=filename)
    else:
        print("No attention maps captured.")

if __name__ == "__main__":
    main()
