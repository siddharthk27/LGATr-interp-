#!/usr/bin/env python3
"""Fixed extract_attention_for_batch that processes jets individually."""

def extract_attention_for_batch_FIXED(self, batch, num_jets=10):
    """
    Extract attention weights for a batch of jets.
    
    FIXED: Processes each jet individually to get proper per-jet attention patterns.
    
    Args:
        batch: Data batch from dataloader
        num_jets: Number of jets to analyze
        
    Returns:
        Dictionary with attention weights and jet information
    """
    from torch_geometric.data import Data
    
    results = {
        'attention_maps': [],  # List of attention maps per layer per jet
        'fourmomenta': [],     # 4-momenta for each jet
        'labels': [],          # Labels for each jet
        'num_particles': [],   # Number of particles per jet
    }
    
    # Process each jet INDIVIDUALLY
    ptr = batch.ptr.cpu().numpy()
    for jet_idx in range(min(num_jets, len(ptr) - 1)):
        start_idx = ptr[jet_idx]
        end_idx = ptr[jet_idx + 1]
        num_particles = end_idx - start_idx
        
        # Create single-jet batch
        single_jet = Data(
            x=batch.x[start_idx:end_idx].clone(),
            scalars=batch.scalars[start_idx:end_idx].clone() if hasattr(batch, 'scalars') and batch.scalars is not None else None,
            ptr=torch.tensor([0, num_particles], dtype=torch.long),
            label=batch.label[jet_idx:jet_idx+1].clone()
        )
        
        if jet_idx == 0:
            print(f"Processing jet {jet_idx}: {num_particles} particles")
        
        with torch.no_grad():
            single_jet = single_jet.to(self.device)
            embedding = embed_tagging_data_into_ga(
                single_jet.x, single_jet.scalars, single_jet.ptr, self.config.data
            )
            
            # Storage for QK pairs
            qk_pairs_per_layer = []
            
            def make_attention_hook(layer_idx):
                def hook(module, input, output):
                    q_mv, k_mv, v_mv, q_s, k_s, v_s = input[0], input[1], input[2], input[3], input[4], input[5]
                    
                    if jet_idx == 0 and layer_idx == 0:
                        print(f"  Layer 0 Q/K shapes: q_mv={q_mv.shape}, q_s={q_s.shape if q_s is not None else None}")
                    
                    qk_pairs_per_layer.append({
                        'q_mv': q_mv.detach().cpu(),
                        'k_mv': k_mv.detach().cpu(),
                        'q_s': q_s.detach().cpu() if q_s is not None else None,
                        'k_s': k_s.detach().cpu() if k_s is not None else None,
                    })
                return hook
            
            # Register hooks
            hooks = []
            for layer_idx, block in enumerate(self.model.net.blocks):
                hook = block.attention.attention.register_forward_hook(
                    make_attention_hook(layer_idx)
                )
                hooks.append(hook)
            
            # Forward pass for THIS jet only
            _ = self.model(embedding)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Compute attention from Q/K pairs
            jet_attention_maps = []
            for layer_idx, qk_pair in enumerate(qk_pairs_per_layer):
                attn_weights = self._compute_attention_from_qk(
                    qk_pair['q_mv'], qk_pair['k_mv'],
                    qk_pair['q_s'], qk_pair['k_s']
                )
                
                if jet_idx == 0 and layer_idx == 0:
                    print(f"  Layer 0 attention: shape={attn_weights.shape}, range=[{attn_weights.min():.6f}, {attn_weights.max():.6f}], mean={attn_weights.mean():.6f}")
                
                jet_attention_maps.append(attn_weights)
            
            # Store results for this jet
            results['attention_maps'].append(jet_attention_maps)
            results['fourmomenta'].append(single_jet.x.cpu().numpy())
            results['labels'].append(single_jet.label[0].cpu().item())
            results['num_particles'].append(num_particles)
    
    return results
