#!/usr/bin/env python
"""
Complete interpretation pipeline for L-GATr top tagging model.
"""

import torch
from pathlib import Path
import argparse
from torch_geometric.loader import DataLoader

from experiments.tagging.interpret_model import LGATrInterpreter
from experiments.tagging.interpret_ga_components import GAComponentAnalyzer
from experiments.tagging.interpret_lrp import LGATrLRP
from experiments.tagging.dataset import TopTaggingDataset


def main(args):
    # Setup paths
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("L-GATr Interpretation Pipeline")
    print("="*80)
    
    # 1. Initialize interpreter
    print("\n[1/5] Loading model...")
    interpreter = LGATrInterpreter(str(checkpoint_path), str(config_path))
    
    # 2. Load data
    print("\n[2/5] Loading test data...")
    test_dataset = TopTaggingDataset(rescale_data=interpreter.config.data.rescale_data)
    test_dataset.load_data(args.data_path, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. Extract and visualize attention
    print("\n[3/5] Extracting attention maps...")
    batch = next(iter(test_loader))
    results = interpreter.extract_attention_for_batch(batch, num_jets=args.num_jets)
    
    attention_dir = output_dir / "attention"
    attention_dir.mkdir(exist_ok=True)
    
    # Plot attention distribution
    for jet_idx in range(min(args.num_jets, len(results['attention_maps']))):
        attention_maps = results['attention_maps'][jet_idx]
        fourmomenta = results['fourmomenta'][jet_idx]
        label = "Top" if results['labels'][jet_idx] else "QCD"
        
        # Distribution plot
        interpreter.plot_attention_distribution(
            attention_maps,
            save_path=attention_dir / f"jet{jet_idx}_{label}_distribution.png"
        )
        
        # Heatmaps for final layer
        final_layer_idx = len(attention_maps) - 1
        for head_idx in range(attention_maps[final_layer_idx].shape[0]):
            interpreter.visualize_attention_heatmap(
                attention_maps[final_layer_idx],
                layer_idx=final_layer_idx,
                head_idx=head_idx,
                save_path=attention_dir / f"jet{jet_idx}_{label}_heatmap_head{head_idx}.png",
                title=f"Jet {jet_idx} ({label}), Layer {final_layer_idx}, Head {head_idx}"
            )
        
        # Eta-phi visualization
        interpreter.visualize_attention_on_eta_phi(
            fourmomenta,
            attention_maps[final_layer_idx],
            head_idx=0,
            threshold=0.1,
            save_path=attention_dir / f"jet{jet_idx}_{label}_etaphi.png"
        )
    
    # 4. Geometric Algebra Component Analysis
    print("\n[4/5] Analyzing GA components...")
    ga_analyzer = GAComponentAnalyzer(interpreter.model, config=interpreter.config, device=interpreter.device)
    importance_scores = ga_analyzer.compute_component_importance(
        test_loader, 
        num_batches=10
    )
    ga_analyzer.plot_component_importance(
        importance_scores,
        save_path=output_dir / "ga_component_importance.png"
    )
    
    # 5. Layer-wise Relevance Propagation
    print("\n[5/5] Computing relevance scores...")
    lrp = LGATrLRP(interpreter.model)
    
    relevance_dir = output_dir / "relevance"
    relevance_dir.mkdir(exist_ok=True)
    
    for jet_idx in range(min(3, len(results['fourmomenta']))):
        # Prepare embedding for single jet
        fourmomenta = torch.tensor(results['fourmomenta'][jet_idx]).unsqueeze(0).to(interpreter.device)
        
        from experiments.tagging.embedding import embed_tagging_data_into_ga
        
        ptr = torch.tensor([0, len(fourmomenta[0])], dtype=torch.long, device=interpreter.device)
        embedding = embed_tagging_data_into_ga(
            fourmomenta[0], 
            torch.zeros(len(fourmomenta[0]), 0, device=interpreter.device),
            ptr,
            interpreter.config.data
        )
        
        # Compute relevance
        relevance = lrp.compute_relevance_scores(embedding, target_class=1)
        
        # Plot
        lrp.plot_particle_relevance(
            results['fourmomenta'][jet_idx],
            relevance,
            save_path=relevance_dir / f"jet{jet_idx}_relevance.png"
        )
    
    print(f"\n✓ Interpretation complete! Results saved to {output_dir}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpret L-GATr top tagging model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint (.pt file)")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config file")
    parser.add_argument("--data-path", type=str, default="data/toptagging_full.npz",
                       help="Path to test data")
    parser.add_argument("--output-dir", type=str, default="interpretation_results",
                       help="Directory to save results")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for data loading")
    parser.add_argument("--num-jets", type=int, default=5,
                       help="Number of jets to analyze in detail")
    
    args = parser.parse_args()
    main(args)
