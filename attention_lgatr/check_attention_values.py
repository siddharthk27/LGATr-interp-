#!/usr/bin/env python3
"""Quick check of extracted attention values."""

import numpy as np
import sys
from pathlib import Path

# Check one of the generated heatmap images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load and check a heatmap
heatmap_path = Path("results/interp1/attention/jet0_QCD_heatmap_head0.png")

if heatmap_path.exists():
    img = mpimg.imread(heatmap_path)
    print(f"Heatmap image loaded: {heatmap_path}")
    print(f"Image shape: {img.shape}")
    print(f"Image value range: [{img.min():.4f}, {img.max():.4f}]")
    print(f"Image mean: {img.mean():.4f}")
    
    # Check if image is all one color (would indicate all zeros or uniform attention)
    if img.std() < 0.01:
        print("⚠️  WARNING: Image has very low variance - attention may be uniform or all zeros!")
    else:
        print("✓ Image has variation - attention patterns exist!")
else:
    print(f"❌ Heatmap file not found: {heatmap_path}")

# Also check the distribution plot
dist_path = Path("results/interp1/attention/jet0_QCD_distribution.png")
if dist_path.exists():
    img = mpimg.imread(dist_path)
    print(f"\nDistribution image loaded: {dist_path}")
    print(f"Image shape: {img.shape}")
    print(f"Image value range: [{img.min():.4f}, {img.max():.4f}]")
else:
    print(f"❌ Distribution file not found: {dist_path}")

print("\n" + "="*60)
print("To manually inspect:")
print(f"  Open: {heatmap_path}")
print(f"  Open: {dist_path}")
print("="*60)
