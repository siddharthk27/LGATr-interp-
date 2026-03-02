#!/usr/bin/env python3
"""Check the actual model structure to find attention modules."""

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

# Load config
config_path = "runs/topt/GATr_7327/config.yaml"
config = OmegaConf.load(config_path)
config.model.net.in_s_channels = 1
config.data.add_scalar_features = False

# Create model
model = instantiate(config.model)

print("="*80)
print("MODEL STRUCTURE")
print("="*80)

# Check blocks
print(f"\nNumber of blocks: {len(model.net.blocks)}")
print(f"\nFirst block type: {type(model.net.blocks[0])}")
print(f"\nFirst block attributes:")
for attr in dir(model.net.blocks[0]):
    if not attr.startswith('_'):
        print(f"  - {attr}")

print("\n" + "="*80)
print("ATTENTION MODULE PATH")
print("="*80)

block = model.net.blocks[0]
print(f"\nblock.attention type: {type(block.attention)}")
print(f"block.attention attributes:")
for attr in dir(block.attention):
    if not attr.startswith('_') and not callable(getattr(block.attention, attr)):
        print(f"  - {attr}: {type(getattr(block.attention, attr))}")

if hasattr(block.attention, 'attention'):
    print(f"\nblock.attention.attention type: {type(block.attention.attention)}")
else:
    print("\n⚠️  WARNING: block.attention does NOT have 'attention' attribute!")
    print("\nSearching for attention mechanism...")
    for attr_name in dir(block.attention):
        attr = getattr(block.attention, attr_name)
        if 'attention' in str(type(attr)).lower() or 'Attention' in attr_name:
            print(f"  Found: {attr_name} -> {type(attr)}")

print("\n" + "="*80)
print("SEARCHING FOR GEOMETRIC ATTENTION")
print("="*80)

def find_modules(module, prefix=""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        print(f"{full_name}: {type(child).__name__}")
        if 'Attention' in type(child).__name__:
            print(f"  ^^^ FOUND ATTENTION MODULE!")
        find_modules(child, full_name)

find_modules(model.net.blocks[0])
