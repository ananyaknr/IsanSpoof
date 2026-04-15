"""
Quick sanity check — run this at the end of Day 4 integration sync.

Usage:
    python src/verify.py
"""
import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dataset import AntiSpoofDataset

PROTOCOL  = 'protocols/protocol.txt'
FEAT_ROOT = 'data/processed'

for feat in ['lfcc', 'mfcc', 'cqcc']:
    try:
        ds = AntiSpoofDataset(PROTOCOL, FEAT_ROOT, feature_type=feat, split='train')
        if len(ds) == 0:
            print(f"[{feat}] No samples found — check protocol + feature paths")
            continue
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        feats, labels = next(iter(loader))
        print(f"[{feat}] feats: {tuple(feats.shape)}  labels: {tuple(labels.shape)}")
        print(f"        bonafide={int((labels==1).sum())}  spoof={int((labels==0).sum())}")
    except Exception as e:
        print(f"[{feat}] ERROR: {e}")

print("\nDone. If shapes are (32, ~400, 180) — Sprint 1 is complete.")
