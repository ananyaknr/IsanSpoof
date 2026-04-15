import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class AntiSpoofDataset(Dataset):
    """
    PyTorch Dataset for Isan anti-spoofing.

    Expects a protocol file with tab-separated columns:
        utt_id  dataset  speaker_id  label  split  wav_path

    Args:
        protocol_path: path to the protocol .txt file
        feature_root:  root of processed features directory
        feature_type:  one of 'lfcc', 'mfcc', 'cqcc'
        split:         one of 'train', 'dev', 'eval'
    """

    LABEL_MAP = {'bonafide': 1, 'spoof': 0}

    def __init__(self, protocol_path: str, feature_root: str,
                 feature_type: str = 'lfcc', split: str = 'train'):
        self.samples = []
        feature_root = Path(feature_root)

        with open(protocol_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) < 6:
                    continue
                utt_id, dataset, _, label, s, _ = parts
                if s != split:
                    continue
                feat_path = (feature_root / dataset / 'features'
                             / feature_type / split / f"{utt_id}.npy")
                if feat_path.exists():
                    self.samples.append((feat_path, self.LABEL_MAP[label]))

        print(f"[AntiSpoofDataset] split={split}, feature={feature_type}, "
              f"samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, label = self.samples[idx]
        feat = np.load(feat_path)           # (T, 180)
        return torch.FloatTensor(feat), torch.tensor(label, dtype=torch.long)
