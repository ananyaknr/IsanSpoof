"""
Batch feature extraction across all datasets listed in the protocol file.

Usage:
    python src/extract_features.py \
        --protocol protocols/protocol.txt \
        --output   data/processed

Saves one .npy file per utterance per feature type under:
    data/processed/<dataset>/features/<lfcc|mfcc|cqcc>/<split>/<utt_id>.npy
"""
import argparse
import numpy as np
from pathlib import Path

from preprocess import preprocess_file
from features import extract_lfcc, extract_mfcc, extract_cqcc

FEATURE_EXTRACTORS = {
    'lfcc': extract_lfcc,
    'mfcc': extract_mfcc,
    'cqcc': extract_cqcc,
}


def extract_all(protocol_path: str, output_root: str):
    output_root = Path(output_root)
    failed = []

    with open(protocol_path) as f:
        lines = [l.strip().split('\t') for l in f if l.strip() and not l.startswith('#')]

    total = len(lines)
    for i, parts in enumerate(lines):
        if len(parts) < 6:
            continue
        utt_id, dataset, _, label, split, wav_path = parts

        try:
            audio = preprocess_file(wav_path)
        except Exception as e:
            print(f"  [FAIL] {utt_id}: {e}")
            failed.append((utt_id, str(e)))
            continue

        for feat_name, extractor in FEATURE_EXTRACTORS.items():
            out_dir = output_root / dataset / 'features' / feat_name / split
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{utt_id}.npy"
            if out_path.exists():
                continue  # skip already-extracted
            feat = extractor(audio)
            np.save(out_path, feat)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{total} done...")

    print(f"\nExtraction complete. Failed: {len(failed)}")
    if failed:
        with open(output_root / 'failed.txt', 'w') as f:
            for utt_id, err in failed:
                f.write(f"{utt_id}\t{err}\n")
        print(f"Failed list saved → {output_root}/failed.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default='protocols/protocol.txt')
    parser.add_argument('--output',   default='data/processed')
    args = parser.parse_args()
    extract_all(args.protocol, args.output)
