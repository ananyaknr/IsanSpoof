"""
Run this script once Person A has delivered the raw data folders.

Usage:
    python src/build_protocol.py --data_root data/raw --output protocols/protocol.txt
"""
import random
import argparse
from pathlib import Path

DATASET_CONFIGS = {
    'typhoon_isan':    {'label': 'bonafide', 'prefix': 'IS'},
    'asvspoof2019_la': {'label': 'spoof',    'prefix': 'AS'},
    'thaispoof':       {'label': 'spoof',    'prefix': 'TS'},
    'isan_tts_spoofs': {'label': 'spoof',    'prefix': 'IT'},
}

SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train / dev / eval
RANDOM_SEED  = 42


def build_protocol(data_root: str, output_path: str):
    random.seed(RANDOM_SEED)
    data_root   = Path(data_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for dataset_name, cfg in DATASET_CONFIGS.items():
        folder = data_root / dataset_name
        if not folder.exists():
            print(f"  [skip] {dataset_name} — folder not found")
            continue

        wav_files = sorted(folder.glob('**/*.wav'))
        random.shuffle(wav_files)
        n = len(wav_files)
        n_train = int(n * SPLIT_RATIOS[0])
        n_dev   = int(n * SPLIT_RATIOS[1])
        n_eval  = n - n_train - n_dev

        splits = [('train', n_train), ('dev', n_dev), ('eval', n_eval)]
        idx = 0
        for split_name, count in splits:
            for f in wav_files[idx: idx + count]:
                utt_id = f"{cfg['prefix']}_{f.stem}"
                entries.append('\t'.join([
                    utt_id, dataset_name, f.stem,
                    cfg['label'], split_name, str(f)
                ]))
            idx += count
        print(f"  [ok] {dataset_name}: {n} files")

    with open(output_path, 'w') as out:
        out.write('\n'.join(entries) + '\n')
    print(f"\nProtocol written → {output_path}  ({len(entries)} entries)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/raw')
    parser.add_argument('--output',    default='protocols/protocol.txt')
    args = parser.parse_args()
    build_protocol(args.data_root, args.output)
