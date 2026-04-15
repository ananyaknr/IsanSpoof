"# IsanSpoof

## Overview

IsanSpoof is a project focused on audio anti-spoofing detection, specifically targeting spoofing attacks in speech data. It processes multiple audio datasets, including ASVspoof 2019 LA, ThaiSpoof, and Typhoon_Isan, to extract features commonly used in spoofing detection tasks such as Linear Frequency Cepstral Coefficients (LFCC), Mel Frequency Cepstral Coefficients (MFCC), and Constant Q Cepstral Coefficients (CQCC). The project prepares data for training machine learning models (using PyTorch) to distinguish between genuine ("bonafide") and spoofed audio samples.

## Features

- Audio preprocessing: Loading, normalization, resampling, and padding/trimming to fixed lengths.
- Feature extraction: Support for LFCC, MFCC, and CQCC features.
- Dataset management: PyTorch-compatible dataset class for easy integration with ML pipelines.
- Protocol generation: Automated splitting of data into train/dev/eval sets.
- Verification: Sanity checks for processed data and feature shapes.

## Project Structure

```
README.md
data/
    processed/          # Processed features and protocols
        asvspoof2019_la/
            protocol.txt
            features/
                cqcc/
                lfcc/
                mfcc/
        thaispoof/
        typhoon_isan/
    raw/                # Raw audio files (.wav)
        asvspoof2019_la/
        asvspoof2019_toy/
        thaispoof/
        typhoon_isan/
protocols/              # Protocol files for splits
    dev.txt
    eval.txt
    train.txt
src/                    # Source code
    build_protocol.py   # Generate protocol files from raw data
    dataset.py          # PyTorch dataset class
    extract_features.py # Batch feature extraction
    features.py         # Feature extraction functions
    preprocess.py       # Audio preprocessing
    verify.py           # Data verification script
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/ananyaknr/IsanSpoof.git
   cd IsanSpoof
   ```

2. Install dependencies (assuming Python 3.8+):
   ```
   pip install torch librosa numpy scipy
   ```

3. Place raw audio data in `data/raw/` with the appropriate subfolders (e.g., `asvspoof2019_la/`, `thaispoof/`, etc.).

## Usage

1. **Build Protocols**: Generate protocol files from raw data.
   ```
   python src/build_protocol.py --data_root data/raw --output protocols/protocol.txt
   ```

2. **Extract Features**: Process audio and extract features.
   ```
   python src/extract_features.py --protocol protocols/protocol.txt --output data/processed
   ```

3. **Verify Data**: Check the processed dataset.
   ```
   python src/verify.py
   ```

4. **Train Model**: Use the `AntiSpoofDataset` class in `dataset.py` to load data for training (e.g., with PyTorch DataLoader).

## Datasets

- **ASVspoof 2019 LA**: Logical Access dataset for spoofing detection.
- **ThaiSpoof**: Thai-language spoofing audio.
- **Typhoon_Isan**: Isan-region audio data (possibly including typhoon-related speech).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

[Specify license if applicable]" 
