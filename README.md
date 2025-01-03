# Dog and Cat Sound Classification 🐕 🐱

## Overview
A machine learning project that classifies audio sounds between dogs and cats using advanced deep learning techniques.

## Features
- Audio feature extraction using Librosa
- Data augmentation techniques
- Convolutional Neural Network (CNN) for sound classification
- Supports various audio file formats (WAV, MP3)

## Prerequisites
- Python 3.8+
- GPU recommended for training (optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mohasbks/dog-cat-sound-classification.git
cd dog-cat-sound-classification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation
- Organize your audio files in two directories:
  - `dog_sounds/`: Contains dog sound audio files
  - `cats/`: Contains cat sound audio files

## Usage
```bash
python dog_cat_sound_classification.py
```

## Model Architecture
- 1D Convolutional Neural Network
- Feature extraction using:
  - Mel-frequency cepstral coefficients (MFCCs)
  - Mel Spectrogram
  - Spectral features

## Performance
- Supports data augmentation
- Handles various audio qualities
- Provides detailed training metrics

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License
MIT License - see the LICENSE file for details

## Acknowledgments
- Librosa for audio processing
- TensorFlow for deep learning
- ESC-50 dataset for sound classification
