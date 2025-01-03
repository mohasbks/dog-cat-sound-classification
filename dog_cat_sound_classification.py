# -*- coding: utf-8 -*-
"""
Dog and Cat Sound Classification

A machine learning model to classify dog and cat sounds.
"""

# Install required libraries
# Uncomment and run these if not already installed
# !pip install librosa tensorflow numpy scikit-learn pandas matplotlib

# Import required libraries
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal as signal
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

class AdvancedAudioClassifier:
    def __init__(self, sr=22050, duration=3):
        self.sr = sr
        self.duration = duration
        self.model = None
        self.scaler = StandardScaler()

    def extract_features(self, audio):
        """Extract a variety of audio features"""
        features = []

        # Extract MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        features.extend(mfccs_scaled)

        # Extract Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_mean = np.mean(mel_spec_db.T, axis=0)
        features.extend(mel_spec_mean)

        # Extract additional features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]

        features.extend([
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.mean(zero_crossing_rate),
            np.std(zero_crossing_rate)
        ])

        return np.array(features)

    def augment_audio(self, audio):
        """Apply multiple techniques to increase data"""
        augmented = []
        augmented.append(audio)

        # Change speed
        augmented.append(librosa.effects.time_stretch(audio, rate=0.8))
        augmented.append(librosa.effects.time_stretch(audio, rate=1.2))

        # Change pitch
        augmented.append(librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=2))
        augmented.append(librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=-2))

        # Add noise
        noise_factor = 0.005
        noise = np.random.normal(0, 1, len(audio))
        augmented.append(audio + noise_factor * noise)

        return augmented

    def prepare_audio(self, audio_path):
        """Prepare audio file with feature extraction and data augmentation"""
        try:
            audio, _ = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            audio = librosa.util.normalize(audio)

            # Data augmentation
            augmented_audios = self.augment_audio(audio)

            # Feature extraction
            features_list = []
            for aug_audio in augmented_audios:
                features = self.extract_features(aug_audio)
                features_list.append(features)

            return np.array(features_list)
        except Exception as e:
            print(f"Error loading {audio_path}: {str(e)}")
            raise

    def build_model(self, input_shape):
        """Build advanced CNN model"""
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Reshape((-1, 1)),

            # First block
            layers.Conv1D(64, 3, padding='same', kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),

            # Second block
            layers.Conv1D(128, 3, padding='same', kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),

            # Fully connected layers
            layers.Flatten(),
            layers.Dense(256, kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X_train, y_train, validation_split=0.2, epochs=50, batch_size=32):
        """Train the model"""
        if self.model is None:
            self.model = self.build_model(X_train.shape[1:])

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        return history

def collect_audio_files(directories):
    """Collect all audio files from specified directories"""
    audio_files = []
    for directory in directories:
        if os.path.exists(directory):
            if os.path.isdir(directory):
                files = [os.path.join(directory, f) for f in os.listdir(directory)
                        if f.endswith(('.wav', '.mp3'))]
                audio_files.extend(files)
    return audio_files

def check_directory_structure():
    """Check directory structure"""
    data_dirs = [
        "ESC-50-master",
        "combined_data",
        "dog_sounds",
        "cats"
    ]

    print("Checking directory structure...")

    for data_dir in data_dirs:
        print(f"\n{'='*50}")
        print(f"Checking directory: {data_dir}")

        if os.path.exists(data_dir):
            print("Directory exists")
            print("Contents:", os.listdir(data_dir))

            # Check dogs directory
            dog_dir = os.path.join(data_dir, 'dogs')
            if os.path.exists(dog_dir):
                audio_files = [f for f in os.listdir(dog_dir) if f.endswith(('.wav', '.mp3'))]
                print(f"\nDogs directory:")
                print(f"Found {len(audio_files)} audio files")
                if audio_files:
                    print("Sample files:", audio_files[:5])
            else:
                print("\nDogs directory not found")

            # Check cats directory
            cat_dir = os.path.join(data_dir, 'cats')
            if os.path.exists(cat_dir):
                audio_files = [f for f in os.listdir(cat_dir) if f.endswith(('.wav', '.mp3'))]
                print(f"\nCats directory:")
                print(f"Found {len(audio_files)} audio files")
                if audio_files:
                    print("Sample files:", audio_files[:5])
            else:
                print("\nCats directory not found")
        else:
            print("Directory does not exist")

# Check directory structure
check_directory_structure()

# Initialize audio classifier
print("\nInitializing Audio Classifier...")
classifier = AdvancedAudioClassifier()

# Specify data sources
combined_data_dir = "combined_data"

dog_sources = [
    "dog_sounds",
    os.path.join(combined_data_dir, "dogs")
]

cat_sources = [
    "cats",
    os.path.join(combined_data_dir, "cats")
]

# Collect audio files
print("\nCollecting audio files...")
dog_files = collect_audio_files(dog_sources)
cat_files = collect_audio_files(cat_sources)

print(f"Found {len(dog_files)} dog audio files")
print(f"Found {len(cat_files)} cat audio files")

# Process audio files
X_all = []
y_all = []

# Process dog files
print("\nProcessing dog files...")
for file in dog_files:
    try:
        features = classifier.prepare_audio(file)
        X_all.extend(features)
        y_all.extend([1] * len(features))
        print(f"Processed: {os.path.basename(file)}")
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        continue

# Process cat files
print("\nProcessing cat files...")
for file in cat_files:
    try:
        features = classifier.prepare_audio(file)
        X_all.extend(features)
        y_all.extend([0] * len(features))
        print(f"Processed: {os.path.basename(file)}")
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        continue

# Check if data is sufficient
if len(X_all) == 0 or len(y_all) == 0:
    raise ValueError("No data was processed! Check your audio files.")

X_all = np.array(X_all)
y_all = np.array(y_all)

print("\nFinal Dataset Summary:")
print(f"Total samples: {len(X_all)}")
print(f"Dogs: {sum(y_all)} samples")
print(f"Cats: {len(y_all) - sum(y_all)} samples")

# Split data and train model
print("\nSplitting Dataset...")
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

print("\nTraining Model...")
history = classifier.train(X_train, y_train)

print("\nEvaluating Model...")
test_loss, test_accuracy = classifier.model.evaluate(X_test, y_test)
print(f"Model Test Accuracy: {test_accuracy*100:.2f}%")