#!/usr/bin/env python3
"""
MIDI Data Preprocessor for Composer Classification
Processes MIDI files and extracts features for machine learning
"""

import os
import numpy as np
import pandas as pd
import pretty_midi
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class MIDIPreprocessor:
    def __init__(self, raw_data_dir="data/raw", processed_data_dir="data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.composers = ["bach", "beethoven", "chopin", "mozart"]
        self.label_encoder = LabelEncoder()
        
        # Feature extraction parameters
        self.chunk_duration = 30  # seconds
        self.min_notes = 10  # minimum notes per chunk
        
    def load_midi_file(self, filepath):
        """Load and validate MIDI file"""
        try:
            pm = pretty_midi.PrettyMIDI(str(filepath))
            if len(pm.instruments) == 0:
                return None
            return pm
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def extract_note_features(self, pm):
        """Extract note-based features from MIDI"""
        features = {}
        
        # Combine all instruments
        all_notes = []
        for instrument in pm.instruments:
            if not instrument.is_drum:
                all_notes.extend(instrument.notes)
        
        if len(all_notes) == 0:
            return None
        
        # Sort notes by start time
        all_notes.sort(key=lambda x: x.start)
        
        # Extract basic features
        pitches = [note.pitch for note in all_notes]
        velocities = [note.velocity for note in all_notes]
        durations = [note.end - note.start for note in all_notes]
        intervals = [all_notes[i+1].start - all_notes[i].start 
                    for i in range(len(all_notes)-1)]
        
        # Pitch statistics
        features['pitch_mean'] = np.mean(pitches)
        features['pitch_std'] = np.std(pitches)
        features['pitch_min'] = np.min(pitches)
        features['pitch_max'] = np.max(pitches)
        features['pitch_range'] = features['pitch_max'] - features['pitch_min']
        
        # Velocity statistics
        features['velocity_mean'] = np.mean(velocities)
        features['velocity_std'] = np.std(velocities)
        
        # Duration statistics
        features['duration_mean'] = np.mean(durations)
        features['duration_std'] = np.std(durations)
        
        # Interval statistics
        if len(intervals) > 0:
            features['interval_mean'] = np.mean(intervals)
            features['interval_std'] = np.std(intervals)
        else:
            features['interval_mean'] = 0
            features['interval_std'] = 0
        
        # Note density
        total_time = pm.get_end_time()
        features['note_density'] = len(all_notes) / max(total_time, 1)
        
        # Pitch class distribution (12-tone)
        pitch_classes = [pitch % 12 for pitch in pitches]
        for i in range(12):
            features[f'pitch_class_{i}'] = pitch_classes.count(i) / len(pitch_classes)
        
        return features
    
    def extract_harmonic_features(self, pm):
        """Extract harmonic and rhythmic features"""
        features = {}
        
        # Get chroma features (simplified)
        try:
            # Sample the MIDI at regular intervals
            fs = 100  # samples per second
            total_time = pm.get_end_time()
            times = np.arange(0, total_time, 1.0/fs)
            
            # Get piano roll
            piano_roll = pm.get_piano_roll(fs=fs)
            
            if piano_roll.shape[1] > 0:
                # Chroma features (12-dimensional)
                chroma = np.zeros((12, piano_roll.shape[1]))
                for i in range(piano_roll.shape[0]):
                    chroma[i % 12] += piano_roll[i]
                
                # Normalize and get statistics
                chroma_norm = chroma / (np.sum(chroma, axis=0, keepdims=True) + 1e-8)
                
                for i in range(12):
                    features[f'chroma_mean_{i}'] = np.mean(chroma_norm[i])
                    features[f'chroma_std_{i}'] = np.std(chroma_norm[i])
                
                # Tempo estimation (simplified)
                onset_times = []
                for instrument in pm.instruments:
                    if not instrument.is_drum:
                        onset_times.extend([note.start for note in instrument.notes])
                
                if len(onset_times) > 1:
                    onset_times.sort()
                    onset_intervals = np.diff(onset_times)
                    # Remove very short intervals (likely ornaments)
                    onset_intervals = onset_intervals[onset_intervals > 0.1]
                    if len(onset_intervals) > 0:
                        estimated_tempo = 60.0 / np.median(onset_intervals)
                        features['estimated_tempo'] = min(max(estimated_tempo, 60), 200)
                    else:
                        features['estimated_tempo'] = 120
                else:
                    features['estimated_tempo'] = 120
            else:
                # Default values if no piano roll
                for i in range(12):
                    features[f'chroma_mean_{i}'] = 0
                    features[f'chroma_std_{i}'] = 0
                features['estimated_tempo'] = 120
                
        except Exception as e:
            print(f"Error extracting harmonic features: {e}")
            # Default values
            for i in range(12):
                features[f'chroma_mean_{i}'] = 0
                features[f'chroma_std_{i}'] = 0
            features['estimated_tempo'] = 120
        
        return features
    
    def extract_features_from_midi(self, filepath, composer):
        """Extract all features from a MIDI file"""
        pm = self.load_midi_file(filepath)
        if pm is None:
            return None
        
        # Extract different types of features
        note_features = self.extract_note_features(pm)
        if note_features is None:
            return None
        
        harmonic_features = self.extract_harmonic_features(pm)
        
        # Combine all features
        features = {**note_features, **harmonic_features}
        features['composer'] = composer
        features['filename'] = filepath.name
        features['duration'] = pm.get_end_time()
        
        return features
    
    def process_all_files(self):
        """Process all MIDI files and extract features"""
        all_features = []
        
        print("Processing MIDI files...")
        
        for composer in self.composers:
            composer_dir = self.raw_data_dir / composer
            if not composer_dir.exists():
                print(f"Directory not found: {composer_dir}")
                continue
            
            midi_files = list(composer_dir.glob("*.mid")) + list(composer_dir.glob("*.midi"))
            print(f"Processing {len(midi_files)} files for {composer}")
            
            for midi_file in midi_files:
                features = self.extract_features_from_midi(midi_file, composer)
                if features is not None:
                    all_features.append(features)
                    print(f"  Processed: {midi_file.name}")
        
        if len(all_features) == 0:
            print("No features extracted!")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        print(f"\\nExtracted features from {len(df)} files")
        print(f"Feature dimensions: {df.shape}")
        print(f"Composers distribution:")
        print(df['composer'].value_counts())
        
        return df
    
    def prepare_training_data(self, df):
        """Prepare data for machine learning"""
        # Separate features and labels
        feature_columns = [col for col in df.columns 
                          if col not in ['composer', 'filename', 'duration']]
        
        X = df[feature_columns].values
        y = df['composer'].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Handle any NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\\nTraining data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, feature_columns):
        """Save processed data and metadata"""
        # Save training data
        np.save(self.processed_data_dir / "X_train.npy", X_train)
        np.save(self.processed_data_dir / "X_test.npy", X_test)
        np.save(self.processed_data_dir / "y_train.npy", y_train)
        np.save(self.processed_data_dir / "y_test.npy", y_test)
        
        # Save metadata
        metadata = {
            'feature_columns': feature_columns,
            'label_encoder': self.label_encoder,
            'composers': self.composers,
            'n_features': len(feature_columns),
            'n_classes': len(self.label_encoder.classes_)
        }
        
        with open(self.processed_data_dir / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save feature names for reference
        with open(self.processed_data_dir / "feature_names.txt", 'w') as f:
            for i, feature in enumerate(feature_columns):
                f.write(f"{i}: {feature}\\n")
        
        print(f"\\nSaved processed data to {self.processed_data_dir}")
    
    def run_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        print("Starting MIDI preprocessing...")
        
        # Process all files
        df = self.process_all_files()
        if df is None:
            return False
        
        # Save raw features
        df.to_csv(self.processed_data_dir / "raw_features.csv", index=False)
        
        # Prepare training data
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_training_data(df)
        
        # Save processed data
        self.save_processed_data(X_train, X_test, y_train, y_test, feature_columns)
        
        print("Preprocessing completed successfully!")
        return True

if __name__ == "__main__":
    preprocessor = MIDIPreprocessor("../data/raw", "../data/processed")
    preprocessor.run_preprocessing()

