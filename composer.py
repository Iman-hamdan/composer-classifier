import os
import sys
import tempfile
import numpy as np
import pickle
import tensorflow as tf
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pretty_midi
from sklearn.preprocessing import StandardScaler

composer_bp = Blueprint('composer', __name__)

# Global variables to store loaded models and metadata
loaded_models = {}
metadata = None
scaler = None

def load_models():
    """Load trained models and metadata"""
    global loaded_models, metadata, scaler
    
    if metadata is not None:
        return  # Already loaded
    
    try:
        # Get the backend directory path
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        models_dir = os.path.join(backend_dir, 'models')
        data_dir = os.path.join(backend_dir, 'data', 'processed')
        
        # Load metadata from processed data
        metadata_path = os.path.join(data_dir, 'metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        else:
            print(f"Metadata file not found at {metadata_path}")
            return
        
        # Load the best performing model (dense)
        model_path = os.path.join(models_dir, 'dense_model.h5')
        scaler_path = os.path.join(models_dir, 'dense_scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            loaded_models['dense'] = tf.keras.models.load_model(model_path)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("Models loaded successfully")
        else:
            print(f"Model files not found at {models_dir}")
            
    except Exception as e:
        print(f"Error loading models: {e}")

def extract_note_features(pm):
    """Extract note-based features from MIDI (same as preprocessing)"""
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

def extract_harmonic_features(pm):
    """Extract harmonic and rhythmic features (same as preprocessing)"""
    features = {}
    
    try:
        # Sample the MIDI at regular intervals
        fs = 100  # samples per second
        total_time = pm.get_end_time()
        
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

def extract_features_from_midi(filepath):
    """Extract all features from a MIDI file"""
    try:
        pm = pretty_midi.PrettyMIDI(str(filepath))
        if len(pm.instruments) == 0:
            return None
        
        # Extract different types of features
        note_features = extract_note_features(pm)
        if note_features is None:
            return None
        
        harmonic_features = extract_harmonic_features(pm)
        
        # Combine all features
        features = {**note_features, **harmonic_features}
        
        return features
        
    except Exception as e:
        print(f"Error processing MIDI file: {e}")
        return None

@composer_bp.route('/predict', methods=['POST'])
def predict_composer():
    """Predict composer from uploaded MIDI file"""
    load_models()  # Ensure models are loaded
    
    if metadata is None or 'dense' not in loaded_models or scaler is None:
        return jsonify({'error': 'Models not loaded properly'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get top_k parameter
    top_k = request.form.get('top_k', 4, type=int)
    top_k = min(max(top_k, 1), 4)  # Ensure between 1 and 4
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as tmp_file:
            file.save(tmp_file.name)
            
            # Extract features
            features = extract_features_from_midi(tmp_file.name)
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
            if features is None:
                return jsonify({'error': 'Could not extract features from MIDI file'}), 400
            
            # Prepare features for prediction
            feature_columns = metadata['feature_columns']
            feature_vector = []
            
            for col in feature_columns:
                feature_vector.append(features.get(col, 0))
            
            # Convert to numpy array and scale
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = scaler.transform(X)
            
            # Make prediction
            model = loaded_models['dense']
            predictions = model.predict(X_scaled)[0]
            
            # Get composer names and create results
            composers = metadata['composers']
            results = []
            
            for i, confidence in enumerate(predictions):
                results.append({
                    'composer': composers[i].capitalize(),
                    'confidence': float(confidence)
                })
            
            # Sort by confidence (descending)
            results.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Return top K results
            return jsonify({
                'predictions': results[:top_k],
                'filename': filename
            })
            
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@composer_bp.route('/sample/<composer>', methods=['GET'])
def get_sample_file(composer):
    """Get a sample MIDI file for testing"""
    try:
        # Path to sample files
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        samples_dir = os.path.join(backend_dir, 'data', 'raw', composer.lower())
        sample_file = os.path.join(samples_dir, f'{composer.lower()}_sample.mid')
        
        if os.path.exists(sample_file):
            return send_file(sample_file, as_attachment=True, 
                           download_name=f'{composer}_sample.mid')
        else:
            return jsonify({'error': 'Sample file not found'}), 404
            
    except Exception as e:
        return jsonify({'error': f'Error retrieving sample: {str(e)}'}), 500

@composer_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    load_models()
    
    status = {
        'status': 'healthy',
        'models_loaded': len(loaded_models) > 0,
        'metadata_loaded': metadata is not None,
        'scaler_loaded': scaler is not None
    }
    
    if metadata:
        status['composers'] = metadata['composers']
        status['n_features'] = metadata['n_features']
    
    return jsonify(status)

