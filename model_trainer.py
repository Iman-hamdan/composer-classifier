#!/usr/bin/env python3
"""
Deep Learning Model Trainer for Composer Classification
Implements LSTM and CNN models for classifying classical music composers
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class ComposerClassifier:
    def __init__(self, processed_data_dir="data/processed", models_dir="models"):
        self.processed_data_dir = Path(processed_data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.models = {}
        self.history = {}
        
    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        
        # Load training data
        self.X_train = np.load(self.processed_data_dir / "X_train.npy")
        self.X_test = np.load(self.processed_data_dir / "X_test.npy")
        self.y_train = np.load(self.processed_data_dir / "y_train.npy")
        self.y_test = np.load(self.processed_data_dir / "y_test.npy")
        
        # Load metadata
        with open(self.processed_data_dir / "metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.n_features = self.metadata['n_features']
        self.n_classes = self.metadata['n_classes']
        self.composers = self.metadata['composers']
        
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Test data shape: {self.X_test.shape}")
        print(f"Number of features: {self.n_features}")
        print(f"Number of classes: {self.n_classes}")
        print(f"Composers: {self.composers}")
        
        # Normalize features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Convert labels to categorical
        self.y_train_cat = keras.utils.to_categorical(self.y_train, self.n_classes)
        self.y_test_cat = keras.utils.to_categorical(self.y_test, self.n_classes)
        
        return True
    
    def create_dense_model(self):
        """Create a dense neural network model"""
        model = keras.Sequential([
            layers.Input(shape=(self.n_features,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_lstm_model(self):
        """Create LSTM model for sequential features"""
        # Reshape data for LSTM (samples, timesteps, features)
        # We'll treat each feature as a timestep
        X_train_lstm = self.X_train_scaled.reshape(self.X_train_scaled.shape[0], self.n_features, 1)
        X_test_lstm = self.X_test_scaled.reshape(self.X_test_scaled.shape[0], self.n_features, 1)
        
        model = keras.Sequential([
            layers.Input(shape=(self.n_features, 1)),
            layers.LSTM(64, return_sequences=True, dropout=0.2),
            layers.LSTM(32, dropout=0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model, X_train_lstm, X_test_lstm
    
    def create_cnn_model(self):
        """Create CNN model for pattern recognition"""
        # Reshape data for CNN (samples, height, width, channels)
        # We'll create a 2D representation of features
        feature_dim = int(np.sqrt(self.n_features)) + 1
        pad_size = feature_dim * feature_dim - self.n_features
        
        X_train_padded = np.pad(self.X_train_scaled, ((0, 0), (0, pad_size)), mode='constant')
        X_test_padded = np.pad(self.X_test_scaled, ((0, 0), (0, pad_size)), mode='constant')
        
        X_train_cnn = X_train_padded.reshape(X_train_padded.shape[0], feature_dim, feature_dim, 1)
        X_test_cnn = X_test_padded.reshape(X_test_padded.shape[0], feature_dim, feature_dim, 1)
        
        model = keras.Sequential([
            layers.Input(shape=(feature_dim, feature_dim, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model, X_train_cnn, X_test_cnn
    
    def train_model(self, model, X_train, X_test, model_name, epochs=100, batch_size=8):
        """Train a model and return history"""
        print(f"\\nTraining {model_name} model...")
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # Train model
        history = model.fit(
            X_train, self.y_train_cat,
            validation_data=(X_test, self.y_test_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, X_test, model_name):
        """Evaluate model performance"""
        print(f"\\nEvaluating {model_name} model...")
        
        # Predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Accuracy
        test_loss, test_accuracy = model.evaluate(X_test, self.y_test_cat, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Classification report
        print("\\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=self.composers))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        return {
            'accuracy': test_accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self, history, model_name):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{model_name} - Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title(f'{model_name} - Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.models_dir / f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, cm, model_name):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.composers, yticklabels=self.composers)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.models_dir / f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self, model, model_name, results):
        """Save model and results"""
        # Save model
        model.save(self.models_dir / f'{model_name}_model.h5')
        
        # Save scaler
        with open(self.models_dir / f'{model_name}_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save results
        results_to_save = {
            'accuracy': float(results['accuracy']),
            'model_name': model_name,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'composers': self.composers
        }
        
        with open(self.models_dir / f'{model_name}_results.json', 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Saved {model_name} model and results")
    
    def train_all_models(self):
        """Train all models and compare performance"""
        print("Starting model training...")
        
        if not self.load_data():
            return False
        
        model_results = {}
        
        # 1. Dense Neural Network
        print("\\n" + "="*50)
        print("TRAINING DENSE NEURAL NETWORK")
        print("="*50)
        
        dense_model = self.create_dense_model()
        dense_history = self.train_model(
            dense_model, self.X_train_scaled, self.X_test_scaled, 
            "Dense", epochs=100, batch_size=8
        )
        dense_results = self.evaluate_model(dense_model, self.X_test_scaled, "Dense")
        
        self.plot_training_history(dense_history, "Dense")
        self.plot_confusion_matrix(dense_results['confusion_matrix'], "Dense")
        self.save_model(dense_model, "dense", dense_results)
        
        model_results['dense'] = dense_results['accuracy']
        
        # 2. LSTM Model
        print("\\n" + "="*50)
        print("TRAINING LSTM MODEL")
        print("="*50)
        
        lstm_model, X_train_lstm, X_test_lstm = self.create_lstm_model()
        lstm_history = self.train_model(
            lstm_model, X_train_lstm, X_test_lstm, 
            "LSTM", epochs=100, batch_size=8
        )
        lstm_results = self.evaluate_model(lstm_model, X_test_lstm, "LSTM")
        
        self.plot_training_history(lstm_history, "LSTM")
        self.plot_confusion_matrix(lstm_results['confusion_matrix'], "LSTM")
        self.save_model(lstm_model, "lstm", lstm_results)
        
        model_results['lstm'] = lstm_results['accuracy']
        
        # 3. CNN Model
        print("\\n" + "="*50)
        print("TRAINING CNN MODEL")
        print("="*50)
        
        cnn_model, X_train_cnn, X_test_cnn = self.create_cnn_model()
        cnn_history = self.train_model(
            cnn_model, X_train_cnn, X_test_cnn, 
            "CNN", epochs=100, batch_size=8
        )
        cnn_results = self.evaluate_model(cnn_model, X_test_cnn, "CNN")
        
        self.plot_training_history(cnn_history, "CNN")
        self.plot_confusion_matrix(cnn_results['confusion_matrix'], "CNN")
        self.save_model(cnn_model, "cnn", cnn_results)
        
        model_results['cnn'] = cnn_results['accuracy']
        
        # Compare results
        print("\\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        for model_name, accuracy in model_results.items():
            print(f"{model_name.upper()}: {accuracy:.4f}")
        
        best_model = max(model_results, key=model_results.get)
        print(f"\\nBest performing model: {best_model.upper()} ({model_results[best_model]:.4f})")
        
        # Save comparison results
        with open(self.models_dir / "model_comparison.json", 'w') as f:
            json.dump(model_results, f, indent=2)
        
        return True

if __name__ == "__main__":
    trainer = ComposerClassifier("../data/processed", "../models")
    trainer.train_all_models()

