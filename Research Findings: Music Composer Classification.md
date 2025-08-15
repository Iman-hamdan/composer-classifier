# Research Findings: Music Composer Classification

## Project Overview
Based on the user's friend's project and research, I need to create a web application that can classify classical music composers (Bach, Beethoven, Chopin, Mozart) from uploaded MIDI files using deep learning.

## Key Technical Components

### 1. MIDI Processing Libraries
- **pretty_midi**: Primary library for MIDI file processing and analysis
- **miditoolkit**: Alternative library inspired by pretty_midi with similar functionality
- **mido**: Lower-level MIDI manipulation library
- **musicaiz**: Object-oriented library for symbolic music analysis and generation

### 2. Data Processing Pipeline (from GitHub example)
The classical-music-artist-classification project shows this approach:
1. **Load MIDI files** organized by composer folders
2. **Extract metadata** from filepath
3. **Split MIDI files into chunks** (15, 30, or 60 seconds)
4. **Trim chunks** to precise durations
5. **Store in DataFrame** with composer labels
6. **Create train/test splits** ensuring no composition appears in both

### 3. Feature Extraction Approaches
From research papers and implementations:
- **Sequential features**: Note sequences, chord progressions, tempo patterns
- **Spectral features**: Convert MIDI to audio then extract Mel-spectrograms, MFCCs
- **Musical features**: Harmony, rhythm, melody patterns specific to each composer
- **Time-series data**: Treating music as sequential data for LSTM processing

### 4. Deep Learning Models
- **LSTM (Long Short-Term Memory)**: For sequential music pattern recognition
- **CNN (Convolutional Neural Network)**: For pattern detection in music representations
- **Hybrid approaches**: Combining LSTM and CNN architectures

### 5. Web Application Components
From the reference application:
- **File upload interface**: Accept .mid/.midi files
- **Audio player**: Play/pause/stop controls with progress bar
- **Prediction display**: Show top K predictions with confidence scores
- **Sample files**: Provide example MIDI files for testing

## Implementation Strategy

### Phase 1: Data Collection and Preprocessing
- Download MIDI files for Bach, Beethoven, Chopin, Mozart
- Implement preprocessing pipeline using pretty_midi
- Extract musical features suitable for deep learning
- Create balanced dataset with proper train/validation/test splits

### Phase 2: Model Development
- Implement LSTM model for sequential music data
- Implement CNN model for pattern recognition
- Train and evaluate both approaches
- Select best performing model

### Phase 3: Web Application
- Create React frontend with file upload and audio player
- Develop Flask backend for model inference
- Integrate MIDI to audio conversion for playback
- Deploy complete application

## Key Libraries and Tools
- **Python**: pretty_midi, tensorflow/pytorch, pandas, numpy
- **Frontend**: React, Web Audio API for MIDI playback
- **Backend**: Flask, audio processing libraries
- **Deployment**: Standard web deployment stack

## Next Steps
1. Search for and download MIDI datasets
2. Implement data preprocessing pipeline
3. Begin model development and training

