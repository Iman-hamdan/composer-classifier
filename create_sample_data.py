#!/usr/bin/env python3
"""
Create Sample MIDI Data for Composer Classification
Generates synthetic MIDI files with different musical characteristics for each composer
"""

import pretty_midi
import numpy as np
from pathlib import Path
import random

class SampleMIDIGenerator:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.composers = ["bach", "beethoven", "chopin", "mozart"]
        
    def create_bach_style_midi(self, filename):
        """Create Bach-style MIDI with counterpoint and arpeggios"""
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        # Bach characteristics: counterpoint, arpeggios, baroque scales
        scales = [
            [60, 62, 64, 65, 67, 69, 71, 72],  # C major
            [62, 64, 65, 67, 69, 70, 72, 74],  # D minor
            [67, 69, 71, 72, 74, 76, 77, 79],  # G major
        ]
        
        time = 0
        for _ in range(3):  # 3 phrases
            scale = random.choice(scales)
            
            # Ascending arpeggio
            for i, note_num in enumerate(scale):
                note = pretty_midi.Note(
                    velocity=random.randint(80, 100),
                    pitch=note_num,
                    start=time + i * 0.25,
                    end=time + (i + 1) * 0.25
                )
                instrument.notes.append(note)
            
            time += len(scale) * 0.25
            
            # Descending pattern
            for i, note_num in enumerate(reversed(scale)):
                note = pretty_midi.Note(
                    velocity=random.randint(70, 90),
                    pitch=note_num,
                    start=time + i * 0.25,
                    end=time + (i + 1) * 0.25
                )
                instrument.notes.append(note)
            
            time += len(scale) * 0.25
        
        pm.instruments.append(instrument)
        pm.write(str(filename))
    
    def create_beethoven_style_midi(self, filename):
        """Create Beethoven-style MIDI with strong rhythms and dramatic contrasts"""
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        # Beethoven characteristics: strong chords, dramatic contrasts
        chord_progressions = [
            [48, 52, 55, 60],  # C major chord
            [45, 49, 52, 57],  # A minor chord
            [50, 54, 57, 62],  # D major chord
            [43, 47, 50, 55],  # G major chord
        ]
        
        time = 0
        for _ in range(4):  # 4 phrases
            # Strong chord
            chord = random.choice(chord_progressions)
            for note_num in chord:
                note = pretty_midi.Note(
                    velocity=random.randint(100, 127),
                    pitch=note_num,
                    start=time,
                    end=time + 1.0
                )
                instrument.notes.append(note)
            
            time += 1.0
            
            # Melodic passage
            melody = [chord[0] + 12 + i for i in [0, 2, 4, 5, 7, 9, 11, 12]]
            for i, note_num in enumerate(melody):
                note = pretty_midi.Note(
                    velocity=random.randint(80, 100),
                    pitch=note_num,
                    start=time + i * 0.5,
                    end=time + (i + 1) * 0.5
                )
                instrument.notes.append(note)
            
            time += len(melody) * 0.5
        
        pm.instruments.append(instrument)
        pm.write(str(filename))
    
    def create_chopin_style_midi(self, filename):
        """Create Chopin-style MIDI with flowing melodies and ornaments"""
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        # Chopin characteristics: flowing melodies, ornaments, rubato
        melodies = [
            [72, 74, 76, 77, 79, 81, 83, 84],  # High melody
            [60, 62, 64, 66, 67, 69, 71, 72],  # Mid melody
            [48, 50, 52, 53, 55, 57, 59, 60],  # Low melody
        ]
        
        time = 0
        for _ in range(3):  # 3 phrases
            melody = random.choice(melodies)
            
            # Flowing melody with varied timing
            for i, note_num in enumerate(melody):
                duration = random.uniform(0.3, 0.8)  # Rubato effect
                note = pretty_midi.Note(
                    velocity=random.randint(60, 90),
                    pitch=note_num,
                    start=time,
                    end=time + duration
                )
                instrument.notes.append(note)
                
                # Add ornament occasionally
                if random.random() < 0.3:
                    ornament = pretty_midi.Note(
                        velocity=random.randint(40, 70),
                        pitch=note_num + 1,
                        start=time + duration * 0.5,
                        end=time + duration * 0.7
                    )
                    instrument.notes.append(ornament)
                
                time += duration
        
        pm.instruments.append(instrument)
        pm.write(str(filename))
    
    def create_mozart_style_midi(self, filename):
        """Create Mozart-style MIDI with classical elegance and clarity"""
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        # Mozart characteristics: classical scales, clear structure, elegance
        classical_patterns = [
            [60, 62, 64, 65, 67, 69, 71, 72],  # C major scale
            [65, 67, 69, 71, 72, 74, 76, 77],  # F major scale
            [67, 69, 71, 72, 74, 76, 78, 79],  # G major scale
        ]
        
        time = 0
        for _ in range(4):  # 4 phrases
            pattern = random.choice(classical_patterns)
            
            # Classical scale passage
            for i, note_num in enumerate(pattern):
                note = pretty_midi.Note(
                    velocity=random.randint(70, 95),
                    pitch=note_num,
                    start=time + i * 0.4,
                    end=time + (i + 1) * 0.4
                )
                instrument.notes.append(note)
            
            time += len(pattern) * 0.4
            
            # Simple accompaniment
            bass_notes = [pattern[0] - 12, pattern[2] - 12, pattern[4] - 12]
            for i, note_num in enumerate(bass_notes):
                note = pretty_midi.Note(
                    velocity=random.randint(50, 70),
                    pitch=note_num,
                    start=time + i * 0.8,
                    end=time + (i + 1) * 0.8
                )
                instrument.notes.append(note)
            
            time += len(bass_notes) * 0.8
        
        pm.instruments.append(instrument)
        pm.write(str(filename))
    
    def generate_samples(self, num_per_composer=10):
        """Generate sample MIDI files for each composer"""
        generators = {
            "bach": self.create_bach_style_midi,
            "beethoven": self.create_beethoven_style_midi,
            "chopin": self.create_chopin_style_midi,
            "mozart": self.create_mozart_style_midi
        }
        
        print(f"Generating {num_per_composer} sample files per composer...")
        
        for composer in self.composers:
            composer_dir = self.data_dir / composer
            composer_dir.mkdir(parents=True, exist_ok=True)
            
            generator_func = generators[composer]
            
            for i in range(num_per_composer):
                filename = composer_dir / f"{composer}_generated_{i+1:02d}.mid"
                generator_func(filename)
                print(f"Created: {filename}")
        
        print(f"\\nGenerated {num_per_composer * len(self.composers)} sample MIDI files")

if __name__ == "__main__":
    generator = SampleMIDIGenerator("../data/raw")
    generator.generate_samples(num_per_composer=15)

