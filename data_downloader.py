#!/usr/bin/env python3
"""
MIDI Data Downloader for Composer Classification
Downloads MIDI files for Bach, Beethoven, Chopin, and Mozart from various sources
"""

import os
import requests
import zipfile
import shutil
from pathlib import Path
import time
from urllib.parse import urljoin, urlparse
import re

class MIDIDataDownloader:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.composers = ["bach", "beethoven", "chopin", "mozart"]
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def create_directories(self):
        """Create directory structure for each composer"""
        for composer in self.composers:
            composer_dir = self.data_dir / composer
            composer_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {composer_dir}")
    
    def download_file(self, url, filepath, timeout=30):
        """Download a file from URL to filepath"""
        try:
            response = self.session.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded: {filepath.name}")
            return True
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False
    
    def download_from_piano_midi_de(self):
        """Download MIDI files from piano-midi.de"""
        print("Downloading from piano-midi.de...")
        
        # Define specific pieces for each composer
        pieces = {
            "bach": [
                "bach_846.mid",  # Well-Tempered Clavier
                "bach_847.mid",
                "bach_850.mid",
                "bach_goldberg_aria.mid",
                "bach_invention_1.mid",
                "bach_invention_4.mid",
                "bach_invention_8.mid",
                "bach_wtc1_prelude1.mid",
                "bach_wtc1_fugue1.mid",
                "bach_french_suite_1.mid"
            ],
            "beethoven": [
                "beethoven_opus27_1_mov1.mid",  # Moonlight Sonata
                "beethoven_opus27_1_mov2.mid",
                "beethoven_opus27_1_mov3.mid",
                "beethoven_opus49_1_mov1.mid",
                "beethoven_opus49_1_mov2.mid",
                "beethoven_opus14_1_mov1.mid",
                "beethoven_opus14_1_mov2.mid",
                "beethoven_pathetique_1.mid",
                "beethoven_pathetique_2.mid",
                "beethoven_pathetique_3.mid"
            ],
            "chopin": [
                "chopin_op9_1.mid",  # Nocturnes
                "chopin_op9_2.mid",
                "chopin_op15_1.mid",
                "chopin_op27_1.mid",
                "chopin_op27_2.mid",
                "chopin_waltz_op64_1.mid",  # Minute Waltz
                "chopin_waltz_op64_2.mid",
                "chopin_etude_op10_1.mid",
                "chopin_etude_op10_3.mid",
                "chopin_polonaise_op40_1.mid"
            ],
            "mozart": [
                "mozart_k331_1.mid",  # Piano Sonata K.331
                "mozart_k331_2.mid",
                "mozart_k331_3.mid",
                "mozart_k545_1.mid",  # Piano Sonata K.545
                "mozart_k545_2.mid",
                "mozart_k545_3.mid",
                "mozart_k279_1.mid",
                "mozart_k279_2.mid",
                "mozart_k279_3.mid",
                "mozart_k330_1.mid"
            ]
        }
        
        base_url = "http://piano-midi.de/midi_files/"
        
        for composer, filenames in pieces.items():
            composer_dir = self.data_dir / composer
            for filename in filenames:
                url = urljoin(base_url, filename)
                filepath = composer_dir / filename
                
                if not filepath.exists():
                    if self.download_file(url, filepath):
                        time.sleep(1)  # Be respectful to the server
    
    def download_sample_midis(self):
        """Download some sample MIDI files for testing"""
        print("Creating sample MIDI files...")
        
        # We'll create some simple MIDI files using pretty_midi for testing
        try:
            import pretty_midi
            
            for composer in self.composers:
                composer_dir = self.data_dir / composer
                
                # Create a simple test MIDI file for each composer
                pm = pretty_midi.PrettyMIDI()
                instrument = pretty_midi.Instrument(program=0)  # Piano
                
                # Add some notes (different patterns for each composer)
                if composer == "bach":
                    # Bach-like pattern: arpeggios
                    notes = [60, 64, 67, 72, 67, 64, 60, 64, 67, 72]
                elif composer == "beethoven":
                    # Beethoven-like pattern: strong chords
                    notes = [48, 52, 55, 60, 64, 67, 72, 76]
                elif composer == "chopin":
                    # Chopin-like pattern: flowing melody
                    notes = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76]
                elif composer == "mozart":
                    # Mozart-like pattern: classical scales
                    notes = [60, 62, 64, 65, 67, 69, 71, 72]
                
                for i, note_number in enumerate(notes):
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=note_number,
                        start=i * 0.5,
                        end=(i + 1) * 0.5
                    )
                    instrument.notes.append(note)
                
                pm.instruments.append(instrument)
                
                # Save the MIDI file
                sample_file = composer_dir / f"{composer}_sample.mid"
                pm.write(str(sample_file))
                print(f"Created sample file: {sample_file}")
                
        except ImportError:
            print("pretty_midi not available, skipping sample creation")
    
    def download_all(self):
        """Download MIDI files from all sources"""
        print("Starting MIDI data download...")
        self.create_directories()
        
        # Try to download from piano-midi.de
        self.download_from_piano_midi_de()
        
        # Create sample files for testing
        self.download_sample_midis()
        
        # Report download statistics
        self.report_statistics()
    
    def report_statistics(self):
        """Report download statistics"""
        print("\n=== Download Statistics ===")
        total_files = 0
        
        for composer in self.composers:
            composer_dir = self.data_dir / composer
            midi_files = list(composer_dir.glob("*.mid")) + list(composer_dir.glob("*.midi"))
            count = len(midi_files)
            total_files += count
            print(f"{composer.capitalize()}: {count} MIDI files")
        
        print(f"Total: {total_files} MIDI files")
        
        if total_files == 0:
            print("\nNo MIDI files downloaded. You may need to:")
            print("1. Check internet connection")
            print("2. Manually download MIDI files to the data/raw directories")
            print("3. Use alternative sources like Kaggle datasets")

if __name__ == "__main__":
    downloader = MIDIDataDownloader("../data/raw")
    downloader.download_all()

