#!/usr/bin/env python3
"""Ingest new DICOM files from data/raw_mri_new and convert to .npy."""

import os
import sys
import numpy as np
import pandas as pd
from glob import glob

try:
    import pydicom
except ImportError:
    print("Installing pydicom...")
    os.system("pip install pydicom -q")
    import pydicom

ROOT = os.path.dirname(os.path.abspath(__file__))
NEW_RAW = os.path.join(ROOT, 'data', 'raw_mri_new')
PROCESSED = os.path.join(ROOT, 'data', 'processed')
LABELS_CSV = os.path.join(ROOT, 'data', 'labels_processed.csv')

def find_dicom_files(directory, recursive=True):
    """Find all .dcm files in directory"""
    if recursive:
        pattern = os.path.join(directory, '**', '*.dcm')
        files = glob(pattern, recursive=True)
    else:
        pattern = os.path.join(directory, '*.dcm')
        files = glob(pattern)
    return sorted(files)

def load_and_convert_dicom_series(dcm_files):
    """Load DICOM series and convert to numpy array"""
    if not dcm_files:
        return None
    
    slices = []
    for dcm_file in dcm_files:
        try:
            ds = pydicom.dcmread(dcm_file)
            arr = ds.pixel_array.astype(np.float32)
            slices.append(arr)
        except Exception as e:
            print(f"  Warning: Failed to read {dcm_file}: {e}")
            continue
    
    if not slices:
        return None
    
    # Stack slices into 3D volume
    volume = np.stack(slices, axis=0)
    return volume

def ingest_new_data(subject_label='CN', ask_for_label=True):
    """
    Ingest DICOM files from raw_mri_new.
    
    Args:
        subject_label: Default label for all subjects (CN, MCI, AD)
        ask_for_label: If True, ask user for each subject
    """
    if not os.path.isdir(NEW_RAW):
        print(f"Directory not found: {NEW_RAW}")
        return
    
    print(f"Searching for DICOM files in {NEW_RAW}...")
    dcm_files = find_dicom_files(NEW_RAW, recursive=True)
    print(f"Found {len(dcm_files)} DICOM files")
    
    if not dcm_files:
        print("No DICOM files found!")
        return
    
    # Group by subject folder (first level folder name)
    subjects = {}
    for dcm_file in dcm_files:
        rel_path = os.path.relpath(dcm_file, NEW_RAW)
        subject_id = rel_path.split(os.sep)[0]
        if subject_id not in subjects:
            subjects[subject_id] = []
        subjects[subject_id].append(dcm_file)
    
    print(f"\nFound {len(subjects)} subjects")
    print("Converting to .npy files...\n")
    
    os.makedirs(PROCESSED, exist_ok=True)
    new_entries = []
    
    for subject_id, files in sorted(subjects.items()):
        print(f"Processing {subject_id} ({len(files)} DICOM files)...", end=' ')
        
        # Convert DICOM series to volume
        volume = load_and_convert_dicom_series(files)
        if volume is None:
            print("FAILED (no valid DICOMs)")
            continue
        
        # Normalize
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        # Save as .npy
        out_file = os.path.join(PROCESSED, f"{subject_id}.npy")
        np.save(out_file, volume)
        print(f"✓ Saved ({volume.shape})")
        
        # Get label
        if ask_for_label:
            label = input(f"  Label for {subject_id} (CN/MCI/AD)? [default={subject_label}]: ").strip()
            if label not in ['CN', 'MCI', 'AD']:
                label = subject_label
        else:
            label = subject_label
        
        new_entries.append({'subject_id': subject_id, 'label': label, 'file': f"{subject_id}.npy"})
    
    # Merge with existing labels
    if os.path.exists(LABELS_CSV):
        existing_df = pd.read_csv(LABELS_CSV)
        print(f"\nExisting dataset: {len(existing_df)} subjects")
    else:
        existing_df = pd.DataFrame(columns=['subject_id', 'label', 'file'])
    
    new_df = pd.DataFrame(new_entries)
    merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Remove duplicates (keep new)
    merged_df = merged_df.drop_duplicates(subset=['subject_id'], keep='last')
    merged_df.to_csv(LABELS_CSV, index=False)
    
    print(f"\n✓ Updated {LABELS_CSV}")
    print(f"  Total subjects: {len(merged_df)}")
    print(f"  New subjects: {len(new_df)}")
    print(f"  Distribution:\n{merged_df['label'].value_counts().to_string()}")

if __name__ == '__main__':
    print("="*70)
    print("DICOM Ingestion Pipeline")
    print("="*70)
    
    # Check if data exists
    dcm_count = len(find_dicom_files(NEW_RAW))
    if dcm_count == 0:
        print(f"No DICOM files found in {NEW_RAW}")
        print("Please upload your .dcm files first")
        sys.exit(1)
    
    print(f"\nFound {dcm_count} DICOM files")
    
    # Ask for default label
    default_label = input("\nDefault label for all subjects (CN/MCI/AD)? [default=CN]: ").strip() or 'CN'
    ask_per_subject = input("Ask for label per subject? (y/n) [default=n]: ").strip().lower() == 'y'
    
    ingest_new_data(subject_label=default_label, ask_for_label=ask_per_subject)
    
    print("\n" + "="*70)
    print("Ingestion complete!")
    print("="*70)
