#!/usr/bin/env python3
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script maps face tracks to speaker IDs based on the clustering results.
It renames the face track files to include speaker IDs and creates a mapping file.
"""

import os
import sys
import json
import argparse
import pickle
import shutil
import numpy as np
from pathlib import Path
import cv2

parser = argparse.ArgumentParser(description='Map face tracks to speaker IDs after clustering')
parser.add_argument('--face_tracks_dir', required=True, help='Base directory containing face tracks')
parser.add_argument('--rttm_dir', required=True, help='Directory containing RTTM files')
parser.add_argument('--visual_embs_dir', required=True, help='Directory containing visual embeddings')
parser.add_argument('--output_dir', required=True, help='Output directory for renamed tracks')

def parse_rttm(rttm_file):
    """Parse RTTM file to get speaker segments"""
    segments = []
    with open(rttm_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                rec_id = parts[1]
                start_time = float(parts[3])
                duration = float(parts[4])
                speaker_id = int(parts[7])
                segments.append({
                    'rec_id': rec_id,
                    'start': start_time,
                    'end': start_time + duration,
                    'speaker_id': speaker_id
                })
    return segments

def load_track_metadata(face_tracks_dir, video_id):
    """Load track metadata JSON file"""
    metadata_file = os.path.join(face_tracks_dir, video_id, f"{video_id}_tracks_metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return None

def find_speaker_for_track(track_metadata, speaker_segments):
    """Find which speaker a track belongs to based on temporal overlap"""
    track_start = track_metadata['start_time']
    track_end = track_metadata['end_time']
    track_duration = track_end - track_start
    
    best_speaker = -1
    best_overlap = 0
    
    for segment in speaker_segments:
        # Calculate overlap between track and speaker segment
        overlap_start = max(track_start, segment['start'])
        overlap_end = min(track_end, segment['end'])
        overlap_duration = max(0, overlap_end - overlap_start)
        
        # Calculate overlap ratio
        overlap_ratio = overlap_duration / track_duration if track_duration > 0 else 0
        
        if overlap_ratio > best_overlap:
            best_overlap = overlap_ratio
            best_speaker = segment['speaker_id']
    
    return best_speaker if best_overlap > 0.5 else -1  # Require at least 50% overlap

def main():
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each video
    rttm_files = list(Path(args.rttm_dir).glob("*.rttm"))
    
    for rttm_file in rttm_files:
        video_id = rttm_file.stem
        print(f"Processing video: {video_id}")
        
        # Parse RTTM to get speaker segments
        speaker_segments = parse_rttm(rttm_file)
        
        # Load track metadata
        track_metadata = load_track_metadata(args.face_tracks_dir, video_id)
        if not track_metadata:
            print(f"  No track metadata found for {video_id}")
            continue
        
        # Create output directories for this video
        video_output_dir = os.path.join(args.output_dir, video_id)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Create subdirectory for renamed tracks
        tracks_dir = os.path.join(video_output_dir, 'tracks')
        os.makedirs(tracks_dir, exist_ok=True)
        
        # Map each track to a speaker
        track_to_speaker_mapping = []
        
        for track_info in track_metadata:
            track_idx = track_info['track_idx']
            speaker_id = find_speaker_for_track(track_info, speaker_segments)
            
            # Find original track file
            original_filename = f"track_{track_idx:04d}_frames_{track_info['start_frame']:06d}-{track_info['end_frame']:06d}.mp4"
            original_path = os.path.join(args.face_tracks_dir, video_id, original_filename)
            
            if os.path.exists(original_path):
                if speaker_id >= 0:
                    # Create new filename with speaker ID
                    new_filename = f"speaker_{speaker_id:02d}_track_{track_idx:04d}_frames_{track_info['start_frame']:06d}-{track_info['end_frame']:06d}.mp4"
                    new_path = os.path.join(tracks_dir, new_filename)
                    
                    # Copy file to new location with new name
                    shutil.copy2(original_path, new_path)
                    
                    # Add to mapping
                    track_to_speaker_mapping.append({
                        'track_idx': track_idx,
                        'speaker_id': speaker_id,
                        'original_filename': original_filename,
                        'new_filename': new_filename,
                        'start_time': track_info['start_time'],
                        'end_time': track_info['end_time'],
                        'avg_score': track_info['avg_score']
                    })
                    
                    print(f"  Track {track_idx} -> Speaker {speaker_id}")
                else:
                    print(f"  Track {track_idx} -> No speaker match (insufficient overlap)")
            else:
                print(f"  Warning: Track file not found: {original_filename}")
        
        # Save mapping to JSON
        if track_to_speaker_mapping:
            mapping_file = os.path.join(video_output_dir, f"{video_id}_track_speaker_mapping.json")
            with open(mapping_file, 'w') as f:
                json.dump(track_to_speaker_mapping, f, indent=2)
            print(f"  Saved mapping to {mapping_file}")
        
        # Create summary statistics
        if track_to_speaker_mapping:
            speaker_stats = {}
            for mapping in track_to_speaker_mapping:
                speaker_id = mapping['speaker_id']
                if speaker_id not in speaker_stats:
                    speaker_stats[speaker_id] = {
                        'num_tracks': 0,
                        'total_duration': 0,
                        'avg_score': []
                    }
                speaker_stats[speaker_id]['num_tracks'] += 1
                speaker_stats[speaker_id]['total_duration'] += mapping['end_time'] - mapping['start_time']
                speaker_stats[speaker_id]['avg_score'].append(mapping['avg_score'])
        
            # Calculate average scores
            for speaker_id in speaker_stats:
                scores = speaker_stats[speaker_id]['avg_score']
                speaker_stats[speaker_id]['avg_score'] = float(np.mean(scores))
            
            # Save summary
            summary_file = os.path.join(video_output_dir, f"{video_id}_speaker_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(speaker_stats, f, indent=2)
            print(f"  Saved summary to {summary_file}")

if __name__ == '__main__':
    main()