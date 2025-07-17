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

parser = argparse.ArgumentParser(description='Map face tracks to speaker IDs after clustering')
parser.add_argument('--face_tracks_dir', required=True, help='Base directory containing face tracks')
parser.add_argument('--rttm_dir', required=True, help='Directory containing RTTM files')
parser.add_argument('--visual_embs_dir', required=True, help='Directory containing visual embeddings')
parser.add_argument('--output_dir', required=True, help='Output directory for renamed tracks')
parser.add_argument('--merge_tracks', action='store_true', help='Merge tracks from same speaker')
parser.add_argument('--max_gap_frames', type=int, default=10, help='Maximum gap in frames to merge tracks')
parser.add_argument('--fill_gaps', action='store_true', help='Fill gaps between tracks with interpolated faces')
parser.add_argument('--raw_video_dir', default=None, help='Directory containing raw video files for gap filling')

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

def merge_speaker_tracks(track_mappings, face_tracks_dir, video_id, output_dir, merged_tracks_dir, max_gap_frames=10):
    """Merge tracks from the same speaker that are close together"""
    import cv2
    from scipy.io import wavfile
    import subprocess
    
    # Group tracks by speaker
    speaker_tracks = {}
    for mapping in track_mappings:
        speaker_id = mapping['speaker_id']
        if speaker_id not in speaker_tracks:
            speaker_tracks[speaker_id] = []
        speaker_tracks[speaker_id].append(mapping)
    
    merged_mappings = []
    
    for speaker_id, tracks in speaker_tracks.items():
        # Sort tracks by start time
        tracks.sort(key=lambda x: x['start_time'])
        
        # Find groups of tracks to merge
        track_groups = []
        current_group = [tracks[0]]
        
        for i in range(1, len(tracks)):
            # Calculate gap between current track and previous track
            prev_track = tracks[i-1]
            curr_track = tracks[i]
            
            # Extract frame numbers from filenames
            # Format: track_XXXX_frames_XXXXXX-XXXXXX.mp4
            prev_frames = prev_track['original_filename'].split('frames_')[1].split('.mp4')[0]
            prev_end_frame = int(prev_frames.split('-')[1])
            
            curr_frames = curr_track['original_filename'].split('frames_')[1].split('.mp4')[0]
            curr_start_frame = int(curr_frames.split('-')[0])
            
            gap_frames = curr_start_frame - prev_end_frame
            
            if gap_frames <= max_gap_frames:
                # Add to current group
                current_group.append(curr_track)
            else:
                # Start new group
                track_groups.append(current_group)
                current_group = [curr_track]
        
        # Add the last group
        track_groups.append(current_group)
        
        # Merge each group
        for group_idx, group in enumerate(track_groups):
            if len(group) == 1:
                # Single track, just copy as before
                mapping = group[0]
                original_path = os.path.join(face_tracks_dir, video_id, mapping['original_filename'])
                new_filename = f"speaker_{speaker_id:02d}_segment_{group_idx:03d}.mp4"
                new_path = os.path.join(merged_tracks_dir, new_filename)
                
                if os.path.exists(original_path):
                    shutil.copy2(original_path, new_path)
                    merged_mappings.append({
                        'speaker_id': speaker_id,
                        'segment_idx': group_idx,
                        'original_tracks': [mapping['track_idx']],
                        'filename': new_filename,
                        'start_time': mapping['start_time'],
                        'end_time': mapping['end_time'],
                        'avg_score': mapping['avg_score']
                    })
            else:
                # Multiple tracks to merge
                print(f"  Merging {len(group)} tracks for speaker {speaker_id}, segment {group_idx}")
                
                # Extract start and end frames from group
                first_track = group[0]
                last_track = group[-1]
                
                # Extract frames from filenames
                first_frames = first_track['original_filename'].split('frames_')[1].split('.mp4')[0]
                start_frame = int(first_frames.split('-')[0])
                
                last_frames = last_track['original_filename'].split('frames_')[1].split('.mp4')[0]
                end_frame = int(last_frames.split('-')[1])
                
                # Create output filename
                new_filename = f"speaker_{speaker_id:02d}_segment_{group_idx:03d}_merged_{len(group)}_tracks.mp4"
                new_path = os.path.join(merged_tracks_dir, new_filename)
                
                # Create list of video files to concatenate
                concat_list_path = os.path.join(merged_tracks_dir, f"concat_list_{speaker_id}_{group_idx}.txt")
                with open(concat_list_path, 'w') as f:
                    for track in group:
                        track_path = os.path.join(face_tracks_dir, video_id, track['original_filename'])
                        f.write(f"file '{track_path}'\n")
                
                # Use ffmpeg to concatenate videos
                cmd = [
                    'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                    '-i', concat_list_path,
                    '-c:v', 'libx264', '-c:a', 'aac',
                    new_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Clean up concat list
                os.remove(concat_list_path)
                
                if result.returncode == 0:
                    merged_mappings.append({
                        'speaker_id': speaker_id,
                        'segment_idx': group_idx,
                        'original_tracks': [t['track_idx'] for t in group],
                        'filename': new_filename,
                        'start_time': group[0]['start_time'],
                        'end_time': group[-1]['end_time'],
                        'avg_score': float(np.mean([t['avg_score'] for t in group]))
                    })
                else:
                    print(f"    Error merging tracks: {result.stderr}")
    
    return merged_mappings

def interpolate_bboxes(bbox1, bbox2, num_frames):
    """Linearly interpolate bounding boxes between two tracks"""
    if num_frames <= 0:
        return []
    
    interpolated = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1) if num_frames > 1 else 0
        bbox = []
        for j in range(4):  # x1, y1, x2, y2
            bbox.append(bbox1[j] + alpha * (bbox2[j] - bbox1[j]))
        interpolated.append(bbox)
    
    return interpolated

def extract_face_from_video(video_path, frame_num, bbox, target_size=(224, 224)):
    """Extract and crop face from video at specific frame using bbox"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    # Crop face using bbox
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    
    # Resize to target size
    face = cv2.resize(face, target_size)
    return face

def merge_speaker_tracks_with_gap_filling(track_mappings, face_tracks_dir, video_id, output_dir, 
                                         merged_tracks_dir, max_gap_frames=10, fill_gaps=False,
                                         raw_video_dir=None):
    """Merge tracks from the same speaker with optional gap filling"""
    import cv2
    from scipy.io import wavfile
    import subprocess
    
    # Group tracks by speaker
    speaker_tracks = {}
    for mapping in track_mappings:
        speaker_id = mapping['speaker_id']
        if speaker_id not in speaker_tracks:
            speaker_tracks[speaker_id] = []
        speaker_tracks[speaker_id].append(mapping)
    
    merged_mappings = []
    
    # Load track metadata if gap filling is enabled
    track_metadata = {}
    if fill_gaps:
        metadata_file = os.path.join(face_tracks_dir, video_id, f"{video_id}_tracks_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata_list = json.load(f)
                for meta in metadata_list:
                    track_metadata[meta['track_idx']] = meta
    
    for speaker_id, tracks in speaker_tracks.items():
        # Sort tracks by start time
        tracks.sort(key=lambda x: x['start_time'])
        
        # Find groups of tracks to merge
        track_groups = []
        current_group = [tracks[0]]
        
        for i in range(1, len(tracks)):
            # Calculate gap between current track and previous track
            prev_track = tracks[i-1]
            curr_track = tracks[i]
            
            # Extract frame numbers from filenames
            prev_frames = prev_track['original_filename'].split('frames_')[1].split('.mp4')[0]
            prev_end_frame = int(prev_frames.split('-')[1])
            
            curr_frames = curr_track['original_filename'].split('frames_')[1].split('.mp4')[0]
            curr_start_frame = int(curr_frames.split('-')[0])
            
            gap_frames = curr_start_frame - prev_end_frame
            
            if gap_frames <= max_gap_frames:
                # Add to current group
                current_group.append(curr_track)
            else:
                # Start new group
                track_groups.append(current_group)
                current_group = [curr_track]
        
        # Add the last group
        track_groups.append(current_group)
        
        # Merge each group
        for group_idx, group in enumerate(track_groups):
            if len(group) == 1 and not fill_gaps:
                # Single track without gap filling, just copy as before
                mapping = group[0]
                original_path = os.path.join(face_tracks_dir, video_id, mapping['original_filename'])
                new_filename = f"speaker_{speaker_id:02d}_segment_{group_idx:03d}.mp4"
                new_path = os.path.join(merged_tracks_dir, new_filename)
                
                if os.path.exists(original_path):
                    shutil.copy2(original_path, new_path)
                    merged_mappings.append({
                        'speaker_id': speaker_id,
                        'segment_idx': group_idx,
                        'original_tracks': [mapping['track_idx']],
                        'filename': new_filename,
                        'start_time': mapping['start_time'],
                        'end_time': mapping['end_time'],
                        'avg_score': mapping['avg_score']
                    })
            else:
                # Multiple tracks to merge or gap filling enabled
                if fill_gaps and raw_video_dir:
                    print(f"  Merging {len(group)} tracks for speaker {speaker_id}, segment {group_idx} with gap filling")
                    
                    # Create temporary directory for segments
                    temp_dir = os.path.join(merged_tracks_dir, f"temp_{speaker_id}_{group_idx}")
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # Create list of all segments (tracks + gaps)
                    all_segments = []
                    segment_idx = 0
                    
                    for j, track in enumerate(group):
                        # Add the track video
                        track_path = os.path.join(face_tracks_dir, video_id, track['original_filename'])
                        segment_filename = f"segment_{segment_idx:04d}.mp4"
                        segment_path = os.path.join(temp_dir, segment_filename)
                        shutil.copy2(track_path, segment_path)
                        all_segments.append(segment_path)
                        segment_idx += 1
                        
                        # If not the last track, create gap video
                        if j < len(group) - 1:
                            next_track = group[j + 1]
                            
                            # Get metadata for current and next tracks
                            curr_meta = track_metadata.get(track['track_idx'])
                            next_meta = track_metadata.get(next_track['track_idx'])
                            
                            if curr_meta and next_meta and 'bboxes' in curr_meta and 'bboxes' in next_meta:
                                # Calculate gap frames
                                curr_end_frame = curr_meta['end_frame']
                                next_start_frame = next_meta['start_frame']
                                gap_frame_count = next_start_frame - curr_end_frame - 1
                                
                                if gap_frame_count > 0:
                                    # Get last bbox of current track and first bbox of next track
                                    last_bbox = curr_meta['bboxes'][-1]
                                    first_bbox = next_meta['bboxes'][0]
                                    
                                    # Interpolate bboxes for gap frames
                                    interpolated_bboxes = interpolate_bboxes(last_bbox, first_bbox, gap_frame_count)
                                    
                                    # Get video path
                                    video_path = curr_meta.get('video_path')
                                    if not video_path and raw_video_dir:
                                        video_path = os.path.join(raw_video_dir, f"{video_id}.mp4")
                                    
                                    # Extract faces for gap frames
                                    gap_segment_filename = f"segment_{segment_idx:04d}.mp4"
                                    gap_segment_path = os.path.join(temp_dir, gap_segment_filename)
                                    
                                    # Create video writer for gap segment
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                    gap_writer = cv2.VideoWriter(gap_segment_path, fourcc, 25, (224, 224))
                                    
                                    for k, bbox in enumerate(interpolated_bboxes):
                                        frame_num = curr_end_frame + k + 1
                                        face = extract_face_from_video(video_path, frame_num, bbox)
                                        if face is not None:
                                            gap_writer.write(face)
                                    
                                    gap_writer.release()
                                    
                                    if os.path.exists(gap_segment_path) and os.path.getsize(gap_segment_path) > 0:
                                        all_segments.append(gap_segment_path)
                                        segment_idx += 1
                    
                    # Create concat list
                    concat_list_path = os.path.join(temp_dir, "concat_list.txt")
                    with open(concat_list_path, 'w') as f:
                        for seg_path in all_segments:
                            f.write(f"file '{seg_path}'\n")
                    
                    # Merge all segments
                    new_filename = f"speaker_{speaker_id:02d}_segment_{group_idx:03d}_merged_{len(group)}_tracks_gap_filled.mp4"
                    new_path = os.path.join(merged_tracks_dir, new_filename)
                    
                    cmd = [
                        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                        '-i', concat_list_path,
                        '-c:v', 'libx264', '-c:a', 'aac',
                        new_path
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    # Clean up temporary files
                    shutil.rmtree(temp_dir)
                    
                    if result.returncode == 0:
                        merged_mappings.append({
                            'speaker_id': speaker_id,
                            'segment_idx': group_idx,
                            'original_tracks': [t['track_idx'] for t in group],
                            'filename': new_filename,
                            'start_time': group[0]['start_time'],
                            'end_time': group[-1]['end_time'],
                            'avg_score': float(np.mean([t['avg_score'] for t in group])),
                            'gap_filled': True
                        })
                    else:
                        print(f"    Error merging tracks with gap filling: {result.stderr}")
                else:
                    # Standard merging without gap filling (original implementation)
                    print(f"  Merging {len(group)} tracks for speaker {speaker_id}, segment {group_idx}")
                    
                    # Create output filename
                    new_filename = f"speaker_{speaker_id:02d}_segment_{group_idx:03d}_merged_{len(group)}_tracks.mp4"
                    new_path = os.path.join(merged_tracks_dir, new_filename)
                    
                    # Create list of video files to concatenate
                    concat_list_path = os.path.join(merged_tracks_dir, f"concat_list_{speaker_id}_{group_idx}.txt")
                    with open(concat_list_path, 'w') as f:
                        for track in group:
                            track_path = os.path.join(face_tracks_dir, video_id, track['original_filename'])
                            f.write(f"file '{track_path}'\n")
                    
                    # Use ffmpeg to concatenate videos
                    cmd = [
                        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                        '-i', concat_list_path,
                        '-c:v', 'libx264', '-c:a', 'aac',
                        new_path
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    # Clean up concat list
                    os.remove(concat_list_path)
                    
                    if result.returncode == 0:
                        merged_mappings.append({
                            'speaker_id': speaker_id,
                            'segment_idx': group_idx,
                            'original_tracks': [t['track_idx'] for t in group],
                            'filename': new_filename,
                            'start_time': group[0]['start_time'],
                            'end_time': group[-1]['end_time'],
                            'avg_score': float(np.mean([t['avg_score'] for t in group]))
                        })
                    else:
                        print(f"    Error merging tracks: {result.stderr}")
    
    return merged_mappings

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
        
        # Create subdirectories for original and merged tracks
        original_tracks_dir = os.path.join(video_output_dir, 'original_tracks')
        merged_tracks_dir = os.path.join(video_output_dir, 'merged_tracks')
        os.makedirs(original_tracks_dir, exist_ok=True)
        os.makedirs(merged_tracks_dir, exist_ok=True)
        
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
                    new_path = os.path.join(original_tracks_dir, new_filename)
                    
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
        
        # Merge tracks if requested
        if args.merge_tracks and track_to_speaker_mapping:
            if args.fill_gaps:
                print(f"  Merging tracks with max gap of {args.max_gap_frames} frames and gap filling...")
                merged_mappings = merge_speaker_tracks_with_gap_filling(
                    track_to_speaker_mapping, 
                    args.face_tracks_dir, 
                    video_id, 
                    video_output_dir,
                    merged_tracks_dir,
                    args.max_gap_frames,
                    fill_gaps=True,
                    raw_video_dir=args.raw_video_dir
                )
            else:
                print(f"  Merging tracks with max gap of {args.max_gap_frames} frames...")
                merged_mappings = merge_speaker_tracks(
                    track_to_speaker_mapping, 
                    args.face_tracks_dir, 
                    video_id, 
                    video_output_dir,
                    merged_tracks_dir,
                    args.max_gap_frames
                )
            
            # Save merged mapping
            merged_mapping_file = os.path.join(video_output_dir, f"{video_id}_merged_track_mapping.json")
            with open(merged_mapping_file, 'w') as f:
                json.dump(merged_mappings, f, indent=2)
            print(f"  Saved merged mapping to {merged_mapping_file}")
            
            # Use merged mappings for statistics
            final_mappings = merged_mappings
        else:
            # Save original mapping to JSON
            if track_to_speaker_mapping:
                mapping_file = os.path.join(video_output_dir, f"{video_id}_track_speaker_mapping.json")
                with open(mapping_file, 'w') as f:
                    json.dump(track_to_speaker_mapping, f, indent=2)
                print(f"  Saved mapping to {mapping_file}")
            final_mappings = track_to_speaker_mapping
        
        # Create summary statistics
        if final_mappings:
            speaker_stats = {}
            for mapping in final_mappings:
                speaker_id = mapping['speaker_id']
                if speaker_id not in speaker_stats:
                    speaker_stats[speaker_id] = {
                        'num_segments': 0,
                        'num_original_tracks': 0,
                        'total_duration': 0,
                        'avg_score': []
                    }
                speaker_stats[speaker_id]['num_segments'] += 1
                if 'original_tracks' in mapping:
                    # Merged mapping
                    speaker_stats[speaker_id]['num_original_tracks'] += len(mapping['original_tracks'])
                else:
                    # Non-merged mapping
                    speaker_stats[speaker_id]['num_original_tracks'] += 1
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