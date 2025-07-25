# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script uses pretrained models to perform speaker visual embeddings extracting.
This script use following open source models:
    1. Face detection: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
    2. Active speaker detection: TalkNet, https://github.com/TaoRuijie/TalkNet-ASD
    3. Face quality assessment: https://modelscope.cn/models/iic/cv_manual_face-quality-assessment_fqa
    4. Face recognition: https://modelscope.cn/models/iic/cv_ir101_facerecognition_cfglint
Processing pipeline: 
    1. Face detection (input: video frames)
    2. Active speaker detection (input: consecutive face frames, audio)
    3. Face quality assessment (input: video frames)
    4. Face recognition (input: video frames)
"""


import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d
import os, time, torch, cv2, pickle, python_speech_features
import json
import subprocess

import vision_tools.face_detection as face_detection
import vision_tools.active_speaker_detection as active_speaker_detection
import vision_tools.face_recognition as face_recognition
import vision_tools.face_quality_assessment as face_quality_assessment

from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.video_manager import VideoManager


class VisionProcesser():
    def __init__(
        self, 
        video_file_path, 
        audio_file_path, 
        audio_vad, 
        out_feat_path, 
        onnx_dir, 
        conf, 
        device='cpu', 
        device_id=0, 
        out_video_path=None,
        save_tracks=False,
        face_track_dir=None
        ):
        # read audio data and check the samplerate.
        fs, self.audio = wavfile.read(audio_file_path)
        assert fs == 16000, '[ERROR]: Samplerate of wav must be 16000'
        # convert time interval to integer sampling point interval.
        audio_vad = [[int(i*16000), int(j*16000)] for (i, j) in audio_vad]
        self.video_id = os.path.basename(video_file_path).rsplit('.', 1)[0]
        self.video_file_path = video_file_path

        # read video data
        self.cap = cv2.VideoCapture(video_file_path)
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print('video %s info: w: {}, h: {}, count: {}, fps: {}'.format(w, h, self.count, self.fps) % self.video_id)

        # initial vision models
        self.face_detector = face_detection.Predictor(onnx_dir, device, device_id)
        self.speaker_detector = active_speaker_detection.ASDTalknet(onnx_dir, device, device_id)
        self.face_quality_evaluator = face_quality_assessment.FaceQualityAssess(onnx_dir, device, device_id)
        self.face_embs_extractor = face_recognition.FaceRecIR101(onnx_dir, device, device_id)

        # store facial feats along with the necessary information.
        self.active_facial_embs = {'frameI':np.empty((0,), dtype=int), 'feat':np.empty((0, 512), dtype=np.float32)}

        self.audio_vad = audio_vad
        self.out_video_path = out_video_path
        self.out_feat_path = out_feat_path
        self.save_tracks = save_tracks
        self.face_track_dir = face_track_dir
        self.audio_file_path = audio_file_path

        self.min_track = conf['min_track']
        self.num_failed_det = conf['num_failed_det']
        self.crop_scale = conf['crop_scale']
        self.min_face_size = conf['min_face_size']
        self.face_det_stride = conf['face_det_stride']
        self.shot_stride = conf.get('shot_stride', 50)  # Deprecated
        self.min_iou = conf['min_iou']
        
        # Scene detection configuration
        self.scene_threshold = conf.get('scene_threshold', 27.0)
        self.min_scene_length = conf.get('min_scene_length', 10)
        
        # Face track saving configuration
        self.track_video_fps = conf.get('track_video_fps', 25)
        self.include_audio_in_tracks = conf.get('include_audio_in_tracks', True)
        self.save_track_metadata = conf.get('save_track_metadata', True)
        self.min_avg_score_threshold = conf.get('min_avg_score_threshold', 0.5)
        self.show_scores_on_frames = conf.get('show_scores_on_frames', False)
        self.disable_face_quality_check = conf.get('disable_face_quality_check', False)
        self.asd_audio_batch_size = conf.get('asd_audio_batch_size', 400)  # Batch size for ASD to prevent GPU overflow

        if self.out_video_path is not None:
            # save the active face detection results video (for debugging).
            # Create a temporary video file without audio
            self.temp_video_path = out_video_path.rsplit('.', 1)[0] + '_temp.mp4'
            # Use fixed 25fps for output video
            self.v_out = cv2.VideoWriter(self.temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (int(w), int(h)))

        # record the time spent by each module.
        self.elapsed_time = {'faceTime':[], 'trackTime':[], 'cropTime':[],'asdTime':[], 'visTime':[], 'featTime':[]}
        
        # Store all tracks for later saving
        if self.save_tracks:
            self.all_tracks_data = []
            if self.face_track_dir:
                os.makedirs(self.face_track_dir, exist_ok=True)

    def detect_scenes(self, video_path):
        """
        Detect scenes in the video using scenedetect
        
        Returns:
            List of tuples (start_frame, end_frame) for each scene
        """
        video_manager = VideoManager([video_path])
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        
        # Add ContentDetector with threshold
        scene_manager.add_detector(ContentDetector(threshold=self.scene_threshold))
        
        # Set downscale factor for faster processing
        video_manager.set_downscale_factor()
        
        # Start video manager
        video_manager.start()
        
        # Detect scenes
        scene_manager.detect_scenes(frame_source=video_manager)
        
        # Get scene list
        scene_list = scene_manager.get_scene_list()
        
        # Convert to frame numbers
        scenes = []
        for i, scene in enumerate(scene_list):
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames() - 1  # Exclusive to inclusive
            
            # Only include scenes longer than minimum length
            if end_frame - start_frame + 1 >= self.min_scene_length:
                scenes.append((start_frame, end_frame))
        
        # If no scenes detected, use entire video as one scene
        if not scenes:
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            scenes = [(0, total_frames - 1)]
        
        print(f"Detected {len(scenes)} scenes in video")
        
        # Release video manager
        video_manager.release()
        
        return scenes

    def run(self):
        # First detect scenes in the entire video
        print(f"Detecting scenes in video {self.video_id}...")
        scenes = self.detect_scenes(self.video_file_path)
        
        # Store face detection results for VAD segments
        self.vad_face_results = {}  # frame_idx -> face detection results
        
        # Process each VAD segment with scene boundaries
        for [audio_sample_st, audio_sample_ed] in self.audio_vad:
            # frame_st and frame_ed are the starting and ending frames of current interval.
            frame_st, frame_ed = int(audio_sample_st/640), int(audio_sample_ed/640)
            
            # Find scenes that overlap with this VAD segment
            for scene_start, scene_end in scenes:
                # Check if scene overlaps with VAD segment
                overlap_start = max(frame_st, scene_start)
                overlap_end = min(frame_ed, scene_end)
                
                if overlap_start <= overlap_end:
                    # Process this scene segment
                    frames, face_det_frames = [], []
                    
                    # Set video position to start of overlap
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, overlap_start)
                    
                    # Read frames for this scene segment
                    for frame_idx in range(overlap_start, overlap_end + 1):
                        ret, frame = self.cap.read()
                        if not ret:
                            break
                            
                        if (frame_idx - overlap_start) % self.face_det_stride == 0:
                            face_det_frames.append(frame)
                        frames.append(frame)
                    
                    if len(frames) > 0:
                        # Extract audio for this scene segment
                        audio_start = overlap_start * 640
                        audio_end = (overlap_end + 1) * 640
                        audio = self.audio[audio_start:audio_end]
                        
                        # Process the scene segment
                        self.process_one_shot(frames, face_det_frames, audio, overlap_start)
        
        # Now write all frames to output video
        if self.out_video_path is not None:
            # Reset video capture to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Process all frames
            frame_idx = 0
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Check if we have face detection results for this frame
                if frame_idx in self.vad_face_results:
                    # Draw face detection results
                    for face_info in self.vad_face_results[frame_idx]:
                        self.draw_face_on_frame(frame, face_info)
                
                # Write frame (with or without face detection)
                self.v_out.write(frame)
                frame_idx += 1
            
            self.v_out.release()

        self.cap.release()
        
        if self.out_video_path is not None:
            # Merge video with audio using ffmpeg
            print(f'Merging audio into output video: {self.out_video_path}')
            # Use video filter to ensure proper frame rate and sync
            # Calculate expected video duration based on audio duration
            audio_duration = len(self.audio) / 16000.0  # Audio is 16kHz
            
            cmd = [
                'ffmpeg', '-y',
                '-i', self.temp_video_path,  # Input video
                '-i', self.audio_file_path,  # Input audio
                '-c:v', 'libx264',  # Re-encode video to ensure proper timestamps
                '-r', '25',  # Set output frame rate to 25fps
                '-c:a', 'aac',   # Audio codec
                '-ar', '16000',  # Match audio sample rate
                '-ac', '1',      # Mono audio
                '-t', str(audio_duration),  # Set duration to match audio
                '-vsync', 'cfr',  # Constant frame rate for better sync
                self.out_video_path
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                # Remove temporary video file
                os.remove(self.temp_video_path)
                print(f'Successfully created video with audio: {self.out_video_path}')
            except subprocess.CalledProcessError as e:
                print(f'Error merging audio: {e}')
                print(f'Keeping temporary video without audio: {self.temp_video_path}')

        active_facial_embs = {'embeddings':self.active_facial_embs['feat'], 'times': self.active_facial_embs['frameI']*0.04}
        pickle.dump(active_facial_embs, open(self.out_feat_path, 'wb'))
        
        # Save face tracks if enabled
        if self.save_tracks and self.face_track_dir:
            self.save_face_tracks()

        # print elapsed time
        all_elapsed_time = 0
        for k in self.elapsed_time:
            all_elapsed_time += sum(self.elapsed_time[k])
            self.elapsed_time[k] = sum(self.elapsed_time[k])
        elapsed_time_msg = 'The total processing time for %s is %.2fs, including' % (self.video_id, all_elapsed_time)
        for k in self.elapsed_time:
            elapsed_time_msg += ' %s %.2fs,'%(k, self.elapsed_time[k])
        print(elapsed_time_msg[:-1]+'.')

    def process_one_shot(self, frames, face_det_frames, audio, frame_st=None):
        curTime = time.time()
        dets = self.face_detection(face_det_frames)
        faceTime = time.time()

        allTracks, vidTracks = [], []
        allTracks.extend(self.track_shot(dets))
        trackTime = time.time()

        for ii, track in enumerate(allTracks):
            vidTracks.append(self.crop_video(track, frames, audio))
        cropTime = time.time()

        scores = self.evaluate_asd(vidTracks)
        asdTime = time.time()

        active_facial_embs = self.evaluate_fr(frames, vidTracks, scores)
        self.active_facial_embs['frameI'] = np.append(self.active_facial_embs['frameI'], active_facial_embs['frameI'] + frame_st)
        self.active_facial_embs['feat'] = np.append(self.active_facial_embs['feat'], active_facial_embs['feat'], axis=0)
        featTime = time.time()
        
        # Store track data for saving
        if self.save_tracks:
            for idx, (track, score) in enumerate(zip(vidTracks, scores)):
                track_data = {
                    'track_info': track['track'],
                    'proc_track': track['proc_track'],
                    'video_frames': track['data'][0],  # Cropped face frames
                    'audio': track['data'][1],         # Audio segment
                    'scores': score,
                    'frame_start': frame_st,
                    'track_idx': idx
                }
                self.all_tracks_data.append(track_data)

        if self.out_video_path is not None:
            self.visualization(frames, vidTracks, scores, frame_st)
        visTime = time.time()

        self.elapsed_time['faceTime'].append(faceTime-curTime)
        self.elapsed_time['trackTime'].append(trackTime-faceTime)
        self.elapsed_time['cropTime'].append(cropTime-trackTime)
        self.elapsed_time['asdTime'].append(asdTime-cropTime)
        self.elapsed_time['featTime'].append(featTime-asdTime)
        if self.out_video_path is not None:
            self.elapsed_time['visTime'].append(visTime-featTime)

    def face_detection(self, frames):
        dets = []
        for fidx, image in enumerate(frames):
            image_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes, _, probs = self.face_detector(image_input, top_k=10, prob_threshold=0.9)
            bboxes = torch.cat([bboxes, probs.reshape(-1, 1)], dim=-1)
            dets.append([])
            for bbox in bboxes:
                frame_idex = fidx * self.face_det_stride
                dets[-1].append({'frame':frame_idex, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
        return dets

    def bb_intersection_over_union(self, boxA, boxB, evalCol=False):
        # IOU Function to calculate overlap between two image
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        if evalCol == True:
            iou = interArea / float(boxAArea)
        else:
            iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def track_shot(self, scene_faces):
        # Face tracking
        tracks = []
        while True:   # continuously search for consecutive faces.
            track = []
            for frame_faces in scene_faces:
                for face in frame_faces:
                    if track == []:
                        track.append(face)
                        frame_faces.remove(face)
                        break
                    elif face['frame'] - track[-1]['frame'] <= self.num_failed_det:  # the face does not interrupt for 'num_failed_det' frame.
                        iou = self.bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                        # minimum IOU between consecutive face.
                        if iou > self.min_iou:
                            track.append(face)
                            frame_faces.remove(face)
                            break
                    else:
                        break
            if track == []:
                break
            elif len(track) > 1 and track[-1]['frame'] - track[0]['frame'] + 1 >= self.min_track:
                frame_num = np.array([ f['frame'] for f in track ])
                bboxes = np.array([np.array(f['bbox']) for f in track])
                frameI = np.arange(frame_num[0], frame_num[-1]+1)
                bboxesI = []
                for ij in range(0, 4):
                    interpfn  = interp1d(frame_num, bboxes[:,ij]) # missing boxes can be filled by interpolation.
                    bboxesI.append(interpfn(frameI))
                bboxesI  = np.stack(bboxesI, axis=1)
                if max(np.mean(bboxesI[:,2]-bboxesI[:,0]), np.mean(bboxesI[:,3]-bboxesI[:,1])) > self.min_face_size:  # need face size > min_face_size
                    tracks.append({'frame':frameI,'bbox':bboxesI})
        return tracks

    def crop_video(self, track, frames, audio):
        # crop the face clips
        crop_frames = []
        dets = {'x':[], 'y':[], 's':[]}
        for det in track['bbox']:
            dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
            dets['y'].append((det[1]+det[3])/2) # crop center x 
            dets['x'].append((det[0]+det[2])/2) # crop center y
        for fidx, frame in enumerate(track['frame']):
            cs  = self.crop_scale
            bs  = dets['s'][fidx]   # detection box size
            bsi = int(bs * (1 + 2 * cs))  # pad videos by this amount 
            image = frames[frame]
            frame = np.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
            my  = dets['y'][fidx] + bsi  # BBox center Y
            mx  = dets['x'][fidx] + bsi  # BBox center X
            face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
            crop_frames.append(cv2.resize(face, (224, 224)))
        cropaudio = audio[track['frame'][0]*640:(track['frame'][-1]+1)*640]
        return {'track':track, 'proc_track':dets, 'data':[crop_frames, cropaudio]}

    def evaluate_asd(self, tracks):
        # active speaker detection by pretrained TalkNet
        all_scores = []
        for ins in tracks:
            video, audio = ins['data']
            audio_feature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
            video_feature = []
            for frame in video:
                face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224,224))
                face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
                video_feature.append(face)
            video_feature = np.array(video_feature)
            length = min((audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100, video_feature.shape[0] / 25)
            audio_feature = audio_feature[:int(round(length * 100)),:]
            video_feature = video_feature[:int(round(length * 25)),:,:]
            
            # Check if we need to process in batches
            audio_frames = audio_feature.shape[0]
            if audio_frames > self.asd_audio_batch_size:
                # Process in batches
                batch_scores = []
                num_batches = (audio_frames + self.asd_audio_batch_size - 1) // self.asd_audio_batch_size
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * self.asd_audio_batch_size
                    end_idx = min((batch_idx + 1) * self.asd_audio_batch_size, audio_frames)
                    
                    # Calculate corresponding video indices (audio:video ratio is 100:25 or 4:1)
                    video_start_idx = start_idx // 4
                    video_end_idx = end_idx // 4
                    
                    # Extract batch features
                    audio_batch = audio_feature[start_idx:end_idx, :]
                    video_batch = video_feature[video_start_idx:video_end_idx, :, :]
                    
                    # Prepare batch for model
                    audio_batch = np.expand_dims(audio_batch, axis=0).astype(np.float32)
                    video_batch = np.expand_dims(video_batch, axis=0).astype(np.float32)
                    
                    # Get scores for this batch
                    batch_score = self.speaker_detector(audio_batch, video_batch)
                    batch_scores.append(batch_score)
                
                # Concatenate all batch scores
                score = np.concatenate(batch_scores, axis=0)
            else:
                # Process normally if within batch size
                audio_feature = np.expand_dims(audio_feature, axis=0).astype(np.float32)
                video_feature = np.expand_dims(video_feature, axis=0).astype(np.float32)
                score = self.speaker_detector(audio_feature, video_feature)
            
            all_score = np.round(score, 1).astype(float)
            all_scores.append(all_score)	
        return all_scores

    def evaluate_fr(self, frames, tracks, scores):
        # extract high-quality facial embeddings 
        faces = [[] for i in range(len(frames))]
        for tidx, track in enumerate(tracks):
            score = scores[tidx]
            for fidx, frame in enumerate(track['track']['frame'].tolist()):
                s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
                s = np.mean(s)
                bbox = track['track']['bbox'][fidx]
                face = frames[frame][max(int(bbox[1]), 0):min(int(bbox[3]), frames[frame].shape[0]), max(int(bbox[0]), 0):min(int(bbox[2]), frames[frame].shape[1])]
                faces[frame].append({'track':tidx, 'score':float(s), 'facedata':face})

        active_facial_embs={'frameI':np.empty((0,), dtype=int), 'feat':np.empty((0, 512), dtype=np.float32)}
        for fidx in range(len(faces)):
            if fidx % self.face_det_stride != 0:
                continue
            active_face = None
            active_face_num = 0
            for face in faces[fidx]:
                if face['score'] > 0:
                    active_face = face['facedata']
                    active_face_num += 1
            # process frames containing only one active face.
            if active_face_num == 1:
                # quality assessment
                if not self.disable_face_quality_check:
                    face_quality_score = self.face_quality_evaluator(active_face)
                    if face_quality_score < 0.7:
                        continue
                feature = self.face_embs_extractor(active_face)
                active_facial_embs['frameI'] = np.append(active_facial_embs['frameI'], fidx)
                active_facial_embs['feat'] = np.append(active_facial_embs['feat'], feature, axis=0)
        return active_facial_embs

    def visualization(self, frames, tracks, scores, frame_offset=0):
        faces = [[] for i in range(len(frames))]
        for tidx, track in enumerate(tracks):
            score = scores[tidx]
            for fidx, frame in enumerate(track['track']['frame'].tolist()):
                s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]
                s = np.mean(s)
                faces[frame].append({'track':tidx, 'score':float(s),'bbox':track['track']['bbox'][fidx]})

        # Store face detection results for later visualization
        for fidx, face_list in enumerate(faces):
            if face_list:  # Only store if there are faces
                global_frame_idx = fidx + frame_offset
                self.vad_face_results[global_frame_idx] = face_list
    
    def draw_face_on_frame(self, frame, face_info):
        """Draw face detection results on a frame"""
        colorDict = {0: 0, 1: 255}
        clr = colorDict[int((face_info['score'] >= 0))]
        txt = round(face_info['score'], 1)
        cv2.rectangle(frame, 
                     (int(face_info['bbox'][0]), int(face_info['bbox'][1])), 
                     (int(face_info['bbox'][2]), int(face_info['bbox'][3])),
                     (0, clr, 255-clr), 10)
        cv2.putText(frame, '%s' % (txt), 
                   (int(face_info['bbox'][0]), int(face_info['bbox'][1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, clr, 255-clr), 5)
    
    def save_face_tracks(self):
        """Save individual face tracks as MP4 files with audio"""
        # Filter tracks based on average score threshold
        filtered_tracks = []
        for track_data in self.all_tracks_data:
            avg_score = float(np.mean(track_data['scores']))
            if avg_score >= self.min_avg_score_threshold:
                filtered_tracks.append(track_data)
        
        print(f"Filtered {len(filtered_tracks)} face tracks (out of {len(self.all_tracks_data)} total) with avg score >= {self.min_avg_score_threshold}")
        
        # Save tracks without merging for now - merging will be done after speaker clustering
        track_metadata = []
        
        for track_idx, track_data in enumerate(filtered_tracks):
            # Extract track information
            track_info = track_data['track_info']
            video_frames = track_data['video_frames']
            audio_segment = track_data['audio']
            scores = track_data['scores']
            frame_start = track_data['frame_start']
            original_track_idx = track_data['track_idx']  # Use original track index
            
            # Create filename
            start_frame = track_info['frame'][0] + frame_start
            end_frame = track_info['frame'][-1] + frame_start
            track_filename = f"track_{original_track_idx:04d}_frames_{start_frame:06d}-{end_frame:06d}.mp4"
            track_path = os.path.join(self.face_track_dir, track_filename)
            
            # Create temporary video file
            temp_video_path = track_path.replace('.mp4', '_temp.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_height, frame_width = video_frames[0].shape[:2]
            video_writer = cv2.VideoWriter(temp_video_path, fourcc, self.track_video_fps, 
                                          (frame_width, frame_height))
            
            # Write frames with optional score overlay
            for frame_idx, (frame, score_val) in enumerate(zip(video_frames, scores)):
                if self.show_scores_on_frames:
                    frame_copy = frame.copy()
                    # Add score text overlay
                    cv2.putText(frame_copy, f'Score: {float(score_val):.2f}', 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    video_writer.write(frame_copy)
                else:
                    video_writer.write(frame)
            
            video_writer.release()
            
            # Save audio and combine with video using ffmpeg
            if self.include_audio_in_tracks and len(audio_segment) > 0:
                # Save temporary audio file
                temp_audio_path = track_path.replace('.mp4', '_temp.wav')
                wavfile.write(temp_audio_path, 16000, audio_segment)
                
                # Combine video and audio using ffmpeg
                cmd = [
                    'ffmpeg', '-y', '-i', temp_video_path, '-i', temp_audio_path,
                    '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental',
                    '-shortest', track_path
                ]
                subprocess.run(cmd, capture_output=True, text=True)
                
                # Clean up temporary files
                os.remove(temp_video_path)
                os.remove(temp_audio_path)
            else:
                # Just rename the video file
                os.rename(temp_video_path, track_path)
            
            # Save metadata
            if self.save_track_metadata:
                metadata = {
                    'track_idx': original_track_idx,  # Use original track index
                    'video_id': self.video_id,
                    'start_frame': int(start_frame),
                    'end_frame': int(end_frame),
                    'start_time': float(start_frame) / self.fps,
                    'end_time': float(end_frame) / self.fps,
                    'duration': float(end_frame - start_frame) / self.fps,
                    'num_frames': len(video_frames),
                    'avg_score': float(np.mean(scores)),
                    'max_score': float(np.max(scores)),
                    'min_score': float(np.min(scores)),
                    'bbox_info': {
                        'avg_width': float(np.mean(track_info['bbox'][:, 2] - track_info['bbox'][:, 0])),
                        'avg_height': float(np.mean(track_info['bbox'][:, 3] - track_info['bbox'][:, 1]))
                    }
                }
                track_metadata.append(metadata)
        
        # Save all metadata to a JSON file
        if self.save_track_metadata and track_metadata:
            metadata_path = os.path.join(self.face_track_dir, f"{self.video_id}_tracks_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(track_metadata, f, indent=2)
            print(f"Track metadata saved to {metadata_path}")
