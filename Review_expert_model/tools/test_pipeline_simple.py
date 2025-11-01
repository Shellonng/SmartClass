# -*- coding: utf-8 -*-
"""
Simplified Pipeline Test - MediaPipe only
Skips DeepFace to avoid dependency conflicts
"""
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os
from tqdm import tqdm
import warnings
import sys
import io
warnings.filterwarnings('ignore')

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class SimplePipelineTest:
    def __init__(self):
        print("[*] Loading models...")
        
        print("  [+] Loading MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("  [OK] MediaPipe Pose loaded")
        
        print("  [+] Loading MediaPipe Face Mesh...")
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("  [OK] MediaPipe Face Mesh loaded")
        
        print("\n[OK] All models loaded!\n")
        
    def process_video(self, video_path, question, sample_id):
        """Process one video"""
        print(f"\n{'='*60}")
        print(f"[VIDEO] Processing: {os.path.basename(video_path)}")
        print(f"[QUESTION] {question}")
        print(f"{'='*60}\n")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"  [INFO] Video Info:")
        print(f"     Total frames: {total_frames}")
        print(f"     FPS: {fps:.2f}")
        print(f"     Duration: {duration:.2f}s")
        
        # 10-second windows
        window_size = 10
        annotations = []
        
        num_windows = int(duration / window_size) + (1 if duration % window_size > 0 else 0)
        print(f"\n  [INFO] Will analyze {num_windows} windows ({window_size}s each)\n")
        
        for window_idx in range(num_windows):
            start_time = window_idx * window_size
            end_time = min(start_time + window_size, int(duration))
            
            print(f"  [WINDOW {window_idx+1}/{num_windows}] {start_time}s - {end_time}s")
            
            # Extract features
            features = self.extract_window_features(cap, fps, start_time, end_time)
            
            # Auto score
            scores = self.auto_score(features)
            
            # Detect alerts
            alert_type, alert_text = self.detect_alert(features)
            
            annotations.append({
                'sample_id': sample_id,
                'video_path': video_path,
                'question': question,
                'start_time': start_time,
                'end_time': end_time,
                'focus_score': scores['focus'],
                'psychological_score': scores['psychological'],
                'language_score': scores['language'],
                'professional_score': scores['professional'],
                'alert_type': alert_type,
                'alert_text': alert_text,
                'notes': features['summary']
            })
            
            print(f"     Focus: {scores['focus']:.0f}")
            print(f"     Psychological: {scores['psychological']:.0f}")
            print(f"     Language: {scores['language']:.0f}")
            print(f"     Professional: {scores['professional']:.0f}")
            if alert_type > 0:
                print(f"     [!] Alert: {alert_text}")
            else:
                print(f"     [OK] No issues")
        
        cap.release()
        print(f"\n  [DONE] Video processed! Generated {len(annotations)} annotations\n")
        return annotations
    
    def extract_window_features(self, cap, fps, start_time, end_time):
        """Extract features from time window"""
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        gazes = []
        poses = []
        face_detected = 0
        pose_detected = 0
        
        # Sample every 2 seconds
        sample_interval = max(1, int(fps * 2))
        sample_frames = list(range(start_frame, end_frame, sample_interval))
        
        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Gaze detection
            try:
                face_result = self.mp_face.process(frame_rgb)
                if face_result.multi_face_landmarks:
                    face_detected += 1
                    landmarks = face_result.multi_face_landmarks[0].landmark
                    # iris landmark 468
                    left_iris = landmarks[468]
                    gaze_x = left_iris.x
                    gaze_deviation = abs(gaze_x - 0.5)
                    gazes.append({'x': gaze_x, 'deviation': gaze_deviation})
            except Exception as e:
                pass
            
            # Pose detection
            try:
                pose_result = self.mp_pose.process(frame_rgb)
                if pose_result.pose_landmarks:
                    pose_detected += 1
                    shoulder_l = pose_result.pose_landmarks.landmark[11]
                    shoulder_r = pose_result.pose_landmarks.landmark[12]
                    shoulder_y = (shoulder_l.y + shoulder_r.y) / 2
                    head = pose_result.pose_landmarks.landmark[0]  # nose
                    poses.append({
                        'shoulder_y': shoulder_y,
                        'head_y': head.y,
                        'posture_score': 1.0 if shoulder_y < 0.6 else 0.5
                    })
            except Exception as e:
                pass
        
        return {
            'gazes': gazes,
            'poses': poses,
            'face_detected': face_detected,
            'pose_detected': pose_detected,
            'total_samples': len(sample_frames),
            'summary': self.summarize_features(gazes, poses, face_detected, pose_detected, len(sample_frames))
        }
    
    def summarize_features(self, gazes, poses, face_count, pose_count, total):
        """Generate feature summary"""
        summary = []
        
        summary.append(f"Face:{face_count}/{total}")
        summary.append(f"Pose:{pose_count}/{total}")
        
        if gazes:
            avg_deviation = np.mean([g['deviation'] for g in gazes])
            if avg_deviation > 0.08:
                summary.append("Gaze:Distracted")
            else:
                summary.append("Gaze:Focused")
        
        if poses:
            avg_posture = np.mean([p['posture_score'] for p in poses])
            if avg_posture > 0.8:
                summary.append("Posture:Good")
            else:
                summary.append("Posture:NeedsImprovement")
        
        return "; ".join(summary)
    
    def auto_score(self, features):
        """Auto scoring"""
        # Focus score (based on gaze)
        focus = 75
        if features['gazes']:
            avg_deviation = np.mean([g['deviation'] for g in features['gazes']])
            if avg_deviation < 0.05:
                focus += 15
            elif avg_deviation > 0.1:
                focus -= 20
        
        # Detection rate affects focus
        if features['total_samples'] > 0:
            detection_rate = features['face_detected'] / features['total_samples']
            if detection_rate < 0.5:
                focus -= 10
        
        # Psychological score (based on posture stability)
        psychological = 75
        if features['poses']:
            posture_scores = [p['posture_score'] for p in features['poses']]
            avg_posture = np.mean(posture_scores)
            if avg_posture > 0.8:
                psychological += 10
            elif avg_posture < 0.5:
                psychological -= 10
        
        # Language (default, needs audio)
        language = 70
        
        # Professional (default, needs QA alignment)
        professional = 65
        
        return {
            'focus': max(0, min(100, focus)),
            'psychological': max(0, min(100, psychological)),
            'language': language,
            'professional': professional
        }
    
    def detect_alert(self, features):
        """Detect anomalies"""
        # Gaze issue
        if features['gazes']:
            avg_deviation = np.mean([g['deviation'] for g in features['gazes']])
            if avg_deviation > 0.1:
                return 1, "Please keep eye contact"
        
        # Low detection (not in frame)
        if features['total_samples'] > 0:
            detection_rate = features['face_detected'] / features['total_samples']
            if detection_rate < 0.3:
                return 1, "Please stay in frame"
        
        # Posture issue
        if features['poses']:
            avg_posture = np.mean([p['posture_score'] for p in features['poses']])
            if avg_posture < 0.5:
                return 4, "Adjust posture"
        
        return 0, ""


def main():
    """Run test"""
    print("\n" + "="*60)
    print("     Interview Pipeline Test (Simplified)")
    print("="*60 + "\n")
    
    # Initialize
    tester = SimplePipelineTest()
    
    # testv folder
    video_dir = "./testv"
    if not os.path.exists(video_dir):
        print(f"[ERROR] Directory not found: {video_dir}")
        return
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_files.sort()
    
    print(f"[INFO] Found {len(video_files)} test videos:\n")
    for i, vf in enumerate(video_files, 1):
        size_mb = os.path.getsize(os.path.join(video_dir, vf)) / (1024*1024)
        print(f"   {i}. {vf} ({size_mb:.1f} MB)")
    print()
    
    # Mock questions
    questions = {
        't1-1.mp4': "Please introduce yourself briefly",
        't2-1.mp4': "Tell me about your recent project experience",
        't3-1.mp4': "What's your understanding of Python?"
    }
    
    all_annotations = []
    
    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(video_dir, video_file)
        sample_id = f"test_{i:03d}"
        question = questions.get(video_file, "General interview question")
        
        annotations = tester.process_video(video_path, question, sample_id)
        all_annotations.extend(annotations)
    
    # Save results
    if all_annotations:
        df = pd.DataFrame(all_annotations)
        output_path = "test_annotations.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*60)
        print("[SUCCESS] Pipeline test completed!")
        print("="*60)
        print(f"\n[STATS]")
        print(f"   Videos processed: {len(video_files)}")
        print(f"   Annotations generated: {len(all_annotations)}")
        print(f"   Saved to: {output_path}")
        
        print(f"\n[SCORES Summary]")
        print(f"   Focus: {df['focus_score'].mean():.1f} +/- {df['focus_score'].std():.1f}")
        print(f"   Psychological: {df['psychological_score'].mean():.1f} +/- {df['psychological_score'].std():.1f}")
        print(f"   Language: {df['language_score'].mean():.1f} +/- {df['language_score'].std():.1f}")
        print(f"   Professional: {df['professional_score'].mean():.1f} +/- {df['professional_score'].std():.1f}")
        
        alert_count = (df['alert_type'] > 0).sum()
        print(f"\n[ALERTS] Detected: {alert_count}/{len(all_annotations)} windows")
        
        if alert_count > 0:
            print(f"\nAlert details:")
            alert_df = df[df['alert_type'] > 0]
            for _, row in alert_df.iterrows():
                print(f"   - [{row['sample_id']}] {row['start_time']}-{row['end_time']}s: {row['alert_text']}")
        
        print(f"\n[NEXT STEPS]")
        print(f"   1. [OK] Pipeline basic validation complete")
        print(f"   2. [WARN] DeepFace skipped due to dependencies")
        print(f"   3. [WARN] Whisper transcription needs separate audio processing")
        print(f"   4. [TODO] Check {output_path} for detailed annotations")
        print(f"   5. [TODO] Prepare full Transformer model implementation\n")
    else:
        print("\n[ERROR] No annotations generated")


if __name__ == "__main__":
    main()
