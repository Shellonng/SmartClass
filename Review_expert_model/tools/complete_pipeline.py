# -*- coding: utf-8 -*-
"""
完整Pipeline - 集成情绪检测和音频转录
使用FER替代DeepFace，添加音频处理功能
"""
import cv2
import numpy as np
import mediapipe as mp
from fer import FER
import pandas as pd
import os
from tqdm import tqdm
import warnings
import sys
import io
import subprocess
import tempfile
warnings.filterwarnings('ignore')

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class CompletePipeline:
    def __init__(self):
        print("[*] Loading models...")
        
        # MediaPipe Pose
        print("  [+] Loading MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("  [OK] MediaPipe Pose loaded")
        
        # MediaPipe Face Mesh
        print("  [+] Loading MediaPipe Face Mesh...")
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("  [OK] MediaPipe Face Mesh loaded")
        
        # FER emotion detector
        print("  [+] Loading FER emotion detector...")
        self.emotion_detector = FER(mtcnn=True)
        print("  [OK] FER emotion detector loaded")
        
        # Whisper (lazy load)
        self.whisper_model = None
        
        # Filler words
        self.filler_words = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'so', 'well']
        self.filler_words_zh = ['嗯', '啊', '呃', '那个', '这个', '就是', '然后', '嘛', '吧']
        
        print("\n[OK] All models loaded!\n")
        
    def load_whisper(self):
        """Lazy load Whisper model"""
        if self.whisper_model is None:
            print("  [+] Loading Whisper model...")
            from transformers import pipeline
            self.whisper_model = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-small",
                device=-1  # CPU
            )
            print("  [OK] Whisper loaded")
        return self.whisper_model
    
    def extract_audio_from_video(self, video_path):
        """Extract audio from video using ffmpeg"""
        try:
            # Create temporary wav file
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio.close()
            audio_path = temp_audio.name
            
            # Check if ffmpeg is available
            try:
                subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("    [WARN] ffmpeg not found, skipping audio extraction")
                return None
            
            # Extract audio
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite
                audio_path
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0 and os.path.exists(audio_path):
                return audio_path
            else:
                print(f"    [WARN] Audio extraction failed: {result.stderr[:100]}")
                return None
                
        except Exception as e:
            print(f"    [ERROR] Audio extraction error: {e}")
            return None
    
    def transcribe_audio(self, audio_path, start_time, end_time):
        """Transcribe audio segment using Whisper"""
        try:
            if audio_path is None:
                return ""
            
            # Load Whisper if not loaded
            whisper = self.load_whisper()
            
            # For now, transcribe entire audio (optimization: extract segment)
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            
            # Extract time segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            audio_segment = audio[start_sample:end_sample]
            
            if len(audio_segment) < 1600:  # Too short (< 0.1s)
                return ""
            
            # Transcribe
            result = whisper({
                "array": audio_segment.flatten(),
                "sampling_rate": sr
            })
            
            return result.get('text', '').strip()
            
        except Exception as e:
            print(f"    [WARN] Transcription error: {e}")
            return ""
    
    def detect_filler_words(self, text):
        """Detect filler words in transcription"""
        text_lower = text.lower()
        count = 0
        
        # English fillers
        for filler in self.filler_words:
            count += text_lower.count(filler)
        
        # Chinese fillers
        for filler in self.filler_words_zh:
            count += text.count(filler)
        
        return count
    
    def process_video(self, video_path, question, sample_id):
        """Process one video with full pipeline"""
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
        
        # Extract audio
        print(f"\n  [*] Extracting audio...")
        audio_path = self.extract_audio_from_video(video_path)
        if audio_path:
            print(f"  [OK] Audio extracted to: {audio_path}")
        else:
            print(f"  [WARN] Audio extraction skipped")
        
        # Process windows
        window_size = 10
        annotations = []
        
        num_windows = int(duration / window_size) + (1 if duration % window_size > 0 else 0)
        print(f"\n  [INFO] Will analyze {num_windows} windows ({window_size}s each)\n")
        
        for window_idx in range(num_windows):
            start_time = window_idx * window_size
            end_time = min(start_time + window_size, int(duration))
            
            print(f"  [WINDOW {window_idx+1}/{num_windows}] {start_time}s - {end_time}s")
            
            # Extract features
            features = self.extract_window_features(
                cap, fps, start_time, end_time, 
                audio_path, question
            )
            
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
                'transcription': features.get('transcription', ''),
                'filler_count': features.get('filler_count', 0),
                'notes': features['summary']
            })
            
            print(f"     Focus: {scores['focus']:.0f}")
            print(f"     Psychological: {scores['psychological']:.0f}")
            print(f"     Language: {scores['language']:.0f}")
            print(f"     Professional: {scores['professional']:.0f}")
            if features.get('transcription'):
                print(f"     Transcription: {features['transcription'][:50]}...")
                print(f"     Filler words: {features.get('filler_count', 0)}")
            if alert_type > 0:
                print(f"     [!] Alert: {alert_text}")
            else:
                print(f"     [OK] No issues")
        
        cap.release()
        
        # Clean up temp audio file
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass
        
        print(f"\n  [DONE] Video processed! Generated {len(annotations)} annotations\n")
        return annotations
    
    def extract_window_features(self, cap, fps, start_time, end_time, audio_path, question):
        """Extract features from time window"""
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        emotions = []
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
            
            # Emotion detection with FER
            try:
                emotion_result = self.emotion_detector.detect_emotions(frame)
                if emotion_result and len(emotion_result) > 0:
                    # FER returns list of faces
                    emotion_scores = emotion_result[0]['emotions']
                    emotions.append(emotion_scores)
            except Exception as e:
                pass
            
            # Gaze detection
            try:
                face_result = self.mp_face.process(frame_rgb)
                if face_result.multi_face_landmarks:
                    face_detected += 1
                    landmarks = face_result.multi_face_landmarks[0].landmark
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
                    head = pose_result.pose_landmarks.landmark[0]
                    poses.append({
                        'shoulder_y': shoulder_y,
                        'head_y': head.y,
                        'posture_score': 1.0 if shoulder_y < 0.6 else 0.5
                    })
            except Exception as e:
                pass
        
        # Transcribe audio for this window
        transcription = ""
        filler_count = 0
        
        if audio_path:
            transcription = self.transcribe_audio(audio_path, start_time, end_time)
            if transcription:
                filler_count = self.detect_filler_words(transcription)
        
        return {
            'emotions': emotions,
            'gazes': gazes,
            'poses': poses,
            'face_detected': face_detected,
            'pose_detected': pose_detected,
            'total_samples': len(sample_frames),
            'transcription': transcription,
            'filler_count': filler_count,
            'summary': self.summarize_features(
                emotions, gazes, poses, face_detected, 
                pose_detected, len(sample_frames), transcription, filler_count
            )
        }
    
    def summarize_features(self, emotions, gazes, poses, face_count, 
                          pose_count, total, transcription, filler_count):
        """Generate feature summary"""
        summary = []
        
        summary.append(f"Face:{face_count}/{total}")
        summary.append(f"Pose:{pose_count}/{total}")
        
        # Emotion summary
        if emotions:
            avg_emotions = {}
            for emotion_dict in emotions:
                for key, value in emotion_dict.items():
                    if key not in avg_emotions:
                        avg_emotions[key] = []
                    avg_emotions[key].append(value)
            
            avg_emotions = {k: np.mean(v) for k, v in avg_emotions.items()}
            dominant = max(avg_emotions, key=avg_emotions.get)
            summary.append(f"Emotion:{dominant}({avg_emotions[dominant]:.1%})")
        
        # Gaze
        if gazes:
            avg_deviation = np.mean([g['deviation'] for g in gazes])
            if avg_deviation > 0.08:
                summary.append("Gaze:Distracted")
            else:
                summary.append("Gaze:Focused")
        
        # Posture
        if poses:
            avg_posture = np.mean([p['posture_score'] for p in poses])
            if avg_posture > 0.8:
                summary.append("Posture:Good")
            else:
                summary.append("Posture:NeedsImprovement")
        
        # Transcription
        if transcription:
            word_count = len(transcription.split())
            summary.append(f"Words:{word_count}")
            if filler_count > 0:
                summary.append(f"Fillers:{filler_count}")
        
        return "; ".join(summary)
    
    def auto_score(self, features):
        """Auto scoring with emotion and transcription"""
        # Focus score (gaze + face detection)
        focus = 75
        if features['gazes']:
            avg_deviation = np.mean([g['deviation'] for g in features['gazes']])
            if avg_deviation < 0.05:
                focus += 15
            elif avg_deviation > 0.1:
                focus -= 20
        
        if features['total_samples'] > 0:
            detection_rate = features['face_detected'] / features['total_samples']
            if detection_rate < 0.5:
                focus -= 10
        
        # Psychological score (emotion + posture)
        psychological = 75
        
        if features['emotions']:
            # Average emotions across all frames
            avg_emotions = {}
            for emotion_dict in features['emotions']:
                for key, value in emotion_dict.items():
                    if key not in avg_emotions:
                        avg_emotions[key] = []
                    avg_emotions[key].append(value)
            
            avg_emotions = {k: np.mean(v) for k, v in avg_emotions.items()}
            
            # Positive emotions boost score
            positive = avg_emotions.get('happy', 0) + avg_emotions.get('neutral', 0)
            # Negative emotions decrease score
            negative = avg_emotions.get('fear', 0) + avg_emotions.get('sad', 0) + avg_emotions.get('angry', 0)
            
            if positive > 0.7:
                psychological += 15
            elif negative > 0.5:
                psychological -= 20
        
        if features['poses']:
            posture_scores = [p['posture_score'] for p in features['poses']]
            avg_posture = np.mean(posture_scores)
            if avg_posture > 0.8:
                psychological += 10
            elif avg_posture < 0.5:
                psychological -= 10
        
        # Language score (transcription + filler words)
        language = 70
        
        if features['transcription']:
            word_count = len(features['transcription'].split())
            
            # Speech rate (assuming 10 second window)
            speech_rate = word_count / 10
            
            # Optimal speech rate: 2-4 words/sec
            if 2 <= speech_rate <= 4:
                language += 15
            elif speech_rate > 6:
                language -= 10  # Too fast
            elif speech_rate < 1:
                language -= 10  # Too slow
            
            # Filler word penalty
            filler_ratio = features['filler_count'] / max(word_count, 1)
            if filler_ratio > 0.2:
                language -= 20
            elif filler_ratio > 0.1:
                language -= 10
        
        # Professional (default, needs QA alignment)
        professional = 65
        
        return {
            'focus': max(0, min(100, focus)),
            'psychological': max(0, min(100, psychological)),
            'language': max(0, min(100, language)),
            'professional': professional
        }
    
    def detect_alert(self, features):
        """Detect anomalies and generate alerts"""
        # Gaze issue
        if features['gazes']:
            avg_deviation = np.mean([g['deviation'] for g in features['gazes']])
            if avg_deviation > 0.1:
                return 1, "Please maintain eye contact"
        
        # Low detection rate
        if features['total_samples'] > 0:
            detection_rate = features['face_detected'] / features['total_samples']
            if detection_rate < 0.3:
                return 1, "Please stay in frame"
        
        # Negative emotion
        if features['emotions']:
            avg_emotions = {}
            for emotion_dict in features['emotions']:
                for key, value in emotion_dict.items():
                    if key not in avg_emotions:
                        avg_emotions[key] = []
                    avg_emotions[key].append(value)
            
            avg_emotions = {k: np.mean(v) for k, v in avg_emotions.items()}
            
            fear_level = avg_emotions.get('fear', 0)
            if fear_level > 0.4:
                return 2, "Take a deep breath, relax"
        
        # Excessive filler words
        if features['filler_count'] > 5:
            return 3, "Try to reduce filler words"
        
        # Posture issue
        if features['poses']:
            avg_posture = np.mean([p['posture_score'] for p in features['poses']])
            if avg_posture < 0.5:
                return 4, "Adjust your posture"
        
        return 0, ""


def main():
    """Run complete pipeline test"""
    print("\n" + "="*60)
    print("     Complete Interview Pipeline Test")
    print("     (Emotion Detection + Audio Transcription)")
    print("="*60 + "\n")
    
    # Initialize
    pipeline = CompletePipeline()
    
    # testv folder
    video_dir = "./testv"
    if not os.path.exists(video_dir):
        print(f"[ERROR] Directory not found: {video_dir}")
        return
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_files.sort()
    
    # Limit to first video for testing (can process all later)
    video_files = video_files[:1]  # Process only first video for speed
    
    print(f"[INFO] Found {len(video_files)} test video(s):\n")
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
        sample_id = f"complete_{i:03d}"
        question = questions.get(video_file, "General interview question")
        
        annotations = pipeline.process_video(video_path, question, sample_id)
        all_annotations.extend(annotations)
    
    # Save results
    if all_annotations:
        df = pd.DataFrame(all_annotations)
        output_path = "complete_annotations.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*60)
        print("[SUCCESS] Complete pipeline test finished!")
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
        
        # Show transcriptions
        trans_count = df['transcription'].apply(lambda x: len(x) > 0).sum()
        print(f"\n[TRANSCRIPTIONS] {trans_count}/{len(all_annotations)} windows transcribed")
        
        if trans_count > 0:
            print(f"\nSample transcriptions:")
            for idx, row in df[df['transcription'].str.len() > 0].head(3).iterrows():
                print(f"   [{row['start_time']}-{row['end_time']}s]: {row['transcription'][:80]}...")
        
        print(f"\n[IMPROVEMENTS]")
        print(f"   1. [OK] FER emotion detection working")
        print(f"   2. [OK] Audio transcription integrated")
        print(f"   3. [OK] Filler word detection implemented")
        print(f"   4. [TODO] Implement Transformer model for better scoring")
        print(f"   5. [TODO] Add Question-Answer alignment\n")
    else:
        print("\n[ERROR] No annotations generated")


if __name__ == "__main__":
    main()

