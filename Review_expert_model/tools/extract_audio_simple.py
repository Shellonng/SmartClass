# -*- coding: utf-8 -*-
"""
简化版音频特征提取 - 只提取梅尔频谱，不使用Whisper
环境: interview_audio
"""
import cv2
import numpy as np
import os
import argparse
import json
import soundfile as sf
import librosa

class SimpleAudioExtractor:
    def __init__(self, output_dir='./features/audio'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print("[Simple Audio Extractor] Initialized")
    
    def extract_audio_from_video(self, video_path):
        """从视频提取音频（使用moviepy，避免ffmpeg）"""
        try:
            from moviepy.editor import VideoFileClip
            import tempfile
            
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_audio.name
            temp_audio.close()
            
            video = VideoFileClip(video_path)
            if video.audio:
                video.audio.write_audiofile(
                    temp_path, 
                    fps=16000, 
                    nbytes=2, 
                    codec='pcm_s16le',
                    verbose=False, 
                    logger=None
                )
                video.close()
                return temp_path
            else:
                print("  [WARN] No audio track in video")
                video.close()
                return None
                
        except Exception as e:
            print(f"  [ERROR] Audio extraction failed: {e}")
            return None
    
    def extract_from_video(self, video_path, sample_id):
        """提取音频特征"""
        print(f"\n[Processing] {os.path.basename(video_path)}")
        
        # 提取音频
        print("  [*] Extracting audio...")
        audio_path = self.extract_audio_from_video(video_path)
        
        if not audio_path or not os.path.exists(audio_path):
            print("  [ERROR] Audio extraction failed")
            return None
        
        print(f"  [OK] Audio extracted")
        
        # 获取视频时长
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        
        print(f"  Duration: {duration:.2f}s")
        
        # 读取音频
        audio, sr = sf.read(audio_path)
        
        # 10秒窗口
        window_size = 10
        num_windows = int(duration / window_size) + (1 if duration % window_size > 0 else 0)
        
        all_features = []
        
        for window_idx in range(num_windows):
            start_time = window_idx * window_size
            end_time = min(start_time + window_size, duration)
            
            print(f"  Window {window_idx+1}/{num_windows}: {start_time}s-{end_time}s", end='')
            
            # 提取特征
            features = self.extract_window_features(audio, sr, start_time, end_time)
            
            if features is not None:
                features['sample_id'] = sample_id
                features['window_idx'] = window_idx
                features['start_time'] = start_time
                features['end_time'] = end_time
                all_features.append(features)
                print(f" -> Mel shape: {features['mel_spectrogram'].shape}")
            else:
                print(" -> Failed")
        
        # 清理临时文件
        try:
            os.remove(audio_path)
        except:
            pass
        
        # 保存特征
        if all_features:
            output_file = os.path.join(self.output_dir, f"{sample_id}_audio.npz")
            
            mel_specs = []
            metadata = []
            
            for feat in all_features:
                mel_specs.append(feat['mel_spectrogram'])
                metadata.append({
                    'sample_id': feat['sample_id'],
                    'window_idx': feat['window_idx'],
                    'start_time': feat['start_time'],
                    'end_time': feat['end_time'],
                    'transcription': '',  # 空，因为没有Whisper
                    'word_count': 0,
                    'filler_count': 0,
                    'speech_rate': 0
                })
            
            np.savez_compressed(
                output_file,
                mel_spectrograms=np.array(mel_specs),
                metadata=metadata
            )
            
            print(f"\n[OK] Saved to {output_file}")
            return output_file
        
        return None
    
    def extract_window_features(self, audio, sr, start_time, end_time):
        """提取梅尔频谱（无转录）"""
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        audio_segment = audio[start_sample:end_sample]
        
        if len(audio_segment) < 1600:
            return None
        
        try:
            # 提取梅尔频谱
            mel_spec = librosa.feature.melspectrogram(
                y=audio_segment.flatten() if len(audio_segment.shape) > 1 else audio_segment,
                sr=sr,
                n_mels=80,
                n_fft=400,
                hop_length=160
            )
            
            # 归一化并平均
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_features = np.mean(mel_spec_db, axis=1)  # [80,]
            
            return {
                'mel_spectrogram': mel_features.astype(np.float32)
            }
            
        except Exception as e:
            print(f"\n    [ERROR] Mel extraction failed: {e}")
            return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default='./testv')
    parser.add_argument('--output_dir', type=str, default='./features/audio')
    parser.add_argument('--sample_prefix', type=str, default='sample')
    
    args = parser.parse_args()
    
    print("="*60)
    print("  Simple Audio Feature Extraction (No Whisper)")
    print("="*60)
    print(f"Video directory: {args.video_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*60 + "\n")
    
    extractor = SimpleAudioExtractor(output_dir=args.output_dir)
    
    if not os.path.exists(args.video_dir):
        print(f"[ERROR] Video directory not found: {args.video_dir}")
        return
    
    video_files = sorted([f for f in os.listdir(args.video_dir) if f.endswith('.mp4')])
    
    if len(video_files) == 0:
        print(f"[ERROR] No video files found")
        return
    
    print(f"[INFO] Found {len(video_files)} videos\n")
    
    results = []
    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(args.video_dir, video_file)
        sample_id = f"{args.sample_prefix}_{i:03d}"
        
        output_file = extractor.extract_from_video(video_path, sample_id)
        
        if output_file:
            results.append({
                'video_file': video_file,
                'sample_id': sample_id,
                'feature_file': output_file
            })
    
    if results:
        index_file = os.path.join(args.output_dir, 'audio_index.json')
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"[SUCCESS] Extraction completed!")
        print(f"  Processed: {len(results)}/{len(video_files)} videos")
        print(f"  Index saved to: {index_file}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

