# -*- coding: utf-8 -*-
"""
音频特征提取脚本  
环境: interview_audio
使用方法:
    conda activate interview_audio
    python extract_audio_features.py --video_dir ./testv --output_dir ./features/audio
"""
import cv2
import numpy as np
import pandas as pd
import os
import argparse
import json
import subprocess
import tempfile
import soundfile as sf
import librosa
from tqdm import tqdm

class AudioFeatureExtractor:
    def __init__(self, output_dir='./features/audio'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("[Audio Extractor] Loading Whisper model...")
        try:
            # 优先使用faster-whisper（更快，无需PyTorch）
            from faster_whisper import WhisperModel
            self.whisper = WhisperModel("small", device="cpu", compute_type="int8")
            self.whisper_type = "faster"
            print("[Audio Extractor] Faster-Whisper loaded")
        except Exception as e:
            print(f"  [WARN] Faster-whisper failed: {e}")
            # 备用：使用transformers pipeline
            from transformers import pipeline
            self.whisper = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-small",
                device=-1
            )
            self.whisper_type = "transformers"
            print("[Audio Extractor] Transformers Whisper loaded")
        
        # 填充词
        self.filler_words_en = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'so', 'well', 'actually']
        self.filler_words_zh = ['嗯', '啊', '呃', '那个', '这个', '就是', '然后', '嘛', '吧', '额']
    
    def extract_audio_from_video(self, video_path):
        """使用ffmpeg提取音频"""
        try:
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio.close()
            audio_path = temp_audio.name
            
            # 检查ffmpeg
            try:
                subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, check=True)
            except:
                print("  [WARN] ffmpeg not found, trying moviepy...")
                # 备用方案：使用moviepy
                from moviepy.editor import VideoFileClip
                video = VideoFileClip(video_path)
                video.audio.write_audiofile(audio_path, fps=16000, nbytes=2, codec='pcm_s16le', verbose=False, logger=None)
                video.close()
                return audio_path
            
            # ffmpeg提取
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                '-y', audio_path,
                '-loglevel', 'error'
            ]
            
            subprocess.run(cmd, check=True)
            
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                return audio_path
            else:
                return None
                
        except Exception as e:
            print(f"  [ERROR] Audio extraction failed: {e}")
            return None
    
    def extract_from_video(self, video_path, sample_id):
        """从视频中提取音频特征"""
        print(f"\n[Processing] {os.path.basename(video_path)}")
        
        # 提取音频
        print("  [*] Extracting audio...")
        audio_path = self.extract_audio_from_video(video_path)
        
        if not audio_path:
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
        
        # 10秒窗口
        window_size = 10
        num_windows = int(duration / window_size) + (1 if duration % window_size > 0 else 0)
        
        # 读取音频
        audio, sr = sf.read(audio_path)
        
        all_features = []
        
        for window_idx in range(num_windows):
            start_time = window_idx * window_size
            end_time = min(start_time + window_size, duration)
            
            print(f"  Window {window_idx+1}/{num_windows}: {start_time}s-{end_time}s", end='')
            
            # 提取特征
            features = self.extract_window_features(
                audio, sr, start_time, end_time
            )
            
            if features is not None:
                features['sample_id'] = sample_id
                features['window_idx'] = window_idx
                features['start_time'] = start_time
                features['end_time'] = end_time
                all_features.append(features)
                print(f" -> {len(features['transcription'])} chars, {features['filler_count']} fillers")
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
            
            # 准备数据
            mel_specs = []
            metadata = []
            
            for feat in all_features:
                mel_specs.append(feat['mel_spectrogram'])
                metadata.append({
                    'sample_id': feat['sample_id'],
                    'window_idx': feat['window_idx'],
                    'start_time': feat['start_time'],
                    'end_time': feat['end_time'],
                    'transcription': feat['transcription'],
                    'word_count': feat['word_count'],
                    'filler_count': feat['filler_count'],
                    'speech_rate': feat['speech_rate']
                })
            
            np.savez_compressed(
                output_file,
                mel_spectrograms=np.array(mel_specs, dtype=object),
                metadata=metadata
            )
            
            print(f"\n[OK] Saved to {output_file}")
            return output_file
        
        return None
    
    def extract_window_features(self, audio, sr, start_time, end_time):
        """提取单个时间窗口的音频特征"""
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # 提取音频片段
        audio_segment = audio[start_sample:end_sample]
        
        if len(audio_segment) < 1600:  # 太短
            return None
        
        try:
            # 1. Whisper转录
            if self.whisper_type == "faster":
                # faster-whisper需要保存临时文件
                import tempfile
                temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_path = temp_audio.name
                temp_audio.close()
                
                sf.write(temp_path, audio_segment, sr)
                
                # 转录
                segments, info = self.whisper.transcribe(temp_path, language="zh")
                transcription = " ".join([segment.text for segment in segments]).strip()
                
                # 清理临时文件
                try:
                    os.remove(temp_path)
                except:
                    pass
            else:
                # transformers pipeline
                result = self.whisper({
                    "array": audio_segment.flatten() if len(audio_segment.shape) > 1 else audio_segment,
                    "sampling_rate": sr
                })
                transcription = result.get('text', '').strip()
            
            # 2. 统计词数
            words = transcription.split()
            word_count = len(words)
            
            # 3. 检测填充词
            filler_count = 0
            text_lower = transcription.lower()
            
            for filler in self.filler_words_en:
                filler_count += text_lower.count(filler.lower())
            
            for filler in self.filler_words_zh:
                filler_count += transcription.count(filler)
            
            # 4. 计算语速
            duration = end_time - start_time
            speech_rate = word_count / duration if duration > 0 else 0
            
            # 5. 提取梅尔频谱
            mel_spec = librosa.feature.melspectrogram(
                y=audio_segment,
                sr=sr,
                n_mels=80,
                n_fft=400,
                hop_length=160
            )
            
            # 归一化并取平均（跨时间维度）
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_features = np.mean(mel_spec_db, axis=1)  # [80,]
            
            return {
                'mel_spectrogram': mel_features.astype(np.float32),
                'transcription': transcription,
                'word_count': word_count,
                'filler_count': filler_count,
                'speech_rate': speech_rate
            }
            
        except Exception as e:
            print(f"\n    [ERROR] Feature extraction failed: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Extract audio features from videos')
    parser.add_argument('--video_dir', type=str, default='./testv',
                       help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, default='./features/audio',
                       help='Output directory for features')
    parser.add_argument('--sample_prefix', type=str, default='sample',
                       help='Prefix for sample IDs')
    
    args = parser.parse_args()
    
    print("="*60)
    print("  Audio Feature Extraction (Whisper)")
    print("="*60)
    print(f"Video directory: {args.video_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*60 + "\n")
    
    # 初始化提取器
    extractor = AudioFeatureExtractor(output_dir=args.output_dir)
    
    # 获取视频文件
    if not os.path.exists(args.video_dir):
        print(f"[ERROR] Video directory not found: {args.video_dir}")
        return
    
    video_files = sorted([f for f in os.listdir(args.video_dir) if f.endswith('.mp4')])
    
    if len(video_files) == 0:
        print(f"[ERROR] No video files found in {args.video_dir}")
        return
    
    print(f"[INFO] Found {len(video_files)} videos\n")
    
    # 处理每个视频
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
    
    # 保存索引
    if results:
        index_file = os.path.join(args.output_dir, 'audio_index.json')
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"[SUCCESS] Extraction completed!")
        print(f"  Processed: {len(results)}/{len(video_files)} videos")
        print(f"  Index saved to: {index_file}")
        print(f"{'='*60}\n")
    else:
        print("\n[ERROR] No features extracted")


if __name__ == "__main__":
    main()

