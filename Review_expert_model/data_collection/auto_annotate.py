"""
自动预标注工具
功能：自动提取特征并生成初步标注，人工只需微调

使用方法：
python auto_annotate.py --video_dir ./raw_data --output annotations.csv
"""
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from transformers import pipeline
import pandas as pd
import os
import json
from tqdm import tqdm

class AutoAnnotator:
    def __init__(self):
        # 加载模型
        print("加载模型...")
        self.asr = pipeline("automatic-speech-recognition", 
                           model="openai/whisper-small")
        self.mp_pose = mp.solutions.pose.Pose()
        self.mp_face = mp.solutions.face_mesh.FaceMesh()
        
        # 填充词库
        self.filler_words = ['嗯', '啊', '呃', '那个', '这个', '就是', '然后', '嘛', '吧']
        
    def process_video(self, video_path, audio_path, question, sample_id):
        """处理一个视频，生成10秒窗口的标注"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # 每10秒一个窗口
        window_size = 10  # 秒
        annotations = []
        
        for start_time in range(0, int(duration), window_size):
            end_time = min(start_time + window_size, int(duration))
            
            # 提取这个窗口的特征
            features = self.extract_window_features(
                cap, fps, start_time, end_time, audio_path
            )
            
            # 自动评分（基于规则，后续人工调整）
            scores = self.auto_score(features, question)
            
            # 检测异常
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
        
        cap.release()
        return annotations
    
    def extract_window_features(self, cap, fps, start_time, end_time, audio_path):
        """提取10秒窗口的特征"""
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        emotions = []
        gazes = []
        poses = []
        
        # 采样帧（每秒采样1帧，节省时间）
        sample_frames = list(range(start_frame, end_frame, int(fps)))
        
        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # 情绪分析
            try:
                emotion_result = DeepFace.analyze(frame, actions=['emotion'], 
                                                 enforce_detection=False)
                emotions.append(emotion_result[0]['emotion'])
            except:
                pass
            
            # 眼动检测
            try:
                face_result = self.mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if face_result.multi_face_landmarks:
                    landmarks = face_result.multi_face_landmarks[0].landmark
                    left_iris = landmarks[468]
                    gaze_x = left_iris.x
                    gaze_deviation = abs(gaze_x - 0.5)
                    gazes.append({'x': gaze_x, 'deviation': gaze_deviation})
            except:
                pass
            
            # 姿势检测
            try:
                pose_result = self.mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if pose_result.pose_landmarks:
                    shoulder_l = pose_result.pose_landmarks.landmark[11]
                    shoulder_r = pose_result.pose_landmarks.landmark[12]
                    shoulder_y = (shoulder_l.y + shoulder_r.y) / 2
                    poses.append({'shoulder_y': shoulder_y})
            except:
                pass
        
        # 语音转录（这个窗口的音频片段）
        transcription = ""
        try:
            # 简化：这里应该提取音频片段，暂时用整段
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            window_audio = audio[int(start_time*sr):int(end_time*sr)]
            
            if len(window_audio) > 0:
                result = self.asr({"array": window_audio.flatten(), "sampling_rate": sr})
                transcription = result.get('text', '')
        except Exception as e:
            print(f"转录失败: {e}")
        
        # 统计特征
        return {
            'emotions': emotions,
            'gazes': gazes,
            'poses': poses,
            'transcription': transcription,
            'summary': self.summarize_features(emotions, gazes, poses, transcription)
        }
    
    def summarize_features(self, emotions, gazes, poses, transcription):
        """生成特征摘要（供人工参考）"""
        summary = []
        
        # 情绪摘要
        if emotions:
            avg_emotions = {k: np.mean([e[k] for e in emotions]) for k in emotions[0].keys()}
            dominant = max(avg_emotions, key=avg_emotions.get)
            summary.append(f"情绪:{dominant}({avg_emotions[dominant]:.1f}%)")
        
        # 眼动摘要
        if gazes:
            avg_deviation = np.mean([g['deviation'] for g in gazes])
            if avg_deviation > 0.08:
                summary.append("眼神游离")
        
        # 转录摘要
        word_count = len(transcription.split())
        filler_count = sum(transcription.count(w) for w in self.filler_words)
        summary.append(f"{word_count}字, {filler_count}个填充词")
        
        return "; ".join(summary)
    
    def auto_score(self, features, question):
        """自动评分（粗略，需要人工调整）"""
        # 专注度
        focus = 80
        if features['gazes']:
            avg_deviation = np.mean([g['deviation'] for g in features['gazes']])
            if avg_deviation > 0.1:
                focus -= 20
        
        # 心理素质
        psychological = 75
        if features['emotions']:
            avg_emotions = {k: np.mean([e[k] for e in features['emotions']]) 
                           for k in features['emotions'][0].keys()}
            negative = avg_emotions.get('fear', 0) + avg_emotions.get('sad', 0)
            if negative > 50:
                psychological -= 15
        
        # 语言表达
        language = 70
        word_count = len(features['transcription'].split())
        filler_count = sum(features['transcription'].count(w) for w in self.filler_words)
        if word_count > 20:
            language += 10
        if filler_count > 3:
            language -= 15
        
        # 专业能力（需要人工重点标注）
        professional = 60  # 默认中等，让人工调整
        
        return {
            'focus': max(0, min(100, focus)),
            'psychological': max(0, min(100, psychological)),
            'language': max(0, min(100, language)),
            'professional': professional
        }
    
    def detect_alert(self, features):
        """检测异常"""
        # 眼神问题
        if features['gazes']:
            avg_deviation = np.mean([g['deviation'] for g in features['gazes']])
            if avg_deviation > 0.1:
                return 1, "请保持眼神专注"
        
        # 情绪问题
        if features['emotions']:
            avg_emotions = {k: np.mean([e[k] for e in features['emotions']]) 
                           for k in features['emotions'][0].keys()}
            if avg_emotions.get('fear', 0) > 40:
                return 2, "深呼吸，放轻松"
        
        # 语言问题
        filler_count = sum(features['transcription'].count(w) for w in self.filler_words)
        if filler_count > 4:
            return 3, "减少口头禅"
        
        return 0, ""


def main():
    """批量处理所有视频"""
    annotator = AutoAnnotator()
    
    # 读取录制的元数据
    raw_dir = "./raw_data"
    all_annotations = []
    
    json_files = [f for f in os.listdir(raw_dir) if f.endswith('.json')]
    
    for json_file in tqdm(json_files, desc="处理视频"):
        json_path = os.path.join(raw_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 处理视频
        annotations = annotator.process_video(
            video_path=metadata['video_path'],
            audio_path=metadata['audio_path'],
            question=metadata['question'],
            sample_id=metadata['sample_id']
        )
        
        all_annotations.extend(annotations)
    
    # 保存为CSV
    df = pd.DataFrame(all_annotations)
    output_path = "annotations_auto.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 自动标注完成！")
    print(f"共生成 {len(all_annotations)} 个标注样本")
    print(f"保存至: {output_path}")
    print(f"\n⚠️ 请人工检查并调整评分，尤其是 professional_score")


if __name__ == "__main__":
    main()

