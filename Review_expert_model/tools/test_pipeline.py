"""
测试Pipeline - 处理testv文件夹中的视频
"""
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from transformers import pipeline
import pandas as pd
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TestPipelineAnnotator:
    def __init__(self):
        # 加载模型
        print("🔧 加载模型中...")
        try:
            print("  ├─ 加载Whisper ASR...")
            self.asr = pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-small",
                device=-1  # CPU模式
            )
            print("  ✅ Whisper加载完成")
        except Exception as e:
            print(f"  ⚠️ Whisper加载失败: {e}")
            self.asr = None
        
        print("  ├─ 加载MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("  ✅ MediaPipe Pose加载完成")
        
        print("  ├─ 加载MediaPipe Face Mesh...")
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("  ✅ MediaPipe Face Mesh加载完成")
        
        # 填充词库
        self.filler_words = ['嗯', '啊', '呃', '那个', '这个', '就是', '然后', '嘛', '吧']
        print("\n✅ 所有模型加载完成！\n")
        
    def extract_audio_from_video(self, video_path):
        """从视频中提取音频"""
        try:
            import subprocess
            import tempfile
            
            # 创建临时音频文件
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio.close()
            
            # 使用ffmpeg提取音频（如果有）
            # 简化版：直接用OpenCV读取，不提取音频
            print(f"    ⚠️ 跳过音频提取（需要ffmpeg），将使用视频直接分析")
            return None
        except Exception as e:
            print(f"    ⚠️ 音频提取失败: {e}")
            return None
    
    def process_video(self, video_path, question, sample_id):
        """处理一个视频，生成10秒窗口的标注"""
        print(f"\n{'='*60}")
        print(f"📹 处理视频: {os.path.basename(video_path)}")
        print(f"❓ 模拟问题: {question}")
        print(f"{'='*60}\n")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"  📊 视频信息:")
        print(f"     ├─ 总帧数: {total_frames}")
        print(f"     ├─ 帧率: {fps:.2f} fps")
        print(f"     └─ 时长: {duration:.2f} 秒")
        
        # 每10秒一个窗口
        window_size = 10  # 秒
        annotations = []
        
        num_windows = int(duration / window_size) + (1 if duration % window_size > 0 else 0)
        print(f"\n  🪟 将分析 {num_windows} 个时间窗口（每个{window_size}秒）\n")
        
        for window_idx in range(num_windows):
            start_time = window_idx * window_size
            end_time = min(start_time + window_size, int(duration))
            
            print(f"  ⏱️  窗口 {window_idx+1}/{num_windows}: {start_time}s - {end_time}s")
            
            # 提取这个窗口的特征
            features = self.extract_window_features(
                cap, fps, start_time, end_time
            )
            
            # 自动评分
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
                'transcription': features.get('transcription', ''),
                'notes': features['summary']
            })
            
            print(f"     ├─ 专注度: {scores['focus']:.0f}")
            print(f"     ├─ 心理素质: {scores['psychological']:.0f}")
            print(f"     ├─ 语言表达: {scores['language']:.0f}")
            print(f"     ├─ 专业能力: {scores['professional']:.0f}")
            if alert_type > 0:
                print(f"     └─ ⚠️  提醒: {alert_text}")
            else:
                print(f"     └─ ✅ 无异常")
        
        cap.release()
        print(f"\n  ✅ 视频处理完成！生成 {len(annotations)} 个标注\n")
        return annotations
    
    def extract_window_features(self, cap, fps, start_time, end_time):
        """提取10秒窗口的特征"""
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        emotions = []
        gazes = []
        poses = []
        
        # 采样帧（每2秒采样1帧，节省时间）
        sample_interval = max(1, int(fps * 2))
        sample_frames = list(range(start_frame, end_frame, sample_interval))
        
        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # 情绪分析
            try:
                emotion_result = DeepFace.analyze(
                    frame, 
                    actions=['emotion'], 
                    enforce_detection=False,
                    silent=True
                )
                emotions.append(emotion_result[0]['emotion'])
            except Exception as e:
                # print(f"       情绪分析失败: {e}")
                pass
            
            # 眼动检测
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_result = self.mp_face.process(frame_rgb)
                if face_result.multi_face_landmarks:
                    landmarks = face_result.multi_face_landmarks[0].landmark
                    # MediaPipe iris landmarks: 468-473
                    left_iris = landmarks[468]
                    gaze_x = left_iris.x
                    gaze_deviation = abs(gaze_x - 0.5)
                    gazes.append({'x': gaze_x, 'deviation': gaze_deviation})
            except Exception as e:
                # print(f"       眼动检测失败: {e}")
                pass
            
            # 姿势检测
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_result = self.mp_pose.process(frame_rgb)
                if pose_result.pose_landmarks:
                    # 肩膀关键点: 11=左肩, 12=右肩
                    shoulder_l = pose_result.pose_landmarks.landmark[11]
                    shoulder_r = pose_result.pose_landmarks.landmark[12]
                    shoulder_y = (shoulder_l.y + shoulder_r.y) / 2
                    poses.append({'shoulder_y': shoulder_y})
            except Exception as e:
                # print(f"       姿势检测失败: {e}")
                pass
        
        # 语音转录（暂时跳过，因为需要从视频提取音频）
        transcription = ""
        
        # 统计特征
        return {
            'emotions': emotions,
            'gazes': gazes,
            'poses': poses,
            'transcription': transcription,
            'summary': self.summarize_features(emotions, gazes, poses, transcription)
        }
    
    def summarize_features(self, emotions, gazes, poses, transcription):
        """生成特征摘要"""
        summary = []
        
        # 情绪摘要
        if emotions:
            avg_emotions = {k: np.mean([e[k] for e in emotions]) for k in emotions[0].keys()}
            dominant = max(avg_emotions, key=avg_emotions.get)
            summary.append(f"情绪:{dominant}({avg_emotions[dominant]:.1f}%)")
        else:
            summary.append("情绪:未检测")
        
        # 眼动摘要
        if gazes:
            avg_deviation = np.mean([g['deviation'] for g in gazes])
            if avg_deviation > 0.08:
                summary.append("眼神偏离")
            else:
                summary.append("眼神专注")
        else:
            summary.append("眼神:未检测")
        
        # 姿势摘要
        if poses:
            summary.append(f"姿势正常")
        else:
            summary.append("姿势:未检测")
        
        return "; ".join(summary)
    
    def auto_score(self, features, question):
        """自动评分"""
        # 专注度（基于眼动和姿势）
        focus = 80
        if features['gazes']:
            avg_deviation = np.mean([g['deviation'] for g in features['gazes']])
            if avg_deviation > 0.1:
                focus -= 20
            elif avg_deviation < 0.05:
                focus += 10
        else:
            focus = 70  # 未检测到，给中等分
        
        # 心理素质（基于情绪）
        psychological = 75
        if features['emotions']:
            avg_emotions = {k: np.mean([e[k] for e in features['emotions']]) 
                           for k in features['emotions'][0].keys()}
            
            # 积极情绪
            positive = avg_emotions.get('happy', 0) + avg_emotions.get('neutral', 0)
            # 消极情绪
            negative = avg_emotions.get('fear', 0) + avg_emotions.get('sad', 0)
            
            if positive > 70:
                psychological += 15
            elif negative > 50:
                psychological -= 20
        else:
            psychological = 70
        
        # 语言表达（暂时基于默认值，因为没有音频）
        language = 70
        
        # 专业能力（需要QA对齐，暂时默认）
        professional = 65
        
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
            
            fear_level = avg_emotions.get('fear', 0)
            sad_level = avg_emotions.get('sad', 0)
            
            if fear_level > 40:
                return 2, "深呼吸，放轻松"
            elif sad_level > 50:
                return 2, "保持积极心态"
        
        return 0, ""


def main():
    """测试Pipeline"""
    print("\n" + "="*60)
    print("     🚀 面试评估Pipeline测试")
    print("="*60 + "\n")
    
    # 初始化标注器
    annotator = TestPipelineAnnotator()
    
    # testv文件夹中的视频
    video_dir = "./testv"
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_files.sort()
    
    print(f"📂 发现 {len(video_files)} 个测试视频:\n")
    for i, vf in enumerate(video_files, 1):
        print(f"   {i}. {vf}")
    print()
    
    # 模拟问题（实际使用时应该提供真实问题）
    questions = {
        't1-1.mp4': "请做一个简短的自我介绍",
        't2-1.mp4': "介绍一下你最近的项目经验",
        't3-1.mp4': "说说你对Python的理解"
    }
    
    all_annotations = []
    
    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(video_dir, video_file)
        sample_id = f"test_{i:03d}"
        question = questions.get(video_file, "通用面试问题")
        
        # 处理视频
        annotations = annotator.process_video(
            video_path=video_path,
            question=question,
            sample_id=sample_id
        )
        
        all_annotations.extend(annotations)
    
    # 保存为CSV
    if all_annotations:
        df = pd.DataFrame(all_annotations)
        output_path = "test_annotations.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*60)
        print("✅ Pipeline测试完成！")
        print("="*60)
        print(f"\n📊 统计信息:")
        print(f"   ├─ 处理视频数: {len(video_files)}")
        print(f"   ├─ 生成标注数: {len(all_annotations)}")
        print(f"   └─ 保存文件: {output_path}")
        
        print(f"\n📈 评分摘要:")
        print(f"   ├─ 专注度: {df['focus_score'].mean():.1f} ± {df['focus_score'].std():.1f}")
        print(f"   ├─ 心理素质: {df['psychological_score'].mean():.1f} ± {df['psychological_score'].std():.1f}")
        print(f"   ├─ 语言表达: {df['language_score'].mean():.1f} ± {df['language_score'].std():.1f}")
        print(f"   └─ 专业能力: {df['professional_score'].mean():.1f} ± {df['professional_score'].std():.1f}")
        
        alert_count = (df['alert_type'] > 0).sum()
        print(f"\n⚠️  检测到异常: {alert_count}/{len(all_annotations)} 个窗口")
        
        print(f"\n💡 下一步:")
        print(f"   1. 打开 {output_path} 查看详细标注")
        print(f"   2. 人工调整评分（尤其是 professional_score）")
        print(f"   3. 准备更多视频数据")
        print(f"   4. 开始训练Transformer模型\n")
    else:
        print("\n❌ 没有生成任何标注，请检查视频文件")


if __name__ == "__main__":
    main()

