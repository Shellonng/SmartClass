# -*- coding: utf-8 -*-
"""
智能自动标注脚本（修复版V2）
修复心理素质评分和总分计算问题
"""
import numpy as np
import json
import os
from typing import Dict, List, Tuple

# 扩展的提醒类别（30个）
REMINDER_CATEGORIES = {
    # 积极反馈类 (0-4)
    0: "表现非常出色，继续保持！",
    1: "回答很流畅，逻辑清晰！",
    2: "专注度很好，眼神交流到位！",
    3: "肢体语言自信，姿态端正！",
    4: "情绪稳定，心理素质优秀！",
    
    # 语言表达类 (5-9)
    5: "语速有点快，可以慢一点",
    6: "说话声音太小，增强音量",
    7: "有些语无伦次，理清思路",
    8: "回答太简短，可以详细些",
    9: "表达很清晰，但可以更有条理",
    
    # 心理素质类 (10-14)
    10: "你好像有点紧张，深呼吸放轻松",
    11: "保持自信，不要过度焦虑",
    12: "情绪有些低落，积极一点",
    13: "遇到难题很正常，冷静思考",
    14: "不要害怕犯错，大胆表达",
    
    # 肢体语言类 (15-19)
    15: "坐直一点，保持良好姿态",
    16: "不要频繁晃动，保持稳定",
    17: "手势可以更丰富一些",
    18: "身体前倾显示兴趣，很好",
    19: "避免交叉双臂，显得封闭",
    
    # 专注度类 (20-24)
    20: "注意力集中，不要眼神乱飘",
    21: "保持眼神接触，展现自信",
    22: "眼神有些疲惫，注意休息",
    23: "看着面试官，不要看天花板",
    24: "专注度下降，重新集中注意力",
    
    # 综合建议类 (25-29)
    25: "整体不错，但还有提升空间",
    26: "前半段很好，保持状态",
    27: "思考时间过长，准备要充分",
    28: "回答有进步，继续加油",
    29: "需要更多练习，多做模拟面试"
}


class FixedAnnotatorV2:
    """修复版标注器V2"""
    
    def __init__(self, features_dir='./features', output_dir='./annotations'):
        self.features_dir = features_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_all_indices(self):
        """加载所有特征索引"""
        indices = {}
        
        emotion_index_file = os.path.join(self.features_dir, 'emotion', 'emotion_index.json')
        with open(emotion_index_file, 'r', encoding='utf-8') as f:
            indices['emotion'] = {item['video_file']: item for item in json.load(f)}
        
        audio_index_file = os.path.join(self.features_dir, 'audio', 'audio_index.json')
        with open(audio_index_file, 'r', encoding='utf-8') as f:
            indices['audio'] = {item['video_file']: item for item in json.load(f)}
        
        pose_index_file = os.path.join(self.features_dir, 'pose', 'pose_index.json')
        with open(pose_index_file, 'r', encoding='utf-8') as f:
            indices['pose'] = {item['video_file']: item for item in json.load(f)}
        
        return indices
    
    def find_common_videos(self, indices):
        """找到在所有模态中都存在的视频"""
        emotion_videos = set(indices['emotion'].keys())
        audio_videos = set(indices['audio'].keys())
        pose_videos = set(indices['pose'].keys())
        
        common = emotion_videos & audio_videos & pose_videos
        return list(common)
    
    def load_features_by_video(self, video_file, indices):
        """通过video_file加载特征"""
        features = {}
        
        # 加载情绪特征
        emotion_item = indices['emotion'][video_file]
        emotion_file = emotion_item['feature_file']
        emotion_data = np.load(emotion_file, allow_pickle=True)
        features['emotion'] = {
            'sequences': emotion_data['emotion_sequences'],
            'metadata': emotion_data['metadata'],
            'sample_id': emotion_item['sample_id']
        }
        
        # 加载音频特征
        audio_item = indices['audio'][video_file]
        audio_file = audio_item['feature_file']
        audio_data = np.load(audio_file, allow_pickle=True)
        features['audio'] = {
            'mel_spectrograms': audio_data['mel_spectrograms'],
            'metadata': audio_data['metadata']
        }
        
        # 加载姿势特征
        pose_item = indices['pose'][video_file]
        pose_file = pose_item['feature_file']
        pose_data = np.load(pose_file, allow_pickle=True)
        features['pose'] = {
            'sequences': pose_data['pose_sequences'],
            'metadata': pose_data['metadata']
        }
        
        # 眼动特征
        if 'gaze_sequences' in pose_data:
            features['gaze'] = {
                'sequences': pose_data['gaze_sequences']
            }
        else:
            features['gaze'] = {
                'sequences': [np.array([[0.5, 0.5, 0.0, 0.5, 0.5]] * 5) 
                             for _ in range(len(pose_data['pose_sequences']))]
            }
        
        return features
    
    def analyze_emotion_stability(self, emotion_seq: np.ndarray) -> Dict:
        """
        分析情绪稳定性（修复版）
        ✅ 添加归一化步骤
        """
        # emotion_seq: [5, 7]
        emotions = ['angry', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'neutral']
        
        # ✅ 归一化：将每一帧转换为概率分布
        emotion_seq_normalized = np.zeros_like(emotion_seq, dtype=np.float32)
        for i in range(len(emotion_seq)):
            row_sum = np.sum(emotion_seq[i])
            if row_sum > 0:
                emotion_seq_normalized[i] = emotion_seq[i] / row_sum
            else:
                emotion_seq_normalized[i] = emotion_seq[i]
        
        # 计算平均情绪分布
        avg_emotions = np.mean(emotion_seq_normalized, axis=0)
        dominant_idx = np.argmax(avg_emotions)
        dominant_emotion = emotions[dominant_idx]
        dominant_prob = avg_emotions[dominant_idx]
        
        # 计算情绪变化幅度
        emotion_variance = np.var(emotion_seq_normalized, axis=0).mean()
        
        # 计算积极/消极情绪比例
        positive_emotions = avg_emotions[1]  # happy
        negative_emotions = avg_emotions[0] + avg_emotions[2] + avg_emotions[4]  # angry, sad, fear
        
        return {
            'dominant_emotion': dominant_emotion,
            'dominant_prob': float(dominant_prob),
            'variance': float(emotion_variance),
            'positive_ratio': float(positive_emotions),
            'negative_ratio': float(negative_emotions),
            'stability': 'stable' if emotion_variance < 0.01 else 'unstable'
        }
    
    def analyze_audio_quality(self, mel_spec: np.ndarray) -> Dict:
        """分析音频质量"""
        energy = np.mean(np.abs(mel_spec))
        variance = np.var(mel_spec)
        
        speech_rate = 'normal'
        if variance > 1.5:
            speech_rate = 'fast'
        elif variance < 0.5:
            speech_rate = 'slow'
        
        fluency = min(100, max(50, 70 + (energy - 0.5) * 30))
        
        return {
            'energy': float(energy),
            'variance': float(variance),
            'speech_rate': speech_rate,
            'fluency': float(fluency),
            'volume': 'normal' if 0.3 < energy < 0.7 else ('low' if energy < 0.3 else 'high')
        }
    
    def analyze_pose_confidence(self, pose_seq: np.ndarray) -> Dict:
        """分析姿势自信度"""
        pose_variance = np.var(pose_seq, axis=0).mean()
        upper_body = pose_seq[:, :33]
        upper_variance = np.var(upper_body, axis=0).mean()
        
        posture_quality = 'good'
        if upper_variance > 0.05:
            posture_quality = 'unstable'
        elif upper_variance < 0.001:
            posture_quality = 'rigid'
        
        return {
            'variance': float(pose_variance),
            'upper_body_variance': float(upper_variance),
            'posture_quality': posture_quality,
            'stability': 'stable' if pose_variance < 0.03 else 'unstable'
        }
    
    def analyze_gaze_focus(self, gaze_seq: np.ndarray) -> Dict:
        """分析眼神专注度"""
        deviations = gaze_seq[:, 2]
        avg_deviation = np.mean(deviations)
        max_deviation = np.max(deviations)
        gaze_variance = np.var(gaze_seq[:, :2], axis=0).mean()
        
        focus_level = 'high'
        if avg_deviation > 0.15:
            focus_level = 'low'
        elif avg_deviation > 0.08:
            focus_level = 'medium'
        
        return {
            'avg_deviation': float(avg_deviation),
            'max_deviation': float(max_deviation),
            'variance': float(gaze_variance),
            'focus_level': focus_level,
            'stable': gaze_variance < 0.05
        }
    
    def generate_scores(self, analyses: Dict) -> Dict[str, float]:
        """
        根据分析结果生成5个评分（修复版）
        ✅ 修复心理素质评分算法
        ✅ 修复总分计算（在clip之后）
        """
        
        # 1. 语言表达能力 (基于音频)
        audio_analysis = analyses['audio']
        language_score = 70.0
        
        language_score += (audio_analysis['fluency'] - 70) * 0.3
        
        if audio_analysis['speech_rate'] == 'fast':
            language_score -= 8
        elif audio_analysis['speech_rate'] == 'slow':
            language_score -= 5
        
        if audio_analysis['volume'] == 'low':
            language_score -= 10
        elif audio_analysis['volume'] == 'high':
            language_score -= 5
        
        # 添加随机噪声
        language_score += np.random.randn() * 3
        language_score = float(np.clip(language_score, 0, 100))
        
        # 2. 心理素质 (基于情绪) ✅ 修复版
        emotion_analysis = analyses['emotion']
        psychological_score = 70.0  # 提高基础分
        
        # ✅ 积极情绪加分（降低系数）
        psychological_score += emotion_analysis['positive_ratio'] * 25  # 30→25
        
        # ✅ 消极情绪减分（降低系数，避免过度惩罚）
        psychological_score -= emotion_analysis['negative_ratio'] * 20  # 40→20
        
        # 稳定性影响
        if emotion_analysis['stability'] == 'stable':
            psychological_score += 10
        else:
            psychological_score -= 5  # 减小惩罚 8→5
        
        # 主导情绪影响
        if emotion_analysis['dominant_emotion'] in ['happy', 'neutral']:
            psychological_score += 8
        elif emotion_analysis['dominant_emotion'] in ['fear', 'sad', 'angry']:
            psychological_score -= 10
        elif emotion_analysis['dominant_emotion'] == 'surprise':
            psychological_score += 3
        
        # 添加随机噪声
        psychological_score += np.random.randn() * 3
        psychological_score = float(np.clip(psychological_score, 0, 100))
        
        # 3. 肢体语言 (基于姿势)
        pose_analysis = analyses['pose']
        body_language_score = 75.0
        
        if pose_analysis['posture_quality'] == 'good':
            body_language_score += 15
        elif pose_analysis['posture_quality'] == 'unstable':
            body_language_score -= 12
        elif pose_analysis['posture_quality'] == 'rigid':
            body_language_score -= 8
        
        if pose_analysis['stability'] == 'stable':
            body_language_score += 8
        else:
            body_language_score -= 10
        
        # 添加随机噪声
        body_language_score += np.random.randn() * 3
        body_language_score = float(np.clip(body_language_score, 0, 100))
        
        # 4. 专注度 (基于眼动)
        gaze_analysis = analyses['gaze']
        focus_score = 75.0
        
        if gaze_analysis['focus_level'] == 'high':
            focus_score += 18
        elif gaze_analysis['focus_level'] == 'medium':
            focus_score += 5
        else:
            focus_score -= 20
        
        if gaze_analysis['stable']:
            focus_score += 7
        else:
            focus_score -= 8
        
        # 添加随机噪声
        focus_score += np.random.randn() * 3
        focus_score = float(np.clip(focus_score, 0, 100))
        
        # 5. 总分 ✅ 在所有分数clip之后计算
        total_score = (
            language_score * 0.25 +
            psychological_score * 0.30 +
            body_language_score * 0.20 +
            focus_score * 0.25
        )
        
        # 添加随机噪声
        total_score += np.random.randn() * 2
        total_score = float(np.clip(total_score, 0, 100))
        
        return {
            'language': language_score,
            'psychological': psychological_score,
            'body_language': body_language_score,
            'focus': focus_score,
            'total': total_score
        }
    
    def generate_reminder(self, scores: Dict, analyses: Dict) -> Tuple[int, str]:
        """根据评分和分析生成提醒"""
        
        # 优先级：严重问题 > 中等问题 > 建议 > 鼓励
        
        # 1. 检查严重问题（分数<60）
        if scores['psychological'] < 60 and analyses['emotion']['negative_ratio'] > 0.2:
            if analyses['emotion']['dominant_emotion'] == 'fear':
                return 10, REMINDER_CATEGORIES[10]
            elif analyses['emotion']['dominant_emotion'] == 'sad':
                return 12, REMINDER_CATEGORIES[12]
            elif analyses['emotion']['dominant_emotion'] == 'angry':
                return 11, REMINDER_CATEGORIES[11]
        
        if scores['focus'] < 55:
            if analyses['gaze']['avg_deviation'] > 0.15:
                return 20, REMINDER_CATEGORIES[20]
            elif analyses['gaze']['max_deviation'] > 0.25:
                return 23, REMINDER_CATEGORIES[23]
        
        if scores['body_language'] < 60:
            if analyses['pose']['posture_quality'] == 'unstable':
                return 16, REMINDER_CATEGORIES[16]
            else:
                return 15, REMINDER_CATEGORIES[15]
        
        if scores['language'] < 60:
            if analyses['audio']['speech_rate'] == 'fast':
                return 5, REMINDER_CATEGORIES[5]
            elif analyses['audio']['volume'] == 'low':
                return 6, REMINDER_CATEGORIES[6]
            else:
                return 7, REMINDER_CATEGORIES[7]
        
        # 2. 检查中等问题（60-75）
        if 60 <= scores['focus'] < 75:
            return 24, REMINDER_CATEGORIES[24]
        
        if 60 <= scores['psychological'] < 75:
            if analyses['emotion']['stability'] == 'unstable':
                return 11, REMINDER_CATEGORIES[11]
            else:
                return 13, REMINDER_CATEGORIES[13]
        
        if 60 <= scores['language'] < 75:
            if analyses['audio']['fluency'] < 65:
                return 7, REMINDER_CATEGORIES[7]
            else:
                return 9, REMINDER_CATEGORIES[9]
        
        if 60 <= scores['body_language'] < 75:
            return 17, REMINDER_CATEGORIES[17]
        
        # 3. 表现良好（75-85）- 给建议
        if 75 <= scores['total'] < 85:
            if scores['language'] >= 80:
                return 1, REMINDER_CATEGORIES[1]
            elif scores['body_language'] >= 80:
                return 18, REMINDER_CATEGORIES[18]
            elif scores['focus'] >= 80:
                return 2, REMINDER_CATEGORIES[2]
            else:
                return 25, REMINDER_CATEGORIES[25]
        
        # 4. 表现优秀（85+）- 鼓励
        if scores['total'] >= 85:
            if scores['focus'] >= 90:
                return 2, REMINDER_CATEGORIES[2]
            elif scores['psychological'] >= 85:
                return 4, REMINDER_CATEGORIES[4]
            elif scores['language'] >= 85:
                return 1, REMINDER_CATEGORIES[1]
            elif scores['body_language'] >= 85:
                return 3, REMINDER_CATEGORIES[3]
            else:
                return 0, REMINDER_CATEGORIES[0]
        
        # 默认：通用鼓励
        return 28, REMINDER_CATEGORIES[28]
    
    def annotate_video(self, video_file, indices):
        """为单个视频生成标注"""
        print(f"\n[Annotating] {video_file}")
        
        features = self.load_features_by_video(video_file, indices)
        sample_id = features['emotion']['sample_id']
        
        num_windows = len(features['emotion']['sequences'])
        
        annotations = {
            'video_file': video_file,
            'sample_id': sample_id,
            'num_windows': num_windows,
            'windows': []
        }
        
        for window_idx in range(num_windows):
            emotion_seq = features['emotion']['sequences'][window_idx]
            audio_mel = features['audio']['mel_spectrograms'][window_idx]
            pose_seq = features['pose']['sequences'][window_idx]
            gaze_seq = features['gaze']['sequences'][window_idx]
            
            # 分析各模态
            analyses = {
                'emotion': self.analyze_emotion_stability(emotion_seq),
                'audio': self.analyze_audio_quality(audio_mel),
                'pose': self.analyze_pose_confidence(pose_seq),
                'gaze': self.analyze_gaze_focus(gaze_seq)
            }
            
            # 生成评分
            scores = self.generate_scores(analyses)
            
            # 生成提醒
            reminder_class, reminder_text = self.generate_reminder(scores, analyses)
            
            window_annotation = {
                'window_idx': window_idx,
                'start_time': window_idx * 10,
                'end_time': (window_idx + 1) * 10,
                'scores': scores,
                'reminder': {
                    'class': reminder_class,
                    'text': reminder_text
                },
                'analyses': {
                    'emotion': {
                        'dominant': analyses['emotion']['dominant_emotion'],
                        'stability': analyses['emotion']['stability'],
                        'positive_ratio': analyses['emotion']['positive_ratio'],
                        'negative_ratio': analyses['emotion']['negative_ratio']
                    },
                    'audio': {
                        'speech_rate': analyses['audio']['speech_rate'],
                        'volume': analyses['audio']['volume']
                    },
                    'pose': {
                        'posture_quality': analyses['pose']['posture_quality']
                    },
                    'gaze': {
                        'focus_level': analyses['gaze']['focus_level']
                    }
                }
            }
            
            annotations['windows'].append(window_annotation)
            
            print(f"  W{window_idx:02d}: Scores=[L:{scores['language']:.0f} "
                  f"P:{scores['psychological']:.0f} B:{scores['body_language']:.0f} "
                  f"F:{scores['focus']:.0f} T:{scores['total']:.0f}] "
                  f"R:{reminder_class}")
        
        return annotations
    
    def save_annotations(self, annotations: Dict, sample_id: str):
        """保存标注结果"""
        output_file = os.path.join(self.output_dir, f'{sample_id}_labels.json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Saved to {output_file}")
    
    def generate_dataset(self):
        """生成完整数据集"""
        print("="*60)
        print("  Intelligent Auto Annotation (Fixed V2)")
        print("="*60)
        print(f"Features directory: {self.features_dir}")
        print(f"Output directory: {self.output_dir}")
        print("="*60 + "\n")
        
        indices = self.load_all_indices()
        common_videos = self.find_common_videos(indices)
        
        print(f"[INFO] Found {len(common_videos)} videos with complete features\n")
        
        if len(common_videos) == 0:
            print("[ERROR] No videos with all 3 modalities")
            return
        
        all_annotations = []
        for video_file in common_videos:
            annotations = self.annotate_video(video_file, indices)
            
            if annotations:
                sample_id = annotations['sample_id']
                self.save_annotations(annotations, sample_id)
                all_annotations.append(annotations)
        
        # 保存汇总信息
        summary = {
            'total_videos': len(all_annotations),
            'total_windows': sum(a['num_windows'] for a in all_annotations),
            'reminder_categories': REMINDER_CATEGORIES,
            'videos': [(a['video_file'], a['sample_id']) for a in all_annotations]
        }
        
        summary_file = os.path.join(self.output_dir, 'dataset_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 统计评分分布
        all_scores = {
            'language': [],
            'psychological': [],
            'body_language': [],
            'focus': [],
            'total': []
        }
        
        for ann in all_annotations:
            for window in ann['windows']:
                for key in all_scores.keys():
                    all_scores[key].append(window['scores'][key])
        
        print(f"\n{'='*60}")
        print(f"[SUCCESS] Dataset generation completed!")
        print(f"  Total videos: {summary['total_videos']}")
        print(f"  Total windows: {summary['total_windows']}")
        print(f"\n评分统计:")
        for key, values in all_scores.items():
            print(f"  {key:15s}: {np.mean(values):5.1f} ± {np.std(values):4.1f}  "
                  f"[{np.min(values):5.1f}, {np.max(values):5.1f}]")
        print(f"  Summary saved to: {summary_file}")
        print(f"{'='*60}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-generate training dataset (fixed v2)')
    parser.add_argument('--features_dir', type=str, default='./features',
                       help='Features directory')
    parser.add_argument('--output_dir', type=str, default='./annotations',
                       help='Output directory for annotations')
    
    args = parser.parse_args()
    
    annotator = FixedAnnotatorV2(
        features_dir=args.features_dir,
        output_dir=args.output_dir
    )
    
    annotator.generate_dataset()


if __name__ == "__main__":
    main()


