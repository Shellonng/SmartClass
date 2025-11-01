"""
Speech Processor
语音处理模块：ASR + 填充词检测 + 韵律分析
"""

import whisper
import torch
import numpy as np
import librosa
from typing import Dict, List, Any
from pathlib import Path

from utils.logger import setup_logger

logger = setup_logger(__name__)


class SpeechProcessor:
    """
    语音处理器：集成Whisper ASR和填充词检测
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化语音处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.speech_config = config.get('speech', {})
        self.asr_config = config['models']['asr']
        
        # 加载Whisper模型
        logger.info(f"加载Whisper模型: {self.asr_config['name']}")
        self.asr_model = whisper.load_model(self.asr_config['name'])
        
        # 填充词列表
        self.filler_words = set(self.speech_config.get('filler_words', []))
        
        # 参数
        self.sample_rate = self.speech_config.get('sample_rate', 16000)
        self.long_pause_threshold = self.speech_config['hesitation']['long_pause_threshold']
        self.speech_rate_low = self.speech_config['hesitation']['speech_rate_low']
        
        logger.info("语音处理器初始化完成")
    
    def transcribe_with_analysis(
        self, 
        audio_path: str,
        return_timestamps: bool = True
    ) -> Dict[str, Any]:
        """
        转录音频并分析语音特征
        
        Args:
            audio_path: 音频文件路径
            return_timestamps: 是否返回时间戳信息
            
        Returns:
            包含转录文本和分析结果的字典
        """
        logger.info(f"开始处理音频: {audio_path}")
        
        # 1. Whisper转录
        result = self.asr_model.transcribe(
            audio_path,
            language=self.asr_config.get('language', 'zh'),
            task="transcribe",
            word_timestamps=return_timestamps,
            verbose=False
        )
        
        text = result["text"].strip()
        segments = result.get("segments", [])
        
        if not text:
            logger.warning("转录结果为空")
            return self._empty_result()
        
        # 2. 统计填充词
        filler_analysis = self._analyze_fillers(text, segments)
        
        # 3. 计算犹豫程度
        hesitation_score = self._calculate_hesitation(
            segments, 
            filler_analysis['count'], 
            len(text)
        )
        
        # 4. 检测停顿
        pauses = self._detect_pauses(audio_path) if Path(audio_path).exists() else []
        
        # 5. 计算语速
        speech_rate = self._calculate_speech_rate(text, segments)
        
        result_dict = {
            "text": text,
            "filler_count": filler_analysis['count'],
            "filler_positions": filler_analysis['positions'],
            "filler_words_found": filler_analysis['words'],
            "hesitation_score": hesitation_score,
            "speech_rate": speech_rate,
            "pauses": pauses,
            "confidence": result.get("confidence", 1.0),
            "segments": segments if return_timestamps else []
        }
        
        logger.info(f"转录完成: 文本长度={len(text)}, 填充词={filler_analysis['count']}, "
                   f"犹豫度={hesitation_score:.2f}")
        
        return result_dict
    
    def _analyze_fillers(
        self, 
        text: str, 
        segments: List[Dict]
    ) -> Dict[str, Any]:
        """
        分析填充词
        
        Args:
            text: 转录文本
            segments: Whisper分段结果
            
        Returns:
            填充词分析结果
        """
        filler_count = 0
        filler_positions = []
        filler_words_found = []
        
        # 从segments中提取单词级信息
        for segment in segments:
            words = segment.get("words", [])
            if not words:
                # 如果没有单词级时间戳，分析segment文本
                segment_text = segment.get("text", "")
                for filler in self.filler_words:
                    if filler in segment_text:
                        filler_count += segment_text.count(filler)
                        filler_words_found.append(filler)
            else:
                # 使用单词级时间戳
                for word_info in words:
                    word = word_info.get("word", "").strip()
                    if word in self.filler_words:
                        filler_count += 1
                        filler_positions.append(word_info.get("start", 0))
                        filler_words_found.append(word)
        
        return {
            "count": filler_count,
            "positions": filler_positions,
            "words": list(set(filler_words_found))
        }
    
    def _calculate_hesitation(
        self, 
        segments: List[Dict], 
        filler_count: int, 
        text_length: int
    ) -> float:
        """
        计算犹豫程度评分 (0-1)
        
        Args:
            segments: 音频片段
            filler_count: 填充词数量
            text_length: 文本长度
            
        Returns:
            犹豫评分 (0=流畅, 1=非常犹豫)
        """
        if not segments or text_length == 0:
            return 0.0
        
        # 1. 填充词密度
        words_count = len(segments)
        filler_density = filler_count / max(words_count, 1)
        
        # 2. 语速分析
        total_time = segments[-1]["end"] - segments[0]["start"]
        speech_rate = text_length / max(total_time, 0.1)
        speech_rate_score = 0.0
        if speech_rate < self.speech_rate_low:
            speech_rate_score = 1 - (speech_rate / self.speech_rate_low)
        
        # 3. 停顿频率
        pause_count = 0
        for i in range(len(segments) - 1):
            gap = segments[i + 1]["start"] - segments[i]["end"]
            if gap > 0.5:  # 大于0.5秒的间隔
                pause_count += 1
        pause_score = pause_count / max(len(segments), 1)
        
        # 综合评分（可调权重）
        hesitation_score = (
            filler_density * 0.4 +
            speech_rate_score * 0.3 +
            pause_score * 0.3
        )
        
        return min(hesitation_score, 1.0)
    
    def _detect_pauses(self, audio_path: str) -> List[float]:
        """
        检测音频中的长停顿
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            停顿时长列表（秒）
        """
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # 计算短时能量
            frame_length = int(sr * 0.025)  # 25ms
            hop_length = int(sr * 0.010)    # 10ms
            
            energy = librosa.feature.rms(
                y=y, 
                frame_length=frame_length,
                hop_length=hop_length
            )[0]
            
            # 动态阈值（基于平均能量）
            threshold = np.mean(energy) * 0.3
            
            # 检测静音段
            is_silent = energy < threshold
            
            # 统计连续静音时长
            pauses = []
            current_pause = 0
            
            for silent in is_silent:
                if silent:
                    current_pause += hop_length / sr
                else:
                    if current_pause > self.long_pause_threshold:
                        pauses.append(round(current_pause, 2))
                    current_pause = 0
            
            # 检查最后一段
            if current_pause > self.long_pause_threshold:
                pauses.append(round(current_pause, 2))
            
            return pauses
            
        except Exception as e:
            logger.warning(f"停顿检测失败: {str(e)}")
            return []
    
    def _calculate_speech_rate(
        self, 
        text: str, 
        segments: List[Dict]
    ) -> float:
        """
        计算语速（字/秒）
        
        Args:
            text: 转录文本
            segments: 音频片段
            
        Returns:
            语速（字/秒）
        """
        if not segments:
            return 0.0
        
        total_time = segments[-1]["end"] - segments[0]["start"]
        if total_time <= 0:
            return 0.0
        
        # 中文按字符数，英文按单词数
        char_count = len(text)
        speech_rate = char_count / total_time
        
        return round(speech_rate, 2)
    
    def _empty_result(self) -> Dict[str, Any]:
        """
        返回空结果
        """
        return {
            "text": "",
            "filler_count": 0,
            "filler_positions": [],
            "filler_words_found": [],
            "hesitation_score": 0.0,
            "speech_rate": 0.0,
            "pauses": [],
            "confidence": 0.0,
            "segments": []
        }
    
    def is_answer_sufficient(self, speech_analysis: Dict[str, Any]) -> bool:
        """
        判断回答是否充分（基于语音特征）
        
        Args:
            speech_analysis: 语音分析结果
            
        Returns:
            是否充分
        """
        # 文本过短
        if len(speech_analysis['text']) < 30:
            return False
        
        # 犹豫程度过高
        if speech_analysis['hesitation_score'] > 0.7:
            return False
        
        # 明显的消极回答
        negative_phrases = ["不知道", "不清楚", "不太了解", "没听说过", "不会"]
        text = speech_analysis['text']
        if any(phrase in text for phrase in negative_phrases):
            return False
        
        return True

