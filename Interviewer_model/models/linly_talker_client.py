"""
Linly-Talker API 客户端
用于调用Linly-Talker服务生成数字人视频
"""
import requests
import json
from pathlib import Path
from typing import Optional
import tempfile
import os

class LinlyTalkerClient:
    """Linly-Talker API客户端 - 集成 TTS + Talker"""
    
    def __init__(
        self, 
        tts_url: str = "http://localhost:8001",
        talker_url: str = "http://localhost:8003"
    ):
        """
        初始化客户端
        
        Args:
            tts_url: TTS API服务地址
            talker_url: Talker API服务地址
        """
        self.tts_url = tts_url
        self.talker_url = talker_url
        self.tts_available = self._check_service(tts_url)
        self.talker_available = self._check_service(talker_url)
        self.available = self.tts_available and self.talker_available
    
    def _check_service(self, url: str) -> bool:
        """检查服务是否可用"""
        try:
            response = requests.get(f"{url}/docs", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def generate_audio(
        self,
        text: str,
        tts_method: str = "EdgeTTS",
        voice: str = "zh-CN-XiaoxiaoNeural"
    ) -> str:
        """
        使用 TTS API 生成音频
        
        Args:
            text: 要转换的文本
            tts_method: TTS方法 (EdgeTTS, PaddleTTS, GPT_SoVITS, CosyVoice)
            voice: 语音选项
            
        Returns:
            音频文件路径
        """
        if not self.tts_available:
            raise Exception("TTS API 不可用")
        
        # 先切换 TTS 模型
        try:
            requests.post(
                f"{self.tts_url}/tts_change_model/",
                params={"model_name": tts_method},
                timeout=30
            )
        except:
            pass  # 如果已经加载过，忽略错误
        
        # 生成音频
        data = {
            "text": text,
            "voice": voice
        }
        
        response = requests.post(
            f"{self.tts_url}/tts_response",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            # 保存音频到临时文件
            temp_audio = tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.wav',
                dir='temp_audio'
            )
            temp_audio.write(response.content)
            temp_audio.close()
            return temp_audio.name
        else:
            raise Exception(f"音频生成失败: {response.text}")
    
    def generate_video(
        self,
        text: str,
        avatar_image: Optional[str] = None,
        tts_method: str = "EdgeTTS",
        voice: str = "zh-CN-XiaoxiaoNeural",
        talker_method: str = "SadTalker"
    ) -> str:
        """
        生成数字人说话视频 - 完整流程: 文本 -> 音频 -> 视频
        
        Args:
            text: 要说的话
            avatar_image: 头像图片路径
            tts_method: TTS方法
            voice: 语音选项
            talker_method: 数字人驱动方法
            
        Returns:
            视频文件路径
            
        Raises:
            Exception: 如果服务不可用或生成失败
        """
        if not self.available:
            msg = []
            if not self.tts_available:
                msg.append("TTS服务")
            if not self.talker_available:
                msg.append("Talker服务")
            raise Exception(f"{', '.join(msg)} 不可用，请先启动服务")
        
        # 使用默认头像
        if not avatar_image or not Path(avatar_image).exists():
            avatar_image = "Linly-Talker/examples/source_image/full_body_1.png"
            if not Path(avatar_image).exists():
                raise Exception("请提供有效的头像图片路径")
        
        # 1. 生成音频
        print(f"  [1/2] 生成音频...")
        audio_path = self.generate_audio(text, tts_method, voice)
        
        try:
            # 2. 生成视频
            print(f"  [2/2] 生成视频...")
            with open(avatar_image, 'rb') as img_file:
                with open(audio_path, 'rb') as aud_file:
                    files = {
                        'source_image': img_file,
                        'driven_audio': aud_file
                    }
                    data = {
                        'talker_method': talker_method,
                        'preprocess_type': 'crop',
                        'is_still_mode': False,
                        'enhancer': False,
                        'batch_size': 4,
                        'size_of_image': 256,
                        'pose_style': 0,
                        'facerender': 'facevid2vid',
                        'exp_weight': 1.0,
                        'blink_every': True,
                        'fps': 30
                    }
                    
                    response = requests.post(
                        f"{self.talker_url}/talker_response/",
                        data=data,
                        files=files,
                        timeout=120
                    )
            
            if response.status_code == 200:
                # 保存视频到临时文件
                os.makedirs('temp_videos', exist_ok=True)
                temp_video = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix='.mp4',
                    dir='temp_videos'
                )
                temp_video.write(response.content)
                temp_video.close()
                return temp_video.name
            else:
                raise Exception(f"视频生成失败: {response.text}")
        finally:
            # 清理临时音频文件
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
# 使用示例
if __name__ == "__main__":
    # 创建客户端
    client = LinlyTalkerClient()
    
    print(f"TTS 服务: {'✅ 可用' if client.tts_available else '❌ 不可用'}")
    print(f"Talker 服务: {'✅ 可用' if client.talker_available else '❌ 不可用'}")
    
    if client.available:
        print("\n✅ 所有服务可用，开始生成测试视频...")
        
        # 生成视频
        try:
            video_path = client.generate_video(
                text="您好，欢迎参加今天的面试！请先简单介绍一下自己。",
                avatar_image="Linly-Talker/examples/source_image/full_body_1.png"
            )
            print(f"✅ 视频已生成: {video_path}")
        except Exception as e:
            print(f"❌ 生成失败: {e}")
    else:
        print("\n❌ 服务不可用")
        print("   请先启动服务: python start_linly_services.py")


