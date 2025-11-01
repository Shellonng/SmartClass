"""
数字人模块 - 火山引擎TTS（抖音同款，音质优秀）
文档：https://www.volcengine.com/docs/6561/1329505
"""
from pathlib import Path
import hashlib
import base64
import json
import requests
import uuid

class DigitalHuman:
    """数字人类 - 使用火山引擎TTS"""
    
    def __init__(self):
        """初始化数字人模块"""
        self.audio_dir = Path("audio")
        self.audio_dir.mkdir(exist_ok=True)
        
        # 火山引擎配置
        self.appid = "8133746297"  # 您的APP ID
        self.access_token = "FS0ZfP6NkW95nQwlWhbenvwXCpq8YBG7"
        self.secret_key = "Z0ukMezxmBvri1QtPFfGDXJBsMF1MbB8"
        
        # API配置
        self.api_url = "https://openspeech.bytedance.com/api/v1/tts"
        
        # 语音配置（从您的截图中选择一个音色）
        # 推荐音色：
        self.voice_type = "zh_female_vv_uranus_bigtts"  # vivi 2.0（自然女声）
        # 或者：
        # self.voice_type = "zh_male_dayi_saturn_bigtts"  # 大壹（男声）
        # self.voice_type = "zh_female_santongyongns_saturn_bigtts"  # 流畅女声
        
        self.tts_available = True
        print(f"[INFO] Volcengine TTS initialized")
        print(f"  AppID: {self.appid}")
        print(f"  Voice: {self.voice_type}")
        print(f"  Audio dir: {self.audio_dir}")
    
    def text_to_speech(self, text: str, use_cache: bool = True) -> str:
        """将文字转换为语音
        
        Args:
            text: 要合成的文本
            use_cache: 是否使用缓存
            
        Returns:
            音频文件路径，失败返回None
        """
        # 使用哈希作为文件名（缓存机制）
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        mp3_file = self.audio_dir / f"speech_{text_hash}.mp3"
        
        if use_cache and mp3_file.exists():
            print(f"[INFO] Using cached: {mp3_file.name}")
            return str(mp3_file)
        
        try:
            print(f"[INFO] Generating: {text[:50]}...")
            
            # 准备请求（根据火山引擎官方文档）
            headers = {
                "Authorization": f"Bearer; {self.access_token}",
                "Content-Type": "application/json"
            }
            
            # 生成唯一请求ID
            reqid = str(uuid.uuid4())
            
            payload = {
                "app": {
                    "appid": self.appid,
                    "token": "access_token",  # 固定值
                    "cluster": "volcano_tts"
                },
                "user": {
                    "uid": "388808087185088"  # 可以是任意用户ID
                },
                "audio": {
                    "voice_type": self.voice_type,
                    "encoding": "mp3",
                    "speed_ratio": 1.0,  # 语速（0.5-2.0）
                    "volume_ratio": 1.0,  # 音量（0.1-3.0）
                    "pitch_ratio": 1.0   # 音调（0.5-2.0）
                },
                "request": {
                    "reqid": reqid,
                    "text": text,
                    "text_type": "plain",
                    "operation": "query",
                    "with_frontend": 1,
                    "frontend_type": "unitTson"
                }
            }
            
            # 发送请求
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            print(f"[DEBUG] HTTP Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"[DEBUG] Response code: {result.get('code')}")
                
                if result.get("code") == 3000:  # 成功
                    # 获取音频数据（base64编码）
                    audio_data = result.get("data")
                    if audio_data:
                        # 解码并保存
                        audio_bytes = base64.b64decode(audio_data)
                        mp3_file.write_bytes(audio_bytes)
                        
                        if mp3_file.exists() and mp3_file.stat().st_size > 1000:
                            size_kb = mp3_file.stat().st_size / 1024
                            print(f"[SUCCESS] Generated: {mp3_file.name} ({size_kb:.2f} KB)")
                            return str(mp3_file)
                        else:
                            print(f"[ERROR] File too small")
                            return None
                    else:
                        print(f"[ERROR] No audio data in response")
                        print(f"[DEBUG] Full response: {result}")
                        return None
                else:
                    print(f"[ERROR] API error code: {result.get('code')}")
                    print(f"[ERROR] Message: {result.get('message')}")
                    print(f"[DEBUG] Full response: {result}")
                    return None
            else:
                print(f"[ERROR] HTTP {response.status_code}")
                print(f"[ERROR] Response: {response.text[:500]}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def cleanup_cache(self):
        """清理音频缓存"""
        if self.audio_dir.exists():
            for file in self.audio_dir.glob("speech_*.mp3"):
                file.unlink()
            print("[INFO] Audio cache cleared")
