"""
AI面试评分系统 - 真正的实时演示（修复版）
修复: 单一摄像头捕获，帧分发给各个提取线程
环境: interview_realtime
"""

import cv2
import numpy as np
import torch
import time
import os
from collections import deque
from threading import Thread
import queue
from PIL import Image, ImageDraw, ImageFont

# 导入特征提取库
from deepface import DeepFace
import mediapipe as mp
import pyaudio
import librosa

from model.transformer_model import InterviewTransformer, REMINDER_MAP

# ==================== 配置 ====================
WINDOW_SIZE = 5
MODEL_PATH = './checkpoints/best_model.pth'
CAMERA_ID = 0
AUDIO_RATE = 16000
AUDIO_CHUNK = 1024

# ==================== 全局变量 ====================
emotion_buffer = deque(maxlen=WINDOW_SIZE)
audio_buffer = deque(maxlen=WINDOW_SIZE)
pose_buffer = deque(maxlen=WINDOW_SIZE)
gaze_buffer = deque(maxlen=WINDOW_SIZE)

current_scores = None
current_reminder = "Initializing..."

# 特征队列（线程安全）
emotion_queue = queue.Queue(maxsize=5)
audio_queue = queue.Queue(maxsize=5)
pose_gaze_queue = queue.Queue(maxsize=5)

# 帧分发队列
emotion_frame_queue = queue.Queue(maxsize=2)
pose_frame_queue = queue.Queue(maxsize=2)

is_running = True

# ==================== 模型加载 ====================
def load_model(device='cpu'):
    """加载训练好的模型"""
    model = InterviewTransformer(
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        dropout=0.3,
        num_reminders=30
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"[OK] Model loaded: {MODEL_PATH}")
        return model
    else:
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        return None

# ==================== 真实特征提取（线程） ====================
class EmotionExtractor(Thread):
    """情绪特征提取线程 - 从队列接收帧"""
    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.daemon = True
        
    def run(self):
        global is_running
        print("[Emotion] Thread started")
        frame_count = 0
        
        # 默认中性情绪
        default_emotion = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9], dtype=np.float32)
        
        while is_running:
            try:
                # 从队列获取帧（超时1秒）
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            # 每帧都提取（提高成功率）
            emotion_features = default_emotion.copy()
            
            try:
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv',
                    silent=True
                )
                
                if isinstance(result, list):
                    result = result[0]
                
                emotion_scores = result.get('emotion', {})
                emotion_features = np.array([
                    emotion_scores.get('angry', 0.0) / 100.0,
                    emotion_scores.get('disgust', 0.0) / 100.0,
                    emotion_scores.get('fear', 0.0) / 100.0,
                    emotion_scores.get('happy', 0.0) / 100.0,
                    emotion_scores.get('sad', 0.0) / 100.0,
                    emotion_scores.get('surprise', 0.0) / 100.0,
                    emotion_scores.get('neutral', 0.0) / 100.0
                ], dtype=np.float32)
                
            except Exception as e:
                pass  # 使用默认值
            
            # 总是放入特征（即使是默认值）
            if not emotion_queue.full():
                emotion_queue.put(emotion_features)
            
            frame_count += 1
            time.sleep(0.1)  # 避免CPU占用过高
        
        print("[Emotion] Thread stopped")


class AudioExtractor(Thread):
    """音频特征提取线程（独立录音）"""
    def __init__(self):
        super().__init__()
        self.daemon = True
        
    def run(self):
        global is_running
        
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=AUDIO_RATE,
                input=True,
                frames_per_buffer=AUDIO_CHUNK
            )
            
            print("[Audio] Thread started")
            
            while is_running:
                try:
                    audio_data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    if len(audio_np) > 512:
                        mel_spec = librosa.feature.melspectrogram(
                            y=audio_np,
                            sr=AUDIO_RATE,
                            n_mels=80,
                            hop_length=512
                        )
                        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
                        audio_features = np.mean(mel_db, axis=1).astype(np.float32)
                        
                        if not audio_queue.full():
                            audio_queue.put(audio_features)
                    
                except Exception as e:
                    pass
                
                time.sleep(0.1)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("[Audio] Thread stopped")
            
        except Exception as e:
            print(f"[Audio] Failed to start: {e}")


class PoseGazeExtractor(Thread):
    """姿势和眼动特征提取线程 - 从队列接收帧"""
    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.daemon = True
        
    def run(self):
        global is_running
        
        mp_pose = mp.solutions.pose
        mp_face_mesh = mp.solutions.face_mesh
        
        pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        print("[Pose/Gaze] Thread started")
        frame_count = 0
        
        # 默认特征
        default_pose = np.zeros(99, dtype=np.float32)
        default_gaze = np.array([0.5, 0.5, 0.0, 0.5, 0.5], dtype=np.float32)
        
        while is_running:
            try:
                # 从队列获取帧（超时1秒）
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            # 每帧都提取（提高成功率）
            pose_features = default_pose.copy()
            gaze_features = default_gaze.copy()
            
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 姿势
                pose_results = pose_detector.process(rgb_frame)
                
                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
                    for i, lm in enumerate(landmarks[:33]):
                        pose_features[i*3] = lm.x
                        pose_features[i*3+1] = lm.y
                        pose_features[i*3+2] = lm.z
                
                # 眼动
                face_results = face_mesh.process(rgb_frame)
                
                if face_results.multi_face_landmarks:
                    face_landmarks = face_results.multi_face_landmarks[0].landmark
                    left_eye = face_landmarks[33]
                    right_eye = face_landmarks[263]
                    gaze_features = np.array([
                        left_eye.x, left_eye.y,
                        0.0,
                        right_eye.x, right_eye.y
                    ], dtype=np.float32)
                
            except Exception as e:
                pass  # 使用默认值
            
            # 总是放入特征（即使是默认值）
            if not pose_gaze_queue.full():
                pose_gaze_queue.put((pose_features, gaze_features))
            
            frame_count += 1
            time.sleep(0.1)
        
        print("[Pose/Gaze] Thread stopped")


# ==================== 提醒文本中文映射 ====================
REMINDER_TEXT_CN = {
    0: "说话更清晰", 1: "减少口头禅", 2: "控制语速",
    3: "组织好逻辑", 4: "改善表达方式", 5: "保持冷静",
    6: "更加自信", 7: "控制焦虑", 8: "保持专注",
    9: "避免紧张", 10: "坐直身体", 11: "减少不必要动作",
    12: "适当使用手势", 13: "保持良好姿态", 14: "控制肢体动作",
    15: "保持眼神交流", 16: "避免分心", 17: "保持投入",
    18: "专注面试官", 19: "避免移开视线", 20: "表现出色！",
    21: "表达很好", 22: "心态沉稳", 23: "肢体语言良好",
    24: "注意力集中", 25: "职业形象好", 26: "沟通能力强",
    27: "自信表达", 28: "气场很好", 29: "面试表现优秀！"
}

def get_reminder_text(idx):
    """获取提醒文本（中文）"""
    return REMINDER_TEXT_CN.get(idx, "分析中...")


# ==================== 推理和显示 ====================
def run_inference(model, device='cpu'):
    """运行模型推理"""
    global current_scores, current_reminder
    
    # 检查所有buffer是否都满足要求
    if (len(emotion_buffer) < WINDOW_SIZE or 
        len(audio_buffer) < WINDOW_SIZE or 
        len(pose_buffer) < WINDOW_SIZE or 
        len(gaze_buffer) < WINDOW_SIZE):
        print(f"[Inference] Waiting for buffers: E:{len(emotion_buffer)} A:{len(audio_buffer)} P:{len(pose_buffer)} G:{len(gaze_buffer)}/{WINDOW_SIZE}")
        return
    
    try:
        # 准备输入
        emotion_arr = np.array(list(emotion_buffer))
        audio_arr = np.array(list(audio_buffer))
        pose_arr = np.array(list(pose_buffer))
        gaze_arr = np.array(list(gaze_buffer))
        
        print(f"[Inference] Input shapes: E:{emotion_arr.shape} A:{audio_arr.shape} P:{pose_arr.shape} G:{gaze_arr.shape}")
        
        emotion_seq = torch.tensor(emotion_arr, dtype=torch.float32).unsqueeze(0).to(device)
        audio_seq = torch.tensor(audio_arr, dtype=torch.float32).unsqueeze(0).to(device)
        pose_seq = torch.tensor(pose_arr, dtype=torch.float32).unsqueeze(0).to(device)
        gaze_seq = torch.tensor(gaze_arr, dtype=torch.float32).unsqueeze(0).to(device)
        
        print(f"[Inference] Tensor shapes: E:{emotion_seq.shape} A:{audio_seq.shape} P:{pose_seq.shape} G:{gaze_seq.shape}")
        
        # 推理
        with torch.no_grad():
            scores_pred, reminder_pred, _ = model(emotion_seq, audio_seq, pose_seq, gaze_seq)
        
        scores_np = scores_pred.cpu().numpy()[0]
        reminder_idx = torch.argmax(reminder_pred, dim=1).item()
        
        current_scores = scores_np
        current_reminder = get_reminder_text(reminder_idx)
        
        print(f"[Inference SUCCESS] Scores: {scores_np}, Reminder: {current_reminder}")
        
    except Exception as e:
        import traceback
        print(f"[Inference Error] {e}")
        print(traceback.format_exc())


def cv2_add_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """在OpenCV图像上添加中文文字（使用PIL）"""
    # 转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 尝试使用系统字体
    try:
        # Windows系统字体路径
        font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
        if not os.path.exists(font_path):
            font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
        if not os.path.exists(font_path):
            font_path = None
        
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # 绘制文字
    draw.text(position, text, font=font, fill=color)
    
    # 转换回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_ui(frame, scores, reminder_text, fps):
    """绘制优化的UI界面（支持中文）"""
    h, w = frame.shape[:2]
    
    # 创建右侧评分面板（更宽，深色背景）
    panel = np.ones((h, 500, 3), dtype=np.uint8) * 30
    
    # 顶部渐变装饰条（缩小）
    for i in range(5):
        color_val = 60 + i * 10
        cv2.rectangle(panel, (0, i*2), (500, (i+1)*2), (color_val, color_val, color_val), -1)
    
    # 标题（使用PIL绘制中文，缩小）
    panel = cv2_add_chinese_text(panel, "AI 智能面试评分", (30, 18), 24, (255, 255, 255))
    
    # 装饰线
    cv2.line(panel, (30, 52), (470, 52), (80, 80, 80), 2)
    
    # 分数显示（超紧凑布局）
    if scores is not None:
        labels_cn = ["语言表达", "心理素质", "肢体语言", "专注度", "综合得分"]
        colors_bgr = [(100, 255, 100), (100, 180, 255), (255, 100, 255), (255, 200, 100), (100, 255, 255)]
        
        for i, (label, score, color) in enumerate(zip(labels_cn, scores, colors_bgr)):
            y_start = 65 + i * 62  # 进一步减小间距：从72到62
            
            # 背景卡片（进一步减小高度）
            cv2.rectangle(panel, (20, y_start), (480, y_start + 58), (45, 45, 45), -1)
            cv2.rectangle(panel, (20, y_start), (480, y_start + 58), (70, 70, 70), 2)
            
            # 标签（中文，缩小字体）
            panel = cv2_add_chinese_text(panel, label, (35, y_start + 10), 18, (220, 220, 220))
            
            # 分数（缩小）
            score_text = f"{score:.1f}"
            cv2.putText(panel, score_text, (380, y_start + 36), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # 进度条背景
            cv2.rectangle(panel, (35, y_start + 44), (370, y_start + 55), (60, 60, 60), -1)
            
            # 进度条
            bar_width = int((score / 100.0) * 335)
            cv2.rectangle(panel, (35, y_start + 44), (35 + bar_width, y_start + 55), color, -1)
            
            # 进度条高光效果
            if bar_width > 10:
                highlight_color = tuple(min(255, c + 40) for c in color)
                cv2.rectangle(panel, (35, y_start + 44), (35 + bar_width, y_start + 49), highlight_color, -1)
    
    # 智能提醒区域（进一步缩小）
    y_reminder = h - 70
    cv2.rectangle(panel, (20, y_reminder), (480, h - 30), (50, 50, 70), -1)
    cv2.rectangle(panel, (20, y_reminder), (480, h - 30), (100, 100, 150), 2)
    
    # 提醒内容（合并标题和内容）
    reminder_display = f"💡 {reminder_text}"
    panel = cv2_add_chinese_text(panel, reminder_display, (30, y_reminder + 10), 16, (255, 255, 150))
    
    # FPS和状态（合并到一行）
    cv2.rectangle(panel, (20, h - 28), (480, h - 5), (40, 40, 40), -1)
    cv2.putText(panel, f"FPS: {fps:.1f}", (30, h - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
    
    # Buffer状态（右侧）
    status_text = f"E:{len(emotion_buffer)} A:{len(audio_buffer)} P:{len(pose_buffer)} G:{len(gaze_buffer)}"
    cv2.putText(panel, status_text, (250, h - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    
    # 拼接画面
    combined = np.hstack([frame, panel])
    return combined


# ==================== 主程序 ====================
def main():
    global is_running
    
    print("=" * 80)
    print("  AI Interview Scoring - REAL-TIME LIVE DEMO (Fixed)")
    print("=" * 80)
    print()
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 加载模型
    model = load_model(device)
    if model is None:
        print("[ERROR] Cannot start without model!")
        return
    
    # 打开摄像头（仅主线程）
    print("\nOpening camera (main thread only)...")
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_ID}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 预热摄像头
    print("Warming up camera...")
    for _ in range(10):
        cap.read()
    
    print("[OK] Camera ready")
    
    # 启动特征提取线程
    print("\nStarting feature extraction threads...")
    emotion_thread = EmotionExtractor(emotion_frame_queue)
    audio_thread = AudioExtractor()
    pose_thread = PoseGazeExtractor(pose_frame_queue)
    
    emotion_thread.start()
    audio_thread.start()
    pose_thread.start()
    
    # 主循环
    print("\nMain loop starting...")
    print("Press 'q' or ESC to quit\n")
    
    fps_counter = deque(maxlen=30)
    last_time = time.time()
    last_status_print = time.time()
    frame_count = 0
    
    try:
        while is_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Failed to grab frame")
                time.sleep(0.1)
                continue
            
            # 分发帧到提取线程
            if frame_count % 5 == 0:  # 每5帧分发一次
                try:
                    emotion_frame_queue.put_nowait(frame.copy())
                    pose_frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass
            
            # 收集特征
            try:
                while not emotion_queue.empty():
                    emotion_buffer.append(emotion_queue.get_nowait())
            except queue.Empty:
                pass
            
            try:
                while not audio_queue.empty():
                    audio_buffer.append(audio_queue.get_nowait())
            except queue.Empty:
                pass
            
            try:
                while not pose_gaze_queue.empty():
                    pose_feat, gaze_feat = pose_gaze_queue.get_nowait()
                    pose_buffer.append(pose_feat)
                    gaze_buffer.append(gaze_feat)
            except queue.Empty:
                pass
            
            # 定期打印状态（每3秒）
            if time.time() - last_status_print > 3.0:
                print(f"[Status] Buffers - E:{len(emotion_buffer)} A:{len(audio_buffer)} P:{len(pose_buffer)} G:{len(gaze_buffer)}/{WINDOW_SIZE}")
                last_status_print = time.time()
            
            # 推理（每10帧）
            if frame_count % 10 == 0 and len(emotion_buffer) >= WINDOW_SIZE:
                run_inference(model, device)
            
            # 计算FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            fps_counter.append(fps)
            avg_fps = np.mean(fps_counter)
            last_time = current_time
            
            # 绘制UI
            display_frame = draw_ui(frame, current_scores, current_reminder, avg_fps)
            
            # 显示
            cv2.imshow('AI Interview Scoring', display_frame)
            
            # 按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        print("\nShutting down...")
        is_running = False
        
        # 等待线程结束
        time.sleep(2)
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        
        print("[OK] Cleanup complete")


if __name__ == '__main__':
    main()

