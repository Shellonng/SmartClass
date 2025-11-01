# -*- coding: utf-8 -*-
"""
DeepFace情绪检测调试脚本
测试不同的detector backend和参数
环境: conda activate interview_emotion
"""
import cv2
import numpy as np
from deepface import DeepFace
import os

def test_deepface_backends(image_path):
    """测试不同的人脸检测器backend"""
    
    print("="*60)
    print("  DeepFace Backend Comparison")
    print("="*60 + "\n")
    
    # 读取图像
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Failed to load image")
        return
    
    print(f"Image shape: {img.shape}")
    print(f"Image size: {os.path.getsize(image_path)/1024:.1f} KB\n")
    
    # 测试不同的detector backend
    backends = ['opencv', 'ssd', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet', 'fastmtcnn']
    
    results = {}
    
    for backend in backends:
        print(f"[Testing] Backend: {backend}")
        try:
            result = DeepFace.analyze(
                img,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend=backend,
                silent=True
            )
            
            if result and len(result) > 0:
                emotions = result[0]['emotion']
                dominant = result[0]['dominant_emotion']
                region = result[0].get('region', {})
                
                print(f"  [SUCCESS] Detected!")
                print(f"    Dominant: {dominant}")
                print(f"    Face region: {region}")
                print(f"    Emotions: {emotions}")
                
                results[backend] = {
                    'status': 'success',
                    'dominant': dominant,
                    'emotions': emotions,
                    'region': region
                }
            else:
                print(f"  [WARN] No result returned")
                results[backend] = {'status': 'no_result'}
                
        except Exception as e:
            print(f"  [ERROR] {str(e)[:100]}")
            results[backend] = {'status': 'error', 'error': str(e)[:100]}
        
        print()
    
    # 总结
    print("="*60)
    print("  Summary")
    print("="*60 + "\n")
    
    successful = [k for k, v in results.items() if v['status'] == 'success']
    failed = [k for k, v in results.items() if v['status'] != 'success']
    
    print(f"Successful backends ({len(successful)}/{len(backends)}):")
    for backend in successful:
        print(f"  [OK] {backend}: {results[backend]['dominant']}")
    
    print(f"\nFailed backends ({len(failed)}/{len(backends)}):")
    for backend in failed:
        print(f"  [FAIL] {backend}")
    
    if successful:
        print(f"\n[RECOMMENDATION] Use backend: {successful[0]}")
        return successful[0]
    else:
        print(f"\n[ERROR] All backends failed!")
        return None


def extract_frame_from_video(video_path, frame_number=100):
    """从视频中提取一帧用于测试"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # 保存为图片
        output_path = 'test_frame.jpg'
        cv2.imwrite(output_path, frame)
        print(f"[OK] Extracted frame {frame_number} to {output_path}")
        return output_path
    else:
        print(f"[ERROR] Failed to extract frame")
        return None


def main():
    print("\n[*] DeepFace Emotion Detection Debugger")
    print("="*60 + "\n")
    
    # 从新视频提取一帧
    video_path = 'testv/d10342e332543abee0528c2396d79557.mp4'
    
    print(f"[*] Extracting test frame from: {video_path}\n")
    frame_path = extract_frame_from_video(video_path, frame_number=150)
    
    if frame_path:
        print(f"\n[*] Testing DeepFace with different backends...\n")
        best_backend = test_deepface_backends(frame_path)
        
        if best_backend:
            print(f"\n{'='*60}")
            print(f"  Best Backend Found: {best_backend}")
            print(f"{'='*60}")
            print(f"\nNext step:")
            print(f"  1. Update extract_emotion_features.py to use backend='{best_backend}'")
            print(f"  2. Re-extract emotion features")
            print(f"  3. Verify detection rate improved")
        else:
            print(f"\n{'='*60}")
            print(f"  Alternative Solutions Needed")
            print(f"{'='*60}")
            print(f"\nOptions:")
            print(f"  1. Try FER library (already installed)")
            print(f"  2. Use Hugging Face emotion classifier")
            print(f"  3. Record better quality videos")
            print(f"  4. Use only audio+pose features (2 modalities)")


if __name__ == "__main__":
    main()

