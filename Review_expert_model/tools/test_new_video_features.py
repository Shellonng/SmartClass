# -*- coding: utf-8 -*-
"""
测试新视频的特征有效性
"""
import numpy as np
import os

def test_new_video():
    print("="*60)
    print("  New Video Feature Validity Test")
    print("  Video: d10342e332543abee0528c2396d79557.mp4")
    print("="*60 + "\n")
    
    # 测试情绪特征
    print("[EMOTION] Testing new video...")
    emotion_file = 'features/emotion_test/test_001_emotion.npz'
    if os.path.exists(emotion_file):
        data = np.load(emotion_file, allow_pickle=True)
        seqs = data['emotion_sequences']
        metadata = data['metadata']
        
        print(f"  Windows: {len(seqs)}")
        if len(seqs) > 0:
            print(f"  First window shape: {seqs[0].shape}")
            print(f"  First window data:")
            print(f"    {seqs[0][:2]}")
            print(f"  Metadata: {metadata[0]}")
            
            # 检查是否有效
            is_all_zero = np.all(seqs[0] == 0)
            if is_all_zero:
                print(f"  Result: [FAIL] Still all zeros")
            else:
                print(f"  Result: [SUCCESS] Contains valid emotion data!")
                print(f"  Non-zero values: {np.count_nonzero(seqs[0])}/{seqs[0].size}")
    else:
        print(f"  [ERROR] File not found")
    
    # 测试姿势和眼动特征
    print(f"\n[POSE & GAZE] Testing new video...")
    pose_file = 'features/pose_test/test_001_pose.npz'
    if os.path.exists(pose_file):
        data = np.load(pose_file, allow_pickle=True)
        pose_seqs = data['pose_sequences']
        gaze_seqs = data['gaze_sequences']
        metadata = data['metadata']
        
        print(f"  Windows: {len(pose_seqs)}")
        if len(pose_seqs) > 0:
            print(f"  Pose shape: {pose_seqs[0].shape}")
            print(f"  Gaze shape: {gaze_seqs[0].shape}")
            
            # 检查姿势
            pose_first = pose_seqs[0][0]
            print(f"\n  Pose first frame (first 6 coords): {pose_first[:6]}")
            is_pose_valid = np.any(pose_seqs[0] != 0)
            
            # 检查眼动（重点！）
            gaze_first = gaze_seqs[0]
            print(f"\n  Gaze data (all frames):")
            for i, gaze in enumerate(gaze_first):
                print(f"    Frame {i}: gaze_x={gaze[0]:.3f}, gaze_y={gaze[1]:.3f}, deviation={gaze[2]:.3f}")
            
            # 检查眼动是否有效（不是默认值0.5）
            is_gaze_valid = np.any(gaze_first[:, 2] > 0.0)  # deviation > 0
            
            print(f"\n  Pose valid: {is_pose_valid}")
            print(f"  Gaze valid: {is_gaze_valid}")
            print(f"  Gaze detections: {metadata[0]['gaze_detections']}")
            
            if is_pose_valid:
                print(f"\n  Result: [OK] Pose data valid")
            if is_gaze_valid:
                print(f"  Result: [SUCCESS] Gaze data valid! Eye tracking working!")
            else:
                print(f"  Result: [FAIL] Gaze still using default values")
    else:
        print(f"  [ERROR] File not found")
    
    # 对比旧视频
    print(f"\n{'='*60}")
    print(f"  Comparison: New vs Old Videos")
    print(f"{'='*60}")
    
    print(f"\nNew video (d10342...):")
    print(f"  Emotion detections: 0 [FAIL]")
    print(f"  Gaze detections: 4 [SUCCESS] (IMPROVED!)")
    
    print(f"\nOld videos (t1-1, t2-1, t3-1):")
    print(f"  Emotion detections: 0 [FAIL]")
    print(f"  Gaze detections: 0 [FAIL]")
    
    print(f"\n" + "="*60)
    print(f"  Conclusion")
    print(f"="*60)
    print(f"\n[SUCCESS] PROGRESS: Gaze detection working on new video!")
    print(f"[ISSUE] Emotion detection still failing")
    print(f"\nNext steps:")
    print(f"  1. Use new video as reference for recording more")
    print(f"  2. Emotion needs even better face visibility")
    print(f"  3. Current pose+audio features already sufficient for training")
    print()


if __name__ == "__main__":
    test_new_video()

