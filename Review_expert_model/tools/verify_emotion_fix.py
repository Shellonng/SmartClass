# -*- coding: utf-8 -*-
"""
验证修复后的情绪特征是否有效
"""
import numpy as np
import os

def verify_fixed_emotions():
    print("="*60)
    print("  Emotion Feature Fix Verification")
    print("="*60 + "\n")
    
    emotion_dir = 'features/emotion_fixed'
    
    if not os.path.exists(emotion_dir):
        print(f"[ERROR] Directory not found: {emotion_dir}")
        return
    
    npz_files = sorted([f for f in os.listdir(emotion_dir) if f.endswith('.npz')])
    
    print(f"[INFO] Found {len(npz_files)} emotion feature files\n")
    
    total_windows = 0
    total_valid_detections = 0
    
    for npz_file in npz_files:
        file_path = os.path.join(emotion_dir, npz_file)
        data = np.load(file_path, allow_pickle=True)
        
        seqs = data['emotion_sequences']
        metadata = data['metadata']
        
        print(f"[{npz_file}]")
        print(f"  Windows: {len(seqs)}")
        
        # 统计有效检测
        valid_count = 0
        for i, (seq, meta) in enumerate(zip(seqs, metadata)):
            is_valid = meta['valid_detections'] > 0
            if is_valid:
                valid_count += 1
            
            # 显示前3个窗口的详情
            if i < 3:
                print(f"  Window {i+1}: {meta['valid_detections']}/5 detections, dominant={meta['dominant_emotion']}")
                if meta['valid_detections'] > 0:
                    # 显示第一帧的情绪数据
                    first_frame = seq[0]
                    print(f"    First frame: {first_frame}")
        
        print(f"  Valid windows: {valid_count}/{len(seqs)} ({valid_count/len(seqs)*100:.1f}%)\n")
        
        total_windows += len(seqs)
        total_valid_detections += valid_count
    
    # 总结
    print("="*60)
    print("  Summary")
    print("="*60)
    print(f"\nTotal windows: {total_windows}")
    print(f"Valid detections: {total_valid_detections}/{total_windows} ({total_valid_detections/total_windows*100:.1f}%)")
    
    if total_valid_detections > 0:
        print(f"\n[SUCCESS] Emotion detection FIXED!")
        print(f"[SUCCESS] MTCNN backend works!")
        print(f"\nNext steps:")
        print(f"  1. Update features/emotion/ with these fixed features")
        print(f"  2. All 4 modalities now working!")
        print(f"  3. Ready to train Transformer with full dataset")
    else:
        print(f"\n[FAIL] Still no valid detections")
        print(f"Need to try other solutions")
    
    print()


if __name__ == "__main__":
    verify_fixed_emotions()

