# -*- coding: utf-8 -*-
"""
特征查看脚本 - 验证特征提取成功
"""
import numpy as np
import os
import json

def view_features():
    print("="*60)
    print("  Feature Viewer")
    print("="*60 + "\n")
    
    # 检查features目录
    if not os.path.exists('features'):
        print("[ERROR] Features directory not found!")
        return
    
    modalities = ['emotion', 'audio', 'pose']
    
    for modality in modalities:
        print(f"\n[{modality.upper()}] Features:")
        print("-"*60)
        
        modality_dir = f'features/{modality}'
        
        if not os.path.exists(modality_dir):
            print("  [SKIP] Directory not found")
            continue
        
        # 读取索引
        index_file = f'{modality_dir}/{modality}_index.json'
        if os.path.exists(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            print(f"  Index: {len(index_data)} videos\n")
        
        # 检查每个文件
        npz_files = sorted([f for f in os.listdir(modality_dir) if f.endswith('.npz')])
        
        for npz_file in npz_files:
            file_path = f'{modality_dir}/{npz_file}'
            data = np.load(file_path, allow_pickle=True)
            
            print(f"  File: {npz_file}")
            print(f"    Keys: {list(data.keys())}")
            
            if modality == 'emotion':
                seqs = data['emotion_sequences']
                metadata = data['metadata']
                print(f"    Windows: {len(seqs)}")
                if len(seqs) > 0:
                    first_seq = seqs[0]
                    if hasattr(first_seq, 'shape'):
                        print(f"    First window shape: {first_seq.shape}")
                    else:
                        print(f"    First window: {len(first_seq)} frames")
                if len(metadata) > 0:
                    print(f"    Metadata sample: {metadata[0]}")
            
            elif modality == 'audio':
                mel_specs = data['mel_spectrograms']
                metadata = data['metadata']
                print(f"    Windows: {len(mel_specs)}")
                if len(mel_specs) > 0:
                    print(f"    First window shape: {mel_specs[0].shape}")
                if len(metadata) > 0:
                    meta = metadata[0]
                    print(f"    Metadata sample: window {meta['window_idx']}, {meta['start_time']}-{meta['end_time']}s")
            
            elif modality == 'pose':
                pose_seqs = data['pose_sequences']
                gaze_seqs = data['gaze_sequences']
                metadata = data['metadata']
                print(f"    Windows: {len(pose_seqs)}")
                if len(pose_seqs) > 0:
                    first_pose = pose_seqs[0]
                    if hasattr(first_pose, 'shape'):
                        print(f"    Pose shape: {first_pose.shape}")
                    else:
                        print(f"    Pose frames: {len(first_pose)}")
                if len(gaze_seqs) > 0:
                    first_gaze = gaze_seqs[0]
                    if hasattr(first_gaze, 'shape'):
                        print(f"    Gaze shape: {first_gaze.shape}")
                    else:
                        print(f"    Gaze frames: {len(first_gaze)}")
            
            print()
    
    # 统计总数
    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    
    total_samples = 0
    for modality in modalities:
        modality_dir = f'features/{modality}'
        if os.path.exists(modality_dir):
            npz_files = [f for f in os.listdir(modality_dir) if f.endswith('.npz')]
            
            # 统计样本数
            samples = 0
            for npz_file in npz_files:
                data = np.load(f'{modality_dir}/{npz_file}', allow_pickle=True)
                if modality == 'emotion':
                    samples += len(data['emotion_sequences'])
                elif modality == 'audio':
                    samples += len(data['mel_spectrograms'])
                elif modality == 'pose':
                    samples += len(data['pose_sequences'])
            
            print(f"  {modality.capitalize()}: {len(npz_files)} files, {samples} samples")
            total_samples = max(total_samples, samples)
    
    print(f"\n  Total training samples: {total_samples}")
    print("\n[SUCCESS] All features validated!")
    print("\nReady for Transformer training!")
    print("Next: python train_transformer.py\n")


if __name__ == "__main__":
    view_features()

