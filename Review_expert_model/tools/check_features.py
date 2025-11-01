# -*- coding: utf-8 -*-
"""
特征质量检查脚本
检查提取的特征是否完整和正确
"""
import numpy as np
import os
import json

def check_features():
    """检查所有提取的特征"""
    print("="*60)
    print("  Feature Quality Check")
    print("="*60 + "\n")
    
    feature_dir = './features'
    
    if not os.path.exists(feature_dir):
        print("[ERROR] Features directory not found!")
        print(f"Expected: {feature_dir}")
        print("Please run extract_all_features.bat first")
        return
    
    # 检查各个模态
    modalities = ['emotion', 'audio', 'pose']
    results = {}
    
    for modality in modalities:
        print(f"[{modality.upper()}] Checking...")
        modality_dir = os.path.join(feature_dir, modality)
        
        if not os.path.exists(modality_dir):
            print(f"  [WARN] Directory not found: {modality_dir}")
            results[modality] = {'status': 'missing', 'count': 0}
            continue
        
        # 检查.npz文件
        npz_files = [f for f in os.listdir(modality_dir) if f.endswith('.npz')]
        
        if len(npz_files) == 0:
            print(f"  [WARN] No .npz files found")
            results[modality] = {'status': 'empty', 'count': 0}
            continue
        
        print(f"  Found {len(npz_files)} feature files")
        
        # 检查每个文件
        file_info = []
        for npz_file in sorted(npz_files):
            file_path = os.path.join(modality_dir, npz_file)
            
            try:
                data = np.load(file_path, allow_pickle=True)
                
                # 根据模态检查不同的数据
                if modality == 'emotion':
                    sequences = data['emotion_sequences']
                    metadata = data['metadata']
                    shape_info = f"{len(sequences)} windows"
                    
                elif modality == 'audio':
                    mel_specs = data['mel_spectrograms']
                    metadata = data['metadata']
                    shape_info = f"{len(mel_specs)} windows"
                    
                elif modality == 'pose':
                    pose_seqs = data['pose_sequences']
                    gaze_seqs = data['gaze_sequences']
                    metadata = data['metadata']
                    shape_info = f"{len(pose_seqs)} windows"
                
                file_info.append({
                    'file': npz_file,
                    'shape': shape_info,
                    'status': 'OK'
                })
                
                print(f"    [OK] {npz_file}: {shape_info}")
                
            except Exception as e:
                print(f"    [ERR] {npz_file}: ERROR - {e}")
                file_info.append({
                    'file': npz_file,
                    'status': 'ERROR',
                    'error': str(e)
                })
        
        # 检查索引文件
        index_file = os.path.join(modality_dir, f'{modality}_index.json')
        if os.path.exists(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            print(f"  Index file: {len(index_data)} entries")
        else:
            print(f"  [WARN] Index file not found")
        
        results[modality] = {
            'status': 'ok',
            'count': len(npz_files),
            'files': file_info
        }
        print()
    
    # 总结
    print("="*60)
    print("  Summary")
    print("="*60)
    
    total_ok = sum(1 for r in results.values() if r.get('status') == 'ok')
    
    for modality, result in results.items():
        status = result.get('status', 'unknown')
        count = result.get('count', 0)
        
        if status == 'ok':
            print(f"  [OK] {modality.capitalize()}: {count} files")
        elif status == 'missing':
            print(f"  [MISS] {modality.capitalize()}: Directory missing")
        elif status == 'empty':
            print(f"  [EMPTY] {modality.capitalize()}: No files")
        else:
            print(f"  [?] {modality.capitalize()}: Unknown status")
    
    print()
    
    if total_ok == len(modalities):
        print("[SUCCESS] All features extracted successfully!")
        print("\nReady for Transformer training")
        print("Next step: python train_transformer.py")
    else:
        print(f"[INCOMPLETE] Only {total_ok}/{len(modalities)} modalities ready")
        print("\nMissing modalities need to be extracted")
        print("Run: extract_all_features.bat")
    
    print()
    
    # 显示特征示例
    if total_ok > 0:
        print("="*60)
        print("  Feature Example (first file)")
        print("="*60)
        
        for modality in modalities:
            if results[modality].get('status') == 'ok':
                modality_dir = os.path.join(feature_dir, modality)
                npz_files = sorted([f for f in os.listdir(modality_dir) if f.endswith('.npz')])
                
                if npz_files:
                    first_file = os.path.join(modality_dir, npz_files[0])
                    data = np.load(first_file, allow_pickle=True)
                    
                    print(f"\n[{modality.upper()}] {npz_files[0]}")
                    
                    if modality == 'emotion':
                        seqs = data['emotion_sequences']
                        metadata = data['metadata']
                        print(f"  Shape: {len(seqs)} windows")
                        if len(seqs) > 0 and len(seqs[0]) > 0:
                            print(f"  First window: {seqs[0][0].shape} (frames, 7 emotions)")
                        print(f"  Sample metadata: {metadata[0] if metadata else 'None'}")
                    
                    elif modality == 'audio':
                        mel_specs = data['mel_spectrograms']
                        metadata = data['metadata']
                        print(f"  Shape: {len(mel_specs)} windows")
                        if len(mel_specs) > 0:
                            print(f"  First window: {mel_specs[0].shape} (80 mel bands)")
                        if metadata:
                            print(f"  Transcription: {metadata[0].get('transcription', 'N/A')[:50]}...")
                    
                    elif modality == 'pose':
                        pose_seqs = data['pose_sequences']
                        gaze_seqs = data['gaze_sequences']
                        metadata = data['metadata']
                        print(f"  Pose shape: {len(pose_seqs)} windows")
                        print(f"  Gaze shape: {len(gaze_seqs)} windows")
                        if len(pose_seqs) > 0 and len(pose_seqs[0]) > 0:
                            print(f"  First window pose: {pose_seqs[0][0].shape} (frames, 99 coords)")
                        if len(gaze_seqs) > 0 and len(gaze_seqs[0]) > 0:
                            print(f"  First window gaze: {gaze_seqs[0][0].shape} (frames, 5 features)")
        
        print()


if __name__ == "__main__":
    check_features()

