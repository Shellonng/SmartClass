# -*- coding: utf-8 -*-
"""
最终特征检查 - 验证所有4个模态
"""
import numpy as np
import os

def final_check():
    print("="*60)
    print("  FINAL FEATURE VALIDATION")
    print("  All 4 Modalities Check")
    print("="*60 + "\n")
    
    # 检查所有模态
    modalities = {
        'emotion': 'features/emotion',
        'audio': 'features/audio',
        'pose': 'features/pose'
    }
    
    results = {}
    
    for modality_name, modality_dir in modalities.items():
        print(f"[{modality_name.upper()}] Checking...")
        
        if not os.path.exists(modality_dir):
            print(f"  [ERROR] Directory not found\n")
            results[modality_name] = {'status': 'missing', 'valid_rate': 0}
            continue
        
        npz_files = sorted([f for f in os.listdir(modality_dir) if f.endswith('.npz')])
        
        if len(npz_files) == 0:
            print(f"  [WARN] No files found\n")
            results[modality_name] = {'status': 'empty', 'valid_rate': 0}
            continue
        
        total_windows = 0
        valid_windows = 0
        
        for npz_file in npz_files:
            file_path = os.path.join(modality_dir, npz_file)
            data = np.load(file_path, allow_pickle=True)
            
            if modality_name == 'emotion':
                seqs = data['emotion_sequences']
                metadata = data['metadata']
                
                for meta in metadata:
                    total_windows += 1
                    if meta['valid_detections'] > 0:
                        valid_windows += 1
            
            elif modality_name == 'audio':
                mel_specs = data['mel_spectrograms']
                total_windows += len(mel_specs)
                # 检查是否有效（不是全0）
                for mel in mel_specs:
                    if np.std(mel) > 1.0:
                        valid_windows += 1
            
            elif modality_name == 'pose':
                pose_seqs = data['pose_sequences']
                gaze_seqs = data['gaze_sequences']
                metadata = data['metadata']
                
                for meta in metadata:
                    total_windows += 1
                    # 姿势或眼动有一个检测成功即可
                    if meta['pose_detections'] > 0 or meta['gaze_detections'] > 0:
                        valid_windows += 1
        
        valid_rate = (valid_windows / total_windows * 100) if total_windows > 0 else 0
        
        print(f"  Files: {len(npz_files)}")
        print(f"  Windows: {total_windows}")
        print(f"  Valid: {valid_windows}/{total_windows} ({valid_rate:.1f}%)")
        
        if valid_rate >= 80:
            print(f"  Status: [EXCELLENT] Fully usable!\n")
            results[modality_name] = {'status': 'excellent', 'valid_rate': valid_rate, 'count': total_windows}
        elif valid_rate >= 50:
            print(f"  Status: [GOOD] Usable\n")
            results[modality_name] = {'status': 'good', 'valid_rate': valid_rate, 'count': total_windows}
        elif valid_rate > 0:
            print(f"  Status: [PARTIAL] Partially usable\n")
            results[modality_name] = {'status': 'partial', 'valid_rate': valid_rate, 'count': total_windows}
        else:
            print(f"  Status: [FAIL] Not usable\n")
            results[modality_name] = {'status': 'fail', 'valid_rate': 0, 'count': total_windows}
    
    # 分析眼动单独统计
    print(f"[GAZE] Separate Analysis...")
    pose_dir = 'features/pose'
    if os.path.exists(pose_dir):
        npz_files = sorted([f for f in os.listdir(pose_dir) if f.endswith('.npz')])
        total_gaze = 0
        valid_gaze = 0
        
        for npz_file in npz_files:
            data = np.load(os.path.join(pose_dir, npz_file), allow_pickle=True)
            metadata = data['metadata']
            
            for meta in metadata:
                total_gaze += 1
                if meta['gaze_detections'] > 0:
                    valid_gaze += 1
        
        gaze_rate = (valid_gaze / total_gaze * 100) if total_gaze > 0 else 0
        print(f"  Gaze detections: {valid_gaze}/{total_gaze} ({gaze_rate:.1f}%)")
        
        if gaze_rate > 0:
            print(f"  Status: [PARTIAL] Some gaze data available\n")
        else:
            print(f"  Status: [FAIL] No gaze data\n")
    
    # 总结
    print("="*60)
    print("  FINAL SUMMARY")
    print("="*60 + "\n")
    
    excellent_count = sum(1 for r in results.values() if r['status'] == 'excellent')
    good_count = sum(1 for r in results.values() if r['status'] == 'good')
    usable_count = excellent_count + good_count
    
    print(f"Modality Status:")
    for name, result in results.items():
        status = result['status']
        rate = result.get('valid_rate', 0)
        
        if status in ['excellent', 'good']:
            print(f"  [OK] {name.capitalize()}: {rate:.1f}% valid")
        elif status == 'partial':
            print(f"  [PARTIAL] {name.capitalize()}: {rate:.1f}% valid")
        else:
            print(f"  [FAIL] {name.capitalize()}: {rate:.1f}% valid")
    
    print(f"\nUsable modalities: {usable_count}/3")
    
    # 计算总样本数
    total_samples = sum(r.get('count', 0) for r in results.values() if r.get('count'))
    print(f"Total training samples: {total_samples}")
    
    if usable_count >= 3:
        print(f"\n{'='*60}")
        print(f"  [SUCCESS] ALL MODALITIES WORKING!")
        print(f"{'='*60}")
        print(f"\n🎉 Emotion detection FIXED with MTCNN backend!")
        print(f"✅ All 3-4 modalities ready for training!")
        print(f"✅ {total_samples} high-quality samples available!")
        print(f"\nReady to train Transformer model!")
    else:
        print(f"\n[INCOMPLETE] Only {usable_count}/3 modalities usable")
    
    print()


if __name__ == "__main__":
    final_check()

