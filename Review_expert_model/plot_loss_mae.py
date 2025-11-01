"""
从TensorBoard日志生成Loss和MAE曲线图
"""
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

def load_tensorboard_logs(log_dir):
    """从TensorBoard日志中提取数据"""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': steps, 'values': values}
    
    return data

def find_latest_run(runs_dir='runs'):
    """找到最新的训练运行"""
    all_runs = []
    for root, dirs, files in os.walk(runs_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                all_runs.append(root)
                break
    
    if not all_runs:
        return None
    
    all_runs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return all_runs[0]

def plot_loss_and_mae(data, save_path='training_loss_mae.png'):
    """绘制Loss和MAE曲线"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    fig.suptitle('AI Interview Scoring System - Training History', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    # 1. Loss曲线
    ax1 = axes[0]
    if 'Loss/train_total' in data and 'Loss/val_total' in data:
        train_loss = data['Loss/train_total']
        val_loss = data['Loss/val_total']
        
        ax1.plot(train_loss['steps'], train_loss['values'], 
                label='Training Loss', color='#1f77b4', linewidth=2.5, alpha=0.8)
        ax1.plot(val_loss['steps'], val_loss['values'], 
                label='Validation Loss', color='#ff7f0e', linewidth=2.5)
        
        initial_train = train_loss['values'][0]
        final_train = train_loss['values'][-1]
        initial_val = val_loss['values'][0]
        final_val = val_loss['values'][-1]
        
        reduction_train = (initial_train - final_train) / initial_train * 100
        reduction_val = (initial_val - final_val) / initial_val * 100
        
        ax1.set_title('Total Loss', fontsize=15, fontweight='bold', pad=15)
        ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 信息框 - 右上角
        info_text = (
            f'Initial: Train {initial_train:.1f} | Val {initial_val:.1f}\n'
            f'Final: Train {final_train:.1f} | Val {final_val:.1f}\n'
            f'Reduction: {reduction_train:.1f}% | {reduction_val:.1f}%'
        )
        
        ax1.text(0.98, 0.98, info_text,
                transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, pad=0.8))
        
        # 图例紧贴信息框下方
        ax1.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.0, 0.83), framealpha=0.9)
        
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
    
    # 2. MAE曲线
    ax2 = axes[1]
    if 'MAE/val' in data:
        mae = data['MAE/val']
        ax2.plot(mae['steps'], mae['values'], 
                color='#2ca02c', linewidth=2.5, marker='o', markersize=4)
        
        initial_mae = mae['values'][0]
        final_mae = mae['values'][-1]
        improvement = (initial_mae - final_mae) / initial_mae * 100
        
        ax2.set_title('Mean Absolute Error (MAE)', fontsize=15, fontweight='bold', pad=15)
        ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax2.set_ylabel('MAE (Score Points)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # 信息框 - 右上角
        info_text = (
            f'Initial: {initial_mae:.2f} pts | Final: {final_mae:.2f} pts\n'
            f'Improvement: {improvement:.1f}%\n'
            f'Relative Error: {final_mae:.2f}% (of 100 pts)'
        )
        
        ax2.text(0.98, 0.98, info_text,
                transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.85, pad=0.8))
        
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Training curves saved to: {save_path}")
    
    return save_path

def main():
    print("="*60)
    print("  Generating Loss & MAE Curves")
    print("="*60)
    print()
    
    # 找到最新的训练日志
    latest_run = find_latest_run()
    
    if not latest_run:
        print("[ERROR] No training logs found in 'runs/' directory")
        print("Please run train.py first to generate training data.")
        return
    
    print(f"[INFO] Loading logs from: {latest_run}")
    
    try:
        # 加载数据
        data = load_tensorboard_logs(latest_run)
        
        # 生成图表
        save_path = 'training_loss_mae.png'
        plot_loss_and_mae(data, save_path)
        
        print()
        print("="*60)
        print("  [OK] Plot generation complete!")
        print("="*60)
        print(f"\nView the plot: {save_path}")
        print()
        
        # 输出关键指标
        if 'Loss/val_total' in data and 'MAE/val' in data:
            final_loss = data['Loss/val_total']['values'][-1]
            final_mae = data['MAE/val']['values'][-1]
            print(f"Final Validation Loss: {final_loss:.2f}")
            print(f"Final MAE: {final_mae:.2f} points")
            print(f"Relative Error: {final_mae:.2f}%")
        
    except Exception as e:
        print(f"[ERROR] Failed to generate plot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

