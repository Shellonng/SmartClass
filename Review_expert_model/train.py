# -*- coding: utf-8 -*-
"""
Transformer模型训练脚本
支持4模态输入 + 5评分输出 + 提醒生成
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import argparse
from datetime import datetime

from model.transformer_model import InterviewTransformer, count_parameters, REMINDER_MAP
from data_loader_with_annotations import create_dataloaders_with_annotations


def train_epoch(model, train_loader, score_criterion, reminder_criterion, optimizer, device, epoch, 
                reminder_weight=0.3):
    """
    训练一个epoch
    
    Args:
        reminder_weight: 提醒损失的权重（默认0.3）
    """
    model.train()
    total_loss = 0
    total_score_loss = 0
    total_reminder_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # 移到设备（4个模态）
        emotion_seq = batch['emotion_seq'].to(device)
        audio_mel = batch['audio_mel'].to(device)
        pose_seq = batch['pose_seq'].to(device)
        gaze_seq = batch['gaze_seq'].to(device)  # ✨
        target_scores = batch['scores'].to(device)
        target_reminder = batch['reminder_class'].to(device)  # ✨
        
        # 前向传播
        predicted_scores, reminder_logits, _ = model(
            emotion_seq, audio_mel, pose_seq, gaze_seq
        )
        
        # 计算损失
        score_loss = score_criterion(predicted_scores, target_scores)
        reminder_loss = reminder_criterion(reminder_logits, target_reminder)  # ✨
        
        # 联合损失 ✨
        loss = score_loss + reminder_weight * reminder_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_score_loss += score_loss.item()
        total_reminder_loss += reminder_loss.item()
    
    avg_loss = total_loss / len(train_loader)
    avg_score_loss = total_score_loss / len(train_loader)
    avg_reminder_loss = total_reminder_loss / len(train_loader)
    
    return avg_loss, avg_score_loss, avg_reminder_loss


def validate(model, val_loader, score_criterion, reminder_criterion, device, reminder_weight=0.3):
    """
    验证
    
    Returns:
        avg_loss: 总损失
        avg_score_loss: 评分损失
        avg_reminder_loss: 提醒损失
        mae: 评分MAE
        reminder_acc: 提醒准确率
        predictions: 评分预测结果
        targets: 评分目标值
    """
    model.eval()
    total_loss = 0
    total_score_loss = 0
    total_reminder_loss = 0
    all_predictions = []
    all_targets = []
    all_reminder_preds = []
    all_reminder_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            emotion_seq = batch['emotion_seq'].to(device)
            audio_mel = batch['audio_mel'].to(device)
            pose_seq = batch['pose_seq'].to(device)
            gaze_seq = batch['gaze_seq'].to(device)  # ✨
            target_scores = batch['scores'].to(device)
            target_reminder = batch['reminder_class'].to(device)  # ✨
            
            # 前向传播
            predicted_scores, reminder_logits, _ = model(
                emotion_seq, audio_mel, pose_seq, gaze_seq
            )
            
            # 计算损失
            score_loss = score_criterion(predicted_scores, target_scores)
            reminder_loss = reminder_criterion(reminder_logits, target_reminder)
            loss = score_loss + reminder_weight * reminder_loss
            
            total_loss += loss.item()
            total_score_loss += score_loss.item()
            total_reminder_loss += reminder_loss.item()
            
            # 收集预测结果
            all_predictions.append(predicted_scores.cpu().numpy())
            all_targets.append(target_scores.cpu().numpy())
            
            # 提醒类别预测
            reminder_pred = torch.argmax(reminder_logits, dim=1)
            all_reminder_preds.append(reminder_pred.cpu().numpy())
            all_reminder_targets.append(target_reminder.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    avg_score_loss = total_score_loss / len(val_loader)
    avg_reminder_loss = total_reminder_loss / len(val_loader)
    
    # 计算评分MAE
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    mae = np.mean(np.abs(predictions - targets))
    
    # 计算提醒准确率 ✨
    reminder_preds = np.concatenate(all_reminder_preds, axis=0)
    reminder_targets = np.concatenate(all_reminder_targets, axis=0)
    reminder_acc = np.mean(reminder_preds == reminder_targets) * 100
    
    return avg_loss, avg_score_loss, avg_reminder_loss, mae, reminder_acc, predictions, targets


def main():
    parser = argparse.ArgumentParser(description='Train Interview Transformer')
    parser.add_argument('--feature_dir', type=str, default='./features',
                       help='Feature directory')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max samples for testing (None=all)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--d_model', type=int, default=128,
                       help='Model dimension')
    parser.add_argument('--nhead', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Model save directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 打印配置
    print("="*60)
    print("  Training Interview Transformer")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Max samples: {args.max_samples if args.max_samples else 'All'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Model dim: {args.d_model}")
    print(f"  Attention heads: {args.nhead}")
    print(f"  Encoder layers: {args.num_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Device: {args.device}")
    print(f"\n" + "="*60 + "\n")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建数据加载器（使用带标注版本）
    print("[*] Loading data...")
    train_loader, val_loader = create_dataloaders_with_annotations(
        feature_dir=args.feature_dir,
        annotation_dir='./annotations',
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        train_ratio=0.8
    )
    
    # 创建模型
    print("[*] Creating model...")
    model = InterviewTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        dropout=args.dropout
    )
    model = model.to(args.device)
    
    # 统计参数
    params = count_parameters(model)
    print(f"Model parameters: {params:,} ({params/1e6:.2f}M)\n")
    
    # 损失函数 ✨ 两个损失函数
    score_criterion = nn.MSELoss()  # 评分损失（回归）
    reminder_criterion = nn.CrossEntropyLoss()  # 提醒损失（分类）
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # TensorBoard
    log_dir = os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    
    # 训练
    print("="*60)
    print("  Starting Training")
    print("="*60 + "\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_score_loss, train_reminder_loss = train_epoch(
            model, train_loader, score_criterion, reminder_criterion, 
            optimizer, args.device, epoch, reminder_weight=0.3
        )
        
        # 验证
        val_loss, val_score_loss, val_reminder_loss, val_mae, reminder_acc, predictions, targets = validate(
            model, val_loader, score_criterion, reminder_criterion, args.device, reminder_weight=0.3
        )
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录到TensorBoard ✨ 更多指标
        writer.add_scalar('Loss/train_total', train_loss, epoch)
        writer.add_scalar('Loss/train_score', train_score_loss, epoch)
        writer.add_scalar('Loss/train_reminder', train_reminder_loss, epoch)
        writer.add_scalar('Loss/val_total', val_loss, epoch)
        writer.add_scalar('Loss/val_score', val_score_loss, epoch)
        writer.add_scalar('Loss/val_reminder', val_reminder_loss, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)
        writer.add_scalar('Accuracy/reminder', reminder_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # 打印进度 ✨
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} (Score: {train_score_loss:.4f}, Reminder: {train_reminder_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (Score: {val_score_loss:.4f}, Reminder: {val_reminder_loss:.4f})")
        print(f"  Val MAE: {val_mae:.2f}")
        print(f"  Reminder Acc: {reminder_acc:.1f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'reminder_acc': reminder_acc,
                'config': vars(args)
            }, save_path)
            
            print(f"  [BEST] Model saved to {save_path}")
        else:
            patience_counter += 1
        
        print()
        
        # Early stopping
        if patience_counter >= patience:
            print(f"[INFO] Early stopping triggered after {epoch+1} epochs")
            break
    
    # 训练完成
    print("="*60)
    print("  Training Completed!")
    print("="*60)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.save_dir}/best_model.pth")
    print(f"TensorBoard logs: {log_dir}")
    print(f"\nView training curves:")
    print(f"  tensorboard --logdir=runs")
    print()
    
    writer.close()


if __name__ == "__main__":
    main()



