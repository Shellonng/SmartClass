"""
重新训练Qwen-Decision模型（优化版）
更频繁的验证 + 更好的超参数
"""

import shutil
from pathlib import Path
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*60)
print("🔄 准备重新训练Qwen-Decision（优化版）")
print("="*60)

# 备份旧模型
old_dir = Path("checkpoints/qwen_decision_lora")
backup_dir = Path("checkpoints/qwen_decision_lora_backup_v1")

if old_dir.exists():
    print(f"\n📦 备份旧模型...")
    if backup_dir.exists():
        print(f"  删除旧备份...")
        shutil.rmtree(backup_dir)
    
    shutil.move(str(old_dir), str(backup_dir))
    print(f"  ✓ 已备份至: {backup_dir}")
else:
    print(f"\n⚠️  没有找到旧模型，将进行全新训练")

# 显示优化配置
print(f"\n📊 优化后的训练配置:")
print(f"{'='*60}")
print(f"  Epochs: 3 → 5 (更充分训练)")
print(f"  Learning Rate: 2e-4 → 1.5e-4 (更稳定收敛)")
print(f"  Eval Steps: 200 → 10 (密集监控)")
print(f"  Save Steps: 200 → 100 (更频繁保存)")
print(f"  Warmup Steps: 100 → 50 (更快学习)")
print(f"  Logging Steps: 10 → 5 (实时监控)")
print(f"{'='*60}")

print(f"\n🎯 优化目标:")
print(f"  ✓ 更低的Loss（目标 < 0.5）")
print(f"  ✓ 每10步验证，密切监控")
print(f"  ✓ 避免过拟合（监控train/eval gap）")
print(f"  ✓ 5个epochs充分学习")

print(f"\n⏱️  预计训练时间:")
print(f"  步数: 197步/epoch × 5 epochs = 985步")
print(f"  时间: 约5-6小时（比之前长1.5倍）")

print(f"\n🔍 验证频率:")
print(f"  总验证次数: 985步 ÷ 10 = 约98次")
print(f"  总checkpoint: 985步 ÷ 100 = 约10个")

print(f"\n💾 显存占用:")
print(f"  预计: 5-6GB (与之前相同)")

input(f"\n按Enter开始训练（或Ctrl+C取消）...")

print(f"\n🚀 启动训练...")
print(f"="*60)

import subprocess
result = subprocess.run(
    [r"E:\conda_envs\ai_interviewer\python.exe", "train_qwen_decision.py"],
    cwd=Path.cwd()
)

if result.returncode == 0:
    print(f"\n" + "="*60)
    print(f"✅ 训练完成！")
    print(f"="*60)
    print(f"\n查看结果:")
    print(f"  python analyze_training_results.py")
else:
    print(f"\n❌ 训练失败，返回码: {result.returncode}")


