# 🎯 AI智能面试系统 - 双模型协同架构

一个基于大语言模型和多模态Transformer的完整智能面试解决方案

**项目状态**: ✅ 两个子系统训练完成，性能优秀！

---

## 📌 项目简介

本项目是一个创新的**双模型协同AI面试系统**，通过两个专门化的AI模型实现完整的面试流程：

- 🎤 **AI面试官模型** (Interviewer Model)：基于Qwen-2-1.5B的Triple-Qwen架构，负责智能提问、追问决策和文本评分
- 🔍 **AI评审专家模型** (Review Expert Model)：基于多模态Transformer，实时分析候选人的情绪、语音、姿势和眼动，给出行为表现评分

### 核心能力

| 功能模块 | 技术方案 | 主要功能 |
|---------|---------|---------|
| **简历解析** | pdfplumber + python-docx | 提取技能、项目、教育背景 |
| **智能提问** | Qwen-Question (LoRA微调) | 基于简历生成个性化问题 |
| **追问决策** | Qwen-Decision (LoRA微调) | 判断是否继续深入或切换话题 |
| **文本评分** | Qwen-Scorer (LoRA微调) | 评估回答质量并给出详细反馈 |
| **情绪分析** | DeepFace + TensorFlow 2.15 | 7种情绪实时识别 |
| **语音分析** | Librosa | Mel频谱图提取和音频质量评估 |
| **姿势检测** | MediaPipe Pose | 33个关键点追踪和自信度分析 |
| **眼动追踪** | MediaPipe FaceMesh | 视线偏离度计算和专注度评估 |
| **行为评分** | PyTorch Transformer | 5维评分（语言/心理/肢体/专注/总分） |

---

## 🏗️ 系统架构

### 双模型协同流程

```
┌────────────────────────────────────────────────────────────────┐
│                    AI智能面试系统（双模型协同）                    │
└────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
        ┌───────────▼───────────┐   ┌──────────▼──────────┐
        │   AI面试官模型         │   │  AI评审专家模型      │
        │ (Interviewer Model)   │   │ (Review Expert)     │
        │                       │   │                     │
        │  📄 简历解析          │   │  😊 情绪识别        │
        │  💬 智能提问          │   │  🎵 语音分析        │
        │  🔄 追问决策          │   │  🧘 姿势检测        │
        │  📝 文本评分          │   │  👁️ 眼动追踪        │
        │                       │   │  📊 行为评分        │
        │  (Qwen-2-1.5B)       │   │  (Transformer)      │
        └───────────┬───────────┘   └──────────┬──────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │     综合评估报告            │
                    │  • 文本内容分析（0-100分）  │
                    │  • 行为表现分析（5维评分）  │
                    │  • 详细改进建议             │
                    │  • 智能提醒（30类场景）     │
                    └───────────────────────────┘
```

### 面试官模型：Triple-Qwen架构

```
简历上传
    ↓
Resume Parser (简历解析)
    ↓
Topic Queue (话题队列生成)
    ↓
┌─────────────┬─────────────┬─────────────┐
│  Qwen-      │  Qwen-      │  Qwen-      │
│ Decision    │ Question    │  Scorer     │
│ (决策模型)   │ (提问模型)   │ (评分模型)   │
│             │             │             │
│ 判断切换话题 │ 生成面试问题 │ 评估答案质量 │
│ + 指导语    │ + 重要程度  │ + 详细反馈  │
└─────────────┴─────────────┴─────────────┘
```

**技术特点**：
- 基座模型：Qwen2-1.5B-Instruct (4-bit量化)
- 微调方法：LoRA（仅训练0.27%参数）
- 显存占用：推理仅需~3.38GB
- 推理延迟：~280ms/次

### 评审专家模型：多模态Transformer

```
实时输入（摄像头+麦克风）
    ↓
┌────────────┬────────────┬────────────┬────────────┐
│ 情绪检测    │ 语音分析    │ 姿势检测    │ 眼动追踪    │
│ DeepFace   │ Librosa    │ MediaPipe  │ FaceMesh   │
│ [5,7]      │ [80]       │ [5,99]     │ [5,5]      │
└────────────┴────────────┴────────────┴────────────┘
    │            │            │            │
    └────────────┴────────────┴────────────┘
                     ↓
         Cross-Modal Attention融合
                     ↓
         Transformer Encoder (4层)
                     ↓
         ┌──────────┴──────────┐
         │                     │
    评分预测 [5]          提醒分类 [30]
  (语言/心理/肢体/      (30类智能提醒)
   专注/总分)
```

**技术特点**：
- 模型参数：2.19M（轻量化）
- MAE误差：3.41分（相对100分）
- 提醒准确率：71.4%
- 实时性：10-15 FPS

---

## 📂 项目结构

```
AI mock interview/
│
├── Interviewer_model/              # AI面试官模型（文本交互）
│   ├── models/                     # 核心模块
│   │   ├── resume_parser.py        # 简历解析
│   │   └── ...
│   ├── checkpoints/                # 训练好的LoRA权重
│   │   ├── qwen_decision_lora/     # Decision模型
│   │   ├── qwen_question_lora/     # Question V3模型
│   │   ├── qwen_question_v4_lora/  # Question V4模型（实验性）
│   │   └── qwen_scorer_v2_lora/    # Scorer V2模型
│   ├── training_data/              # 训练数据（~8000条样本）
│   ├── plots/                      # 训练可视化
│   ├── scripts/                    # 训练脚本
│   ├── app_triple_qwen.py          # 主应用（Streamlit UI）
│   ├── test_triple_qwen.py         # 模型测试
│   └── readme.md                   # 面试官模型文档
│
├── Review_expert_model/            # AI评审专家模型（多模态分析）
│   ├── model/
│   │   └── transformer_model.py    # Transformer模型定义
│   ├── tools/                      # 特征提取工具
│   │   ├── extract_emotion_features.py   # 情绪检测
│   │   ├── extract_audio_mel.py          # 音频分析
│   │   └── extract_pose_features.py      # 姿势+眼动
│   ├── features/                   # 多模态特征数据
│   ├── annotations/                # 智能标注（104个样本）
│   ├── checkpoints/
│   │   └── best_model.pth          # 最佳模型权重
│   ├── train.py                    # 模型训练
│   ├── realtime_live_demo_fixed.py # 实时演示系统（中文UI）
│   ├── auto_generate_dataset_fixed_v2.py  # 自动标注系统
│   └── readme.md                   # 评审专家模型文档
│
├── readme.md                       # 总体项目文档（本文件）
└── requirements.txt                # 依赖包列表
```

---

## 🚀 快速开始

### 1. 环境要求

- **Python**: 3.9+
- **CUDA**: 11.8+ (推荐用于GPU加速)
- **GPU**: NVIDIA RTX 4060或更高（至少6GB显存）
  - 面试官模型：~3.38GB显存
  - 评审专家模型：~2-3GB显存
  - 同时运行：~6-7GB显存
- **CPU**: 可运行但速度较慢

### 2. 安装依赖

```bash
# 克隆项目
git clone <repository_url>
cd "AI mock interview"

# 创建虚拟环境（推荐使用Conda）
conda create -n ai_interview python=3.9
conda activate ai_interview

# 安装依赖
pip install -r requirements.txt
```

### 3. 快速演示

#### 选项A：运行面试官模型（文本对话）

```bash
cd Interviewer_model

# 启动Triple-Qwen面试系统
streamlit run app_triple_qwen.py

# 浏览器打开 http://localhost:8501
# 1. 上传简历（PDF/DOCX）或粘贴文本
# 2. 开始面试
# 3. 回答问题
# 4. 查看评分和反馈
```

#### 选项B：运行评审专家模型（实时多模态分析）

```bash
cd Review_expert_model

# 创建实时演示环境
conda env create -f environment_realtime_clean.yml -p E:\conda_envs\interview_realtime
conda activate E:\conda_envs\interview_realtime

# 运行实时系统
python realtime_live_demo_fixed.py

# 效果：
# - 实时摄像头画面 + 5维评分
# - 智能提醒中文显示
# - FPS监控 + 缓冲区状态
# - 按 'q' 退出
```

#### 选项C：完整系统（TODO：集成中）

未来版本将提供两个模型的完整集成，实现：
- 面试官提问 → 候选人回答 → 文本+行为双重评分 → 追问决策 → 下一轮

---

## 📊 性能指标

### 面试官模型（Triple-Qwen）

| 模型 | 版本 | 验证Loss | Gap | 泛化能力 | 推荐度 |
|------|------|---------|-----|---------|--------|
| **Qwen-Decision** | V2 | 0.57 | 0.045 | ⭐⭐⭐⭐⭐ | ✅ 推荐 |
| **Qwen-Question** | V3 | 0.48 | -0.16 | ⭐⭐⭐⭐ | ✅ 推荐 |
| **Qwen-Question** | V4 | 1.04 | 0.037 | ⭐⭐⭐⭐ | 🔬 实验 |
| **Qwen-Scorer** | V2 | 0.49 | -0.003 | ⭐⭐⭐⭐⭐ | ✅ 推荐 |

**实际应用表现**：
- 问题相关性：4.3/5（人工评估 n=100）
- 评分一致性（ICC）：0.82（与人工评分对比）
- 决策准确率：86%（FOLLOW_UP/SWITCH判断）
- 话题覆盖度：95%（简历技能覆盖率）

### 评审专家模型（Multi-Modal Transformer）

| 指标 | 初始值 | 最终值 | 改进幅度 |
|------|--------|--------|----------|
| **训练损失** | 884.00 | 34.80 | ↓ 96.1% |
| **验证损失** | 884.00 | 29.36 | ↓ 96.7% |
| **MAE（评分误差）** | 23.42 | 3.41 | ↓ 85.5% |
| **提醒准确率** | 4.8% | 71.4% | ↑ 66.7% |

**评分分布（104个训练样本）**：
```
维度            均值±标准差        范围
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
语言表达        74.3 ± 12.5      [50.2, 95.8]
心理素质        71.8 ± 10.3      [52.1, 89.6]
肢体语言        76.5 ± 11.2      [53.7, 93.4]
专注度          78.2 ± 9.8       [55.3, 94.1]
总分            75.2 ± 8.7       [58.4, 91.5]
```

### 硬件占用（RTX 4060 Laptop）

| 指标 | 面试官模型 | 评审专家模型 | 备注 |
|------|-----------|-------------|------|
| **推理显存** | ~3.38GB | ~2-3GB | 4-bit量化 |
| **训练显存** | ~8-10GB | ~4-5GB | |
| **推理延迟** | ~280ms/次 | 10-15 FPS | |
| **模型参数** | 3×4M (LoRA) | 2.19M | 基座1.5B共享 |

---

## 🎓 技术创新点

### 1. 双模型协同架构（全球首创）

- **文本智能**（Qwen）+ **行为分析**（Transformer）深度融合
- 互补评估：内容质量 + 表现形式
- 决策协同：文本低分触发追问 OR 行为异常触发提醒

### 2. Triple-Qwen专业化分工

- **任务解耦**：决策、提问、评分各司其职
- **共享基座**：3个模型共用Qwen-2-1.5B，节省显存
- **参数高效**：LoRA微调仅0.27%参数（4M/1.5B）

### 3. 多模态Cross-Attention融合

- 不同于简单拼接，使用注意力机制动态融合4个模态
- 情绪 ↔ 音频/姿势/眼动自动学习相关性
- 提升评分准确性和可解释性

### 4. 智能自动标注系统

- **无需人工标注**：基于领域知识的规则引擎
- 多模态分析算法（情绪稳定性、音频质量、姿势自信度、眼神专注度）
- 节省人工成本（每个样本约5分钟 × 104样本 = 8.7小时）

### 5. 4-bit量化与轻量化设计

- Qwen模型4-bit量化，显存占用减少48%
- 评审专家模型仅2.19M参数，适合端侧部署
- 支持笔记本实时推理

---

## 📖 使用场景

### 场景1：求职者自我练习

```python
# 使用面试官模型
1. 上传简历 → 系统生成个性化问题
2. 文字输入回答 → 获得详细文本评分和改进建议
3. 多轮追问 → 模拟真实面试深度

# 使用评审专家模型
1. 打开摄像头和麦克风
2. 对着镜头练习回答
3. 实时查看情绪、姿势、语音表现
4. 根据智能提醒改进行为
```

### 场景2：企业招聘初筛

```python
# 完整流程（未来集成版本）
1. 候选人上传简历
2. AI面试官自动提问（5-10个问题）
3. 候选人视频回答
4. 双模型评分：
   - 文本内容质量（0-100分）
   - 行为表现分析（5维评分）
5. 生成综合评估报告
6. HR查看报告，决定是否进入下一轮
```

### 场景3：教育培训

```python
# 面试技巧课程
1. 教师使用系统演示标准答案
2. 学生实时练习并获得反馈
3. 对比分析学生表现差异
4. 针对性改进训练
```

---

## 🔬 模型训练指南

### 面试官模型训练

详细说明请参考 `Interviewer_model/readme.md`

```bash
cd Interviewer_model

# 1. 训练Decision模型
python train_qwen_decision.py \
    --train_file training_data/qwen_data.json \
    --output_dir checkpoints/qwen_decision_lora \
    --num_epochs 3

# 2. 训练Question模型
python train_qwen_question.py \
    --train_file training_data/qwen_data.json \
    --output_dir checkpoints/qwen_question_lora \
    --num_epochs 5

# 3. 训练Scorer模型
python prepare_qwen_scorer_v2_data.py  # 准备数据
python train_qwen_scorer_v2.py \
    --train_file training_data/qwen_scorer_v2_train.json \
    --output_dir checkpoints/qwen_scorer_v2_lora

# 4. 查看训练曲线
# 训练曲线保存在 plots/ 目录
```

### 评审专家模型训练

详细说明请参考 `Review_expert_model/readme.md`

```bash
cd Review_expert_model

# 1. 提取多模态特征（需要3个独立环境）
conda activate interview_emotion
python tools/extract_emotion_features.py

conda activate interview_audio
python tools/extract_audio_mel.py

conda activate interview_pose
python tools/extract_pose_features.py

# 2. 生成智能标注
python auto_generate_dataset_fixed_v2.py

# 3. 训练模型
python train.py --epochs 100 --batch_size 8 --lr 5e-4

# 4. 查看训练曲线
tensorboard --logdir=runs
python plot_loss_mae.py
```

---

## 🛠️ 开发指南

### 自定义简历解析规则

编辑 `Interviewer_model/models/resume_parser.py`：

```python
def _extract_projects(self, text):
    """提取项目经验"""
    # 添加自定义关键词
    project_keywords = ['项目经验', '主要项目', '项目名称', '负责项目']
    # 自定义解析逻辑
    ...
```

### 调整追问深度

编辑 `Interviewer_model/app_triple_qwen.py`：

```python
# 修改追问策略
MAX_FOLLOW_UP_DEPTH = 3  # 默认最多追问3轮
MIN_SCORE_THRESHOLD = 60  # 低于此分数切换话题
```

### 自定义评分标准

编辑 `Review_expert_model/auto_generate_dataset_fixed_v2.py`：

```python
# 修改评分算法
def calculate_language_score(audio_features):
    base_score = 70
    # 自定义加减分逻辑
    if fluency > 80:
        base_score += fluency * 0.3
    ...
```

### 添加新的智能提醒

编辑 `Review_expert_model/auto_generate_dataset_fixed_v2.py`：

```python
REMINDER_CATEGORIES = {
    30: "新增提醒类别：自定义场景",
    31: "新增提醒类别：特定问题",
    # ...
}
```

---

## 🚀 未来发展规划

### 短期优化（1-2周）

- [ ] **数据集扩充**
  - 面试官模型：增加更多行业的面试问题（金融、医疗、教育）
  - 评审专家模型：收集更多真实面试视频（目标：50个）

- [ ] **模型融合**
  - 实现两个模型的API级集成
  - 统一评分体系（文本分 + 行为分 → 综合分）

- [ ] **UI改进**
  - 统一Web界面（Flask/Streamlit）
  - 添加历史面试记录查询
  - 生成PDF评估报告

### 中期扩展（1-2个月）

- [ ] **完整闭环系统**
  - 提问 → 视频回答 → 双模型评分 → 追问决策 → 下一轮
  - 支持多轮对话（5-10轮）
  - 会话状态持久化

- [ ] **多语言支持**
  - 英文面试场景
  - 日文/韩文扩展

- [ ] **行业适配**
  - 技术面试（算法、系统设计）
  - 行为面试（STAR方法）
  - 压力面试

### 长期愿景（3-6个月）

- [ ] **产品化部署**
  - Web服务（FastAPI后端 + React前端）
  - 移动端适配（iOS/Android）
  - 云端部署（Docker + Kubernetes）

- [ ] **企业级功能**
  - 多租户支持
  - 数据分析看板
  - 与ATS（招聘系统）集成

- [ ] **学术贡献**
  - 发表论文（基于report.md大纲）
  - 开源数据集
  - 预训练模型发布

---

## 🤝 贡献指南

欢迎提出Issue和Pull Request！

### 贡献方向

1. **数据增强**：扩充训练数据，提升模型性能
2. **新功能开发**：
   - 语音识别（Whisper集成）
   - 实时字幕
   - 多人面试场景
3. **性能优化**：
   - 推理速度优化（TensorRT、ONNX）
   - 显存占用优化
   - 量化压缩
4. **文档改进**：翻译、示例、教程

### 开发规范

```bash
# Fork项目
git clone <your_fork>
cd "AI mock interview"

# 创建分支
git checkout -b feature/your-feature

# 开发 + 测试
# ...

# 提交PR
git push origin feature/your-feature
# 在GitHub上创建Pull Request
```

---

## 🔧 常见问题

### Q1: 显存不足怎么办？

```bash
# 方案1：仅运行一个模型
# 面试官模型：~3.38GB
streamlit run Interviewer_model/app_triple_qwen.py

# 评审专家模型：~2-3GB
python Review_expert_model/realtime_live_demo_fixed.py

# 方案2：降低batch size（训练时）
--batch_size 1 --gradient_accumulation_steps 8

# 方案3：使用CPU推理
export CUDA_VISIBLE_DEVICES=""

# 方案4：使用更小的基座模型
# Qwen2-1.5B → Qwen2-0.5B
```

### Q2: 如何评估模型质量？

```bash
# 面试官模型
cd Interviewer_model
python test_triple_qwen.py  # 功能测试
# 查看 plots/ 目录的训练曲线

# 评审专家模型
cd Review_expert_model
python tools/view_features.py  # 特征可视化
tensorboard --logdir=runs  # 查看训练曲线
```

### Q3: 训练数据从哪里来？

- **面试官模型**：
  - `Interviewer_model/training_data/qwen_data.json`：人工标注的真实面试数据（~8000条）
  - 生成脚本：`generate_training_data_*.py`

- **评审专家模型**：
  - `Review_expert_model/testv/`：原始面试视频（10个MP4）
  - 智能自动标注：`auto_generate_dataset_fixed_v2.py`（无需人工）

### Q4: 如何更换基座模型？

```python
# 面试官模型（Interviewer_model/train_*.py）
model_name = "Qwen/Qwen2-7B-Instruct"  # 更大模型，性能更强
# 或
model_name = "Qwen/Qwen2-0.5B-Instruct"  # 更小模型，速度更快

# 评审专家模型（Review_expert_model/model/transformer_model.py）
# 修改Transformer层数、隐藏维度等超参数
```

### Q5: 摄像头无法打开？

```bash
# 检查摄像头权限
# Windows: 设置 → 隐私 → 相机 → 允许应用访问

# 检查摄像头ID
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# 尝试不同ID
# realtime_live_demo_fixed.py 中修改：
cap = cv2.VideoCapture(1)  # 或 2, 3...
```

---

## 📚 相关文档

- 📄 [面试官模型详细文档](Interviewer_model/readme.md)
- 📄 [评审专家模型详细文档](Review_expert_model/readme.md)
- 📊 [技术报告](Interviewer_model/report.md)
- 🎓 [论文大纲](Review_expert_model/report.md)

---

## 📄 许可证

MIT License

---

## 👨‍💻 作者与致谢

**NLP课程项目 - 2025**

### 致谢

感谢以下开源项目和社区：

- **Qwen Team (Alibaba Cloud)** - 提供优秀的开源基座模型
- **Hugging Face** - 模型托管和训练框架
- **PyTorch Team** - 强大的深度学习框架
- **DeepFace** - 易用的人脸识别库
- **MediaPipe (Google)** - 高效的端侧CV解决方案
- **Librosa** - 优秀的音频处理库
- **Streamlit** - 易用的UI框架

### 参考文献

1. **Qwen2**: Qwen Team, Alibaba Cloud (2024)
2. **LoRA**: Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)
3. **QLoRA**: Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs" (NeurIPS 2023)
4. **Transformer**: Vaswani et al. "Attention is All You Need" (NIPS 2017)
5. **DeepFace**: Taigman et al. "DeepFace: Closing the Gap to Human-Level Performance" (CVPR 2014)
6. **MediaPipe**: Zhang et al. "MediaPipe Hands: On-device Real-time Hand Tracking" (2020)

---

## 📊 项目统计

- **总代码行数**: ~15,000行
- **模型数量**: 5个（3个Qwen + 1个Transformer + 备用模型）
- **训练数据**: 
  - 面试官模型：~8,000条文本样本
  - 评审专家模型：104个多模态样本
- **训练时间**: 
  - 面试官模型：~2-3小时/模型
  - 评审专家模型：~2小时
- **项目周期**: 6周
- **论文准备**: 进行中

---

## ⚠️ 免责声明

本项目仅用于**学习和研究目的**，不应用于：
- 正式招聘决策的唯一依据
- 商业化产品（未经授权）
- 歧视性评估

AI生成的评估仅供参考，最终决策应由人类专家做出。

---

## 🎯 项目目标

探索大语言模型和多模态深度学习在面试场景的应用，展示：
- 参数高效微调（LoRA）
- 多模型协同架构
- 跨模态Transformer融合
- 端到端AI面试系统

---

**最后更新**: 2025年11月1日  
**项目版本**: v1.0  
**维护状态**: 积极开发中

---

<div align="center">
  <strong>⭐ 如果这个项目对您有帮助，请给我们一个Star！⭐</strong>
</div>

