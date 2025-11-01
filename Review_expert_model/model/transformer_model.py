# -*- coding: utf-8 -*-
"""
多模态Transformer面试评分模型
输入: 情绪(7维) + 音频(80维) + 姿势(99维) + 眼动(5维)
输出: 5个评分 [语言表达, 心理素质, 肢体语言, 专注度, 总分] + 提醒生成
"""
import torch
import torch.nn as nn
import math

# 提醒类别映射表
REMINDER_MAP = {
    0: "表现很好，保持状态！",
    1: "你好像有点紧张，放轻松",
    2: "注意力集中，不要眼神乱飘",
    3: "有些语无伦次了，理清思路再说",
    4: "语速太快了，慢慢说清楚",
    5: "坐直一点，保持自信的姿态",
    6: "眼神有些疲惫，注意休息",
    7: "回答很流畅，继续加油",
    8: "情绪有些低落，积极一点",
    9: "专注度很好，保持眼神交流",
    10: "说话声音太小了，大声一点",
    11: "表述有条理，逻辑清晰",
    12: "注意眼神接触，展现自信",
    13: "回答内容不够充实，多举例说明",
    14: "肢体语言僵硬，放松一些",
    15: "情绪稳定，保持下去",
    16: "语言表达需要更精炼",
    17: "思考时间过长，适当加快节奏",
    18: "微笑自然，给人好印象",
    19: "坐姿端正，很专业",
    20: "回答切题，很好",
    21: "表情过于严肃，放松表情",
    22: "语调平淡，加强感染力",
    23: "手势运用得当",
    24: "回答简洁明了",
    25: "展现了专业素养",
    26: "沟通能力强",
    27: "应变能力不错",
    28: "需要更多自信",
    29: "整体表现优秀"
}

class PositionalEncoding(nn.Module):
    """位置编码（用于时序数据）"""
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class ModalityEncoder(nn.Module):
    """单个模态的编码器"""
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        
        # 投影层
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        """
        x: [batch, seq_len, input_dim] 或 [batch, input_dim]
        """
        # 如果是2维，添加seq维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]
        
        # 投影
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer(x)  # [batch, seq_len, d_model]
        
        # 平均池化
        x = x.mean(dim=1)  # [batch, d_model]
        
        return x


class InterviewTransformer(nn.Module):
    """
    多模态面试评分Transformer
    
    输入:
    - emotion_seq: [batch, seq_len, 7] 情绪时序
    - audio_mel: [batch, 80] 音频梅尔频谱
    - pose_seq: [batch, seq_len, 99] 姿势时序
    - gaze_seq: [batch, seq_len, 5] 眼动时序 ✨ 新增
    
    输出:
    - scores: [batch, 5] [语言表达, 心理素质, 肢体语言, 专注度, 总分]
    - reminder_class: [batch] 提醒类别 (0-9)
    """
    
    def __init__(self, d_model=128, nhead=4, num_encoder_layers=2, dropout=0.3, num_reminders=30):
        super().__init__()
        
        # 保存配置
        self.d_model = d_model
        self.num_reminders = num_reminders
        
        # ===== 模态编码器 =====
        # 情绪编码器（时序）
        self.emotion_encoder = ModalityEncoder(
            input_dim=7,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dropout=dropout
        )
        
        # 音频编码器（时序）
        self.audio_encoder = ModalityEncoder(
            input_dim=80,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dropout=dropout
        )
        
        # 姿势编码器（时序）
        self.pose_encoder = ModalityEncoder(
            input_dim=99,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dropout=dropout
        )
        
        # 眼动编码器（时序）✨ 新增
        self.gaze_encoder = ModalityEncoder(
            input_dim=5,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dropout=dropout
        )
        
        # ===== 跨模态融合 =====
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # 融合Transformer
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(fusion_layer, num_layers=2)
        
        # ===== 共享表示层 =====
        self.shared_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ===== 评分预测头 =====
        # 语言表达（主要基于音频）
        self.language_head = self._make_score_head(d_model, dropout)
        
        # 心理素质（主要基于情绪）
        self.psychological_head = self._make_score_head(d_model, dropout)
        
        # 肢体语言（主要基于姿势）
        self.body_language_head = self._make_score_head(d_model, dropout)
        
        # 专注度（主要基于眼动）✨ 新增
        self.focus_head = self._make_score_head(d_model, dropout)
        
        # 总分（综合所有模态）
        self.total_head = self._make_score_head(d_model, dropout)
        
        # ===== 提醒生成头 ✨ 新增 =====
        self.reminder_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_reminders)
            # 不加 Softmax，在损失函数中使用 CrossEntropyLoss
        )
    
    def _make_score_head(self, d_model, dropout):
        """创建评分预测头"""
        return nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出0-1，后续乘100
        )
    
    def forward(self, emotion_seq, audio_mel, pose_seq, gaze_seq):
        """
        前向传播
        
        Args:
            emotion_seq: [batch, seq_len, 7] 情绪时序
            audio_mel: [batch, 80] 音频特征
            pose_seq: [batch, seq_len, 99] 姿势时序
            gaze_seq: [batch, seq_len, 5] 眼动时序 ✨ 新增
        
        Returns:
            scores: [batch, 5] 五个评分 [语言, 心理, 肢体, 专注, 总分]
            reminder_logits: [batch, num_reminders] 提醒类别logits
            attention_weights: dict 注意力权重（可视化用）
        """
        batch_size = emotion_seq.size(0)
        
        # ===== 模态编码 =====
        emotion_emb = self.emotion_encoder(emotion_seq)  # [batch, d_model]
        audio_emb = self.audio_encoder(audio_mel)        # [batch, d_model]
        pose_emb = self.pose_encoder(pose_seq)           # [batch, d_model]
        gaze_emb = self.gaze_encoder(gaze_seq)           # [batch, d_model] ✨
        
        # ===== 跨模态融合 =====
        # 拼接所有模态 [batch, 4, d_model] ✨ 3→4
        all_modalities = torch.stack([emotion_emb, audio_emb, pose_emb, gaze_emb], dim=1)
        
        # Cross-Modal Attention
        fused, cross_attn_weights = self.cross_modal_attention(
            query=all_modalities,
            key=all_modalities,
            value=all_modalities
        )  # [batch, 4, d_model]
        
        # Transformer深度融合
        fused = self.fusion_transformer(fused)  # [batch, 4, d_model]
        
        # 全局表示（平均池化）
        global_repr = fused.mean(dim=1)  # [batch, d_model]
        
        # ===== 共享表示 =====
        shared = self.shared_head(global_repr)  # [batch, d_model]
        
        # ===== 多任务预测 =====
        # 5个评分
        language_score = self.language_head(shared) * 100      # [batch, 1]
        psychological_score = self.psychological_head(shared) * 100
        body_language_score = self.body_language_head(shared) * 100
        focus_score = self.focus_head(shared) * 100            # [batch, 1] ✨
        total_score = self.total_head(shared) * 100
        
        # 拼接分数
        scores = torch.cat([
            language_score,
            psychological_score,
            body_language_score,
            focus_score,
            total_score
        ], dim=1)  # [batch, 5]
        
        # 提醒生成 ✨
        reminder_logits = self.reminder_head(shared)  # [batch, num_reminders]
        
        return scores, reminder_logits, {
            'cross_modal_attention': cross_attn_weights,
            'modality_embeddings': {
                'emotion': emotion_emb,
                'audio': audio_emb,
                'pose': pose_emb,
                'gaze': gaze_emb
            },
            'fused': fused
        }


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    print("="*60)
    print("  Testing InterviewTransformer Model (Updated)")
    print("="*60 + "\n")
    
    # 创建模型
    model = InterviewTransformer(d_model=128, nhead=4, num_encoder_layers=2)
    
    # 统计参数
    params = count_parameters(model)
    print(f"Model parameters: {params:,} ({params/1e6:.2f}M)")
    
    # 测试前向传播
    batch_size = 4
    seq_len = 5
    
    # 模拟输入（4个模态）
    emotion_seq = torch.randn(batch_size, seq_len, 7)
    audio_mel = torch.randn(batch_size, 80)
    pose_seq = torch.randn(batch_size, seq_len, 99)
    gaze_seq = torch.randn(batch_size, seq_len, 5)  # ✨ 新增
    
    print(f"\nInput shapes:")
    print(f"  Emotion: {emotion_seq.shape}")
    print(f"  Audio: {audio_mel.shape}")
    print(f"  Pose: {pose_seq.shape}")
    print(f"  Gaze: {gaze_seq.shape}  [NEW]")
    
    # 前向传播
    scores, reminder_logits, attention_weights = model(
        emotion_seq, audio_mel, pose_seq, gaze_seq
    )
    
    print(f"\nOutput shapes:")
    print(f"  Scores: {scores.shape}  (5个评分)")
    print(f"  Reminder logits: {reminder_logits.shape}  (10个类别)")
    print(f"  Cross-modal attention: {attention_weights['cross_modal_attention'].shape}")
    
    print(f"\nSample scores:")
    print(f"  {scores[0].detach().numpy()}")
    print(f"  Range: [{scores.min():.1f}, {scores.max():.1f}]")
    
    print(f"\nSample reminder:")
    reminder_class = torch.argmax(reminder_logits[0]).item()
    print(f"  Class: {reminder_class}")
    print(f"  Text: {REMINDER_MAP[reminder_class]}")
    
    print(f"\n[SUCCESS] Model architecture verified!")
    print(f"Ready for training!\n")



