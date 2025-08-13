"""
跨模态融合模块

实现跨模态注意力机制和多模态信息融合，整合视觉和文本特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from .multimodal_config import FusionConfig, AttentionType, FusionStrategy


class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 查询、键、值投影层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # 跨模态特殊投影
        self.cross_modal_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.cross_modal_key = nn.Linear(config.hidden_size, self.all_head_size)
        
        # 输出投影
        self.output_projection = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout)
        
        # 温度参数（可学习）
        self.temperature = nn.Parameter(torch.tensor(config.temperature))
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid()
        )
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """调整张量形状以计算注意力分数"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, 
                vision_features: torch.Tensor,
                text_features: torch.Tensor,
                vision_mask: Optional[torch.Tensor] = None,
                text_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        跨模态注意力计算
        
        Args:
            vision_features: (batch_size, vision_seq_len, hidden_size)
            text_features: (batch_size, text_seq_len, hidden_size)
            vision_mask: (batch_size, vision_seq_len)
            text_mask: (batch_size, text_seq_len)
        Returns:
            Dict containing fused features and attention weights
        """
        batch_size = vision_features.size(0)
        
        # Vision -> Text 注意力
        # Vision作为Query，Text作为Key和Value
        v2t_query = self.transpose_for_scores(self.cross_modal_query(vision_features))
        t_key = self.transpose_for_scores(self.key(text_features))
        t_value = self.transpose_for_scores(self.value(text_features))
        
        # 计算Vision到Text的注意力分数
        v2t_attention_scores = torch.matmul(v2t_query, t_key.transpose(-1, -2))
        v2t_attention_scores = v2t_attention_scores / (math.sqrt(self.attention_head_size) * self.temperature)
        
        # 应用文本掩码
        if text_mask is not None:
            # 扩展掩码维度
            extended_text_mask = text_mask.unsqueeze(1).unsqueeze(2)
            extended_text_mask = extended_text_mask.expand(
                batch_size, self.num_attention_heads, vision_features.size(1), text_features.size(1)
            )
            v2t_attention_scores = v2t_attention_scores.masked_fill(extended_text_mask == 0, -1e9)
        
        v2t_attention_probs = F.softmax(v2t_attention_scores, dim=-1)
        v2t_attention_probs = self.dropout(v2t_attention_probs)
        
        # 计算Vision到Text的上下文
        v2t_context = torch.matmul(v2t_attention_probs, t_value)
        v2t_context = v2t_context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = v2t_context.size()[:-2] + (self.all_head_size,)
        v2t_context = v2t_context.view(*new_context_shape)
        
        # Text -> Vision 注意力
        # Text作为Query，Vision作为Key和Value
        t2v_query = self.transpose_for_scores(self.cross_modal_query(text_features))
        v_key = self.transpose_for_scores(self.key(vision_features))
        v_value = self.transpose_for_scores(self.value(vision_features))
        
        # 计算Text到Vision的注意力分数
        t2v_attention_scores = torch.matmul(t2v_query, v_key.transpose(-1, -2))
        t2v_attention_scores = t2v_attention_scores / (math.sqrt(self.attention_head_size) * self.temperature)
        
        # 应用视觉掩码
        if vision_mask is not None:
            extended_vision_mask = vision_mask.unsqueeze(1).unsqueeze(2)
            extended_vision_mask = extended_vision_mask.expand(
                batch_size, self.num_attention_heads, text_features.size(1), vision_features.size(1)
            )
            t2v_attention_scores = t2v_attention_scores.masked_fill(extended_vision_mask == 0, -1e9)
        
        t2v_attention_probs = F.softmax(t2v_attention_scores, dim=-1)
        t2v_attention_probs = self.dropout(t2v_attention_probs)
        
        # 计算Text到Vision的上下文
        t2v_context = torch.matmul(t2v_attention_probs, v_value)
        t2v_context = t2v_context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = t2v_context.size()[:-2] + (self.all_head_size,)
        t2v_context = t2v_context.view(*new_context_shape)
        
        # 输出投影
        enhanced_vision = self.output_projection(v2t_context)
        enhanced_text = self.output_projection(t2v_context)
        
        # 门控融合
        vision_gate = self.gate(torch.cat([vision_features, enhanced_vision], dim=-1))
        text_gate = self.gate(torch.cat([text_features, enhanced_text], dim=-1))
        
        fused_vision = vision_features + vision_gate * enhanced_vision
        fused_text = text_features + text_gate * enhanced_text
        
        return {
            'fused_vision_features': fused_vision,
            'fused_text_features': fused_text,
            'v2t_attention_weights': v2t_attention_probs,
            't2v_attention_weights': t2v_attention_probs,
            'vision_enhancement': enhanced_vision,
            'text_enhancement': enhanced_text
        }


class TemporalFusionLayer(nn.Module):
    """时序融合层"""
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.temporal_window = config.temporal_window
        
        # 时序注意力
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # 位置编码
        self.position_encoding = self._create_position_encoding(
            config.temporal_window, config.hidden_size
        )
        
        # 时序卷积
        self.temporal_conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=3,
            padding=1
        )
        
        # LSTM for temporal modeling
        self.temporal_lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # 融合层
        self.temporal_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
    def _create_position_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def forward(self, 
                temporal_features: torch.Tensor,
                temporal_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        时序特征融合
        
        Args:
            temporal_features: (batch_size, seq_len, hidden_size)
            temporal_mask: (batch_size, seq_len)
        Returns:
            Dict containing temporally fused features
        """
        batch_size, seq_len, hidden_size = temporal_features.shape
        
        # 添加位置编码
        if seq_len <= self.temporal_window:
            pos_encoding = self.position_encoding[:, :seq_len, :].to(temporal_features.device)
            features_with_pos = temporal_features + pos_encoding
        else:
            features_with_pos = temporal_features
        
        # 时序注意力
        if temporal_mask is not None:
            # 转换掩码格式
            key_padding_mask = (temporal_mask == 0)
        else:
            key_padding_mask = None
            
        attn_output, attn_weights = self.temporal_attention(
            features_with_pos, features_with_pos, features_with_pos,
            key_padding_mask=key_padding_mask
        )
        
        # 时序卷积
        conv_input = temporal_features.transpose(1, 2)  # (B, H, S)
        conv_output = self.temporal_conv(conv_input)
        conv_output = conv_output.transpose(1, 2)  # (B, S, H)
        conv_output = F.gelu(conv_output)
        
        # LSTM处理
        lstm_output, (hidden, cell) = self.temporal_lstm(temporal_features)
        
        # 融合所有时序特征
        combined_features = torch.cat([attn_output, lstm_output], dim=-1)
        fused_temporal = self.temporal_fusion(combined_features)
        
        # 残差连接
        output = temporal_features + fused_temporal
        
        return {
            'fused_temporal_features': output,
            'temporal_attention_weights': attn_weights,
            'conv_features': conv_output,
            'lstm_features': lstm_output
        }


class AdaptiveFusionGate(nn.Module):
    """自适应融合门控"""
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # 模态重要性预测器
        self.modality_importance = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 2),  # vision, text importance
            nn.Softmax(dim=-1)
        )
        
        # 自适应权重生成器
        self.adaptive_weights = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Sigmoid()
        )
        
        # 特征对齐层
        self.feature_alignment = nn.Sequential(
            nn.Linear(config.hidden_size, config.fused_feature_dim),
            nn.LayerNorm(config.fused_feature_dim),
            nn.GELU()
        )
        
        # 融合策略选择器
        self.fusion_strategy_selector = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 4),  # 4种融合策略
            nn.Softmax(dim=-1)
        )
        
        self.fusion_strategies = ['concat', 'add', 'multiply', 'attention']
        
    def forward(self, 
                vision_features: torch.Tensor,
                text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        自适应融合
        
        Args:
            vision_features: (batch_size, hidden_size)
            text_features: (batch_size, hidden_size)
        Returns:
            Dict containing adaptively fused features
        """
        batch_size = vision_features.size(0)
        
        # 计算模态重要性
        combined_input = torch.cat([vision_features, text_features], dim=-1)
        modality_weights = self.modality_importance(combined_input)
        vision_weight = modality_weights[:, 0:1]  # (B, 1)
        text_weight = modality_weights[:, 1:2]    # (B, 1)
        
        # 生成自适应权重
        adaptive_weight = self.adaptive_weights(combined_input)
        
        # 特征对齐
        aligned_vision = self.feature_alignment(vision_features)
        aligned_text = self.feature_alignment(text_features)
        
        # 融合策略选择
        strategy_weights = self.fusion_strategy_selector(combined_input)
        
        # 执行不同的融合策略
        # 1. 拼接融合
        concat_fusion = torch.cat([
            aligned_vision * vision_weight.unsqueeze(-1),
            aligned_text * text_weight.unsqueeze(-1)
        ], dim=-1)
        
        # 2. 加法融合
        add_fusion = aligned_vision * vision_weight.unsqueeze(-1) + \
                    aligned_text * text_weight.unsqueeze(-1)
        
        # 3. 乘法融合  
        multiply_fusion = aligned_vision * aligned_text
        
        # 4. 注意力融合
        attention_scores = F.softmax(
            torch.bmm(aligned_vision.unsqueeze(1), aligned_text.unsqueeze(2)), 
            dim=-1
        ).squeeze()
        attention_fusion = aligned_vision * attention_scores.unsqueeze(-1) + \
                          aligned_text * (1 - attention_scores.unsqueeze(-1))
        
        # 策略加权组合
        # 调整融合特征的维度匹配
        if concat_fusion.size(-1) != add_fusion.size(-1):
            # 对concat_fusion进行降维以匹配其他融合方式
            concat_projection = nn.Linear(concat_fusion.size(-1), add_fusion.size(-1)).to(concat_fusion.device)
            concat_fusion = concat_projection(concat_fusion)
        
        final_fusion = (strategy_weights[:, 0:1] * concat_fusion +
                       strategy_weights[:, 1:2] * add_fusion +
                       strategy_weights[:, 2:3] * multiply_fusion +
                       strategy_weights[:, 3:4] * attention_fusion)
        
        return {
            'fused_features': final_fusion,
            'modality_weights': modality_weights,
            'adaptive_weights': adaptive_weight,
            'strategy_weights': strategy_weights,
            'aligned_vision': aligned_vision,
            'aligned_text': aligned_text,
            'fusion_components': {
                'concat': concat_fusion,
                'add': add_fusion,
                'multiply': multiply_fusion,
                'attention': attention_fusion
            }
        }


class MultimodalFusionTransformer(nn.Module):
    """多模态融合Transformer"""
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # 跨模态注意力层
        self.cross_modal_layers = nn.ModuleList([
            CrossModalAttention(config) for _ in range(config.num_fusion_layers)
        ])
        
        # 时序融合层
        if config.use_temporal_attention:
            self.temporal_fusion = TemporalFusionLayer(config)
        else:
            self.temporal_fusion = None
            
        # 自适应融合门控
        self.adaptive_fusion = AdaptiveFusionGate(config)
        
        # 层归一化
        if config.layer_norm:
            self.vision_layer_norms = nn.ModuleList([
                nn.LayerNorm(config.hidden_size) for _ in range(config.num_fusion_layers)
            ])
            self.text_layer_norms = nn.ModuleList([
                nn.LayerNorm(config.hidden_size) for _ in range(config.num_fusion_layers)
            ])
        
        # 最终投影层
        self.final_projection = nn.Sequential(
            nn.Linear(config.fused_feature_dim, config.fused_feature_dim),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(config.fused_feature_dim, config.fused_feature_dim)
        )
        
        # 输出头
        self.classification_head = nn.Linear(config.fused_feature_dim, 3)  # 买入/卖出/持有
        self.regression_head = nn.Linear(config.fused_feature_dim, 1)     # 连续预测值
        
    def forward(self, 
                vision_features: torch.Tensor,
                text_features: torch.Tensor,
                vision_mask: Optional[torch.Tensor] = None,
                text_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        多模态融合前向传播
        
        Args:
            vision_features: (batch_size, vision_seq_len, hidden_size)
            text_features: (batch_size, text_seq_len, hidden_size)
            vision_mask: (batch_size, vision_seq_len)
            text_mask: (batch_size, text_seq_len)
            return_attention: whether to return attention weights
        Returns:
            Dict containing fused features and predictions
        """
        attention_weights = [] if return_attention else None
        
        current_vision = vision_features
        current_text = text_features
        
        # 多层跨模态融合
        for i, cross_modal_layer in enumerate(self.cross_modal_layers):
            # 跨模态注意力
            cross_modal_output = cross_modal_layer(
                current_vision, current_text,
                vision_mask, text_mask
            )
            
            # 更新特征
            current_vision = cross_modal_output['fused_vision_features']
            current_text = cross_modal_output['fused_text_features']
            
            # 层归一化
            if self.config.layer_norm:
                current_vision = self.vision_layer_norms[i](current_vision)
                current_text = self.text_layer_norms[i](current_text)
            
            # 残差连接
            if self.config.residual_connections:
                if i == 0:
                    current_vision = current_vision + vision_features
                    current_text = current_text + text_features
                else:
                    # 与前一层的输出进行残差连接
                    pass  # 已经在CrossModalAttention中处理
            
            if return_attention:
                attention_weights.append({
                    'v2t_attention': cross_modal_output['v2t_attention_weights'],
                    't2v_attention': cross_modal_output['t2v_attention_weights']
                })
        
        # 全局池化获取固定大小特征
        pooled_vision = current_vision.mean(dim=1)  # (batch_size, hidden_size)
        pooled_text = current_text.mean(dim=1)      # (batch_size, hidden_size)
        
        # 时序融合（如果启用）
        temporal_output = None
        if self.temporal_fusion is not None:
            # 将视觉和文本特征组合进行时序建模
            combined_temporal = torch.cat([current_vision, current_text], dim=1)
            temporal_output = self.temporal_fusion(combined_temporal)
            
            # 更新池化特征
            pooled_temporal = temporal_output['fused_temporal_features'].mean(dim=1)
            pooled_vision = (pooled_vision + pooled_temporal[:, :pooled_vision.size(1)]) / 2
            pooled_text = (pooled_text + pooled_temporal[:, pooled_vision.size(1):]) / 2
        
        # 自适应融合
        adaptive_output = self.adaptive_fusion(pooled_vision, pooled_text)
        final_features = adaptive_output['fused_features']
        
        # 最终投影
        final_features = self.final_projection(final_features)
        
        # 生成预测
        classification_output = self.classification_head(final_features)
        regression_output = self.regression_head(final_features)
        
        result = {
            'fused_features': final_features,
            'classification_logits': classification_output,
            'regression_output': regression_output,
            'vision_features': current_vision,
            'text_features': current_text,
            'pooled_vision': pooled_vision,
            'pooled_text': pooled_text,
            'adaptive_fusion_output': adaptive_output
        }
        
        if temporal_output is not None:
            result['temporal_output'] = temporal_output
            
        if return_attention:
            result['attention_weights'] = attention_weights
            
        return result


def create_multimodal_fusion_model(config: FusionConfig) -> MultimodalFusionTransformer:
    """创建多模态融合模型的工厂函数"""
    model = MultimodalFusionTransformer(config)
    
    # 初始化权重
    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    model.apply(init_weights)
    return model


if __name__ == "__main__":
    # 测试跨模态融合模块
    print("=== 测试跨模态融合模块 ===")
    
    config = FusionConfig()
    
    # 创建测试数据
    batch_size = 4
    vision_seq_len = 16
    text_seq_len = 32
    hidden_size = config.hidden_size
    
    vision_features = torch.randn(batch_size, vision_seq_len, hidden_size)
    text_features = torch.randn(batch_size, text_seq_len, hidden_size)
    
    # 测试跨模态注意力
    print("\n--- 测试跨模态注意力 ---")
    cross_attention = CrossModalAttention(config)
    
    with torch.no_grad():
        cross_attention_output = cross_attention(vision_features, text_features)
    
    print(f"融合后视觉特征形状: {cross_attention_output['fused_vision_features'].shape}")
    print(f"融合后文本特征形状: {cross_attention_output['fused_text_features'].shape}")
    print(f"V2T注意力权重形状: {cross_attention_output['v2t_attention_weights'].shape}")
    print(f"T2V注意力权重形状: {cross_attention_output['t2v_attention_weights'].shape}")
    
    # 测试时序融合
    print("\n--- 测试时序融合 ---")
    temporal_fusion = TemporalFusionLayer(config)
    
    temporal_features = torch.randn(batch_size, 30, hidden_size)
    
    with torch.no_grad():
        temporal_output = temporal_fusion(temporal_features)
    
    print(f"时序融合特征形状: {temporal_output['fused_temporal_features'].shape}")
    print(f"时序注意力权重形状: {temporal_output['temporal_attention_weights'].shape}")
    
    # 测试自适应融合门控
    print("\n--- 测试自适应融合门控 ---")
    adaptive_gate = AdaptiveFusionGate(config)
    
    pooled_vision = vision_features.mean(dim=1)
    pooled_text = text_features.mean(dim=1)
    
    with torch.no_grad():
        adaptive_output = adaptive_gate(pooled_vision, pooled_text)
    
    print(f"自适应融合特征形状: {adaptive_output['fused_features'].shape}")
    print(f"模态权重形状: {adaptive_output['modality_weights'].shape}")
    print(f"策略权重: {adaptive_output['strategy_weights'][0].detach().numpy()}")
    
    # 测试完整的多模态融合Transformer
    print("\n--- 测试多模态融合Transformer ---")
    fusion_transformer = create_multimodal_fusion_model(config)
    
    with torch.no_grad():
        fusion_output = fusion_transformer(
            vision_features, text_features,
            return_attention=True
        )
    
    print(f"最终融合特征形状: {fusion_output['fused_features'].shape}")
    print(f"分类输出形状: {fusion_output['classification_logits'].shape}")
    print(f"回归输出形状: {fusion_output['regression_output'].shape}")
    print(f"注意力层数: {len(fusion_output['attention_weights'])}")
    
    # 测试模型参数量
    total_params = sum(p.numel() for p in fusion_transformer.parameters())
    trainable_params = sum(p.numel() for p in fusion_transformer.parameters() if p.requires_grad)
    
    print(f"\n模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    print("\n=== 跨模态融合模块测试完成 ===")