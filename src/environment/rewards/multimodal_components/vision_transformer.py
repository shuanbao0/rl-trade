"""
Vision Transformer图表分析模块

实现基于ViT的金融图表分析，包括技术指标识别、烛台图案识别和价格模式分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore')

from .multimodal_config import VisionConfig, DataModalityType


class PatchEmbedding(nn.Module):
    """图像块嵌入层"""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # 位置嵌入
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches + 1, config.hidden_size)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_size)
        )
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            embeddings: (batch_size, num_patches + 1, hidden_size)
        """
        batch_size = x.shape[0]
        
        # 图像块投影: (B, C, H, W) -> (B, hidden_size, H/P, W/P)
        x = self.projection(x)
        
        # 展平: (B, hidden_size, H/P, W/P) -> (B, hidden_size, num_patches)
        x = x.flatten(2)
        
        # 转置: (B, hidden_size, num_patches) -> (B, num_patches, hidden_size)
        x = x.transpose(1, 2)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置嵌入
        x = x + self.position_embeddings
        
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """调整张量形状以计算注意力分数"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
            attention_mask: (batch_size, seq_length)
        Returns:
            output: (batch_size, seq_length, hidden_size)
            attention_probs: (batch_size, num_heads, seq_length, seq_length)
        """
        # 计算QKV
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax归一化
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 计算上下文向量
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # 输出投影
        output = self.output_projection(context_layer)
        output = self.output_dropout(output)
        
        return output, attention_probs


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        
        self.layernorm_before = nn.LayerNorm(config.hidden_size)
        self.layernorm_after = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
        Returns:
            output: (batch_size, seq_length, hidden_size)
            attention_probs: attention probabilities
        """
        # 自注意力 + 残差连接
        attention_output, attention_probs = self.attention(
            self.layernorm_before(hidden_states), attention_mask
        )
        hidden_states = hidden_states + attention_output
        
        # FFN + 残差连接
        intermediate_output = F.gelu(self.intermediate(self.layernorm_after(hidden_states)))
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        output = hidden_states + layer_output
        
        return output, attention_probs


class VisionTransformer(nn.Module):
    """Vision Transformer主模型"""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        
        # 图像块嵌入
        self.embeddings = PatchEmbedding(config)
        
        # Transformer层
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # 最终层归一化
        self.layernorm = nn.LayerNorm(config.hidden_size)
        
        # 分类头
        self.classifier = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.init_weights()
        
    def init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, pixel_values: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: (batch_size, channels, height, width)
            attention_mask: attention mask
            return_attention: whether to return attention weights
        Returns:
            dict containing:
                - last_hidden_state: (batch_size, seq_length, hidden_size)
                - pooler_output: (batch_size, hidden_size)
                - attention_weights: list of attention weights (optional)
        """
        # 图像块嵌入
        embedding_output = self.embeddings(pixel_values)
        
        attention_weights = [] if return_attention else None
        hidden_states = embedding_output
        
        # 通过所有Transformer层
        for layer in self.encoder_layers:
            hidden_states, attention_probs = layer(hidden_states, attention_mask)
            if return_attention:
                attention_weights.append(attention_probs)
        
        # 最终层归一化
        sequence_output = self.layernorm(hidden_states)
        
        # 池化输出 (CLS token)
        pooled_output = self.classifier(sequence_output[:, 0])
        
        result = {
            "last_hidden_state": sequence_output,
            "pooler_output": pooled_output
        }
        
        if return_attention:
            result["attention_weights"] = attention_weights
            
        return result


class ChartFeatureExtractor(nn.Module):
    """图表特征提取器"""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.vit = VisionTransformer(config)
        
        # 技术指标特征提取头
        self.technical_indicator_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, len(config.technical_indicators))
        )
        
        # 价格模式识别头
        self.pattern_recognition_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 32)  # 32种常见模式
        )
        
        # 趋势分析头
        self.trend_analysis_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 3)  # 上涨/下跌/震荡
        )
        
    def forward(self, chart_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            chart_images: (batch_size, channels, height, width)
        Returns:
            Dict containing extracted features
        """
        # ViT特征提取
        vit_outputs = self.vit(chart_images, return_attention=True)
        pooled_features = vit_outputs["pooler_output"]
        
        # 各种特征提取
        technical_features = self.technical_indicator_head(pooled_features)
        pattern_features = self.pattern_recognition_head(pooled_features)
        trend_features = self.trend_analysis_head(pooled_features)
        
        return {
            "vit_features": pooled_features,
            "technical_indicators": technical_features,
            "pattern_recognition": pattern_features,
            "trend_analysis": trend_features,
            "attention_weights": vit_outputs["attention_weights"]
        }


class TechnicalIndicatorEncoder(nn.Module):
    """技术指标编码器"""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        
        # 各种技术指标的编码器
        self.indicator_encoders = nn.ModuleDict({
            "sma": self._create_indicator_encoder(5),  # 5种不同期间的SMA
            "ema": self._create_indicator_encoder(5),  # 5种不同期间的EMA
            "rsi": self._create_indicator_encoder(1),  # RSI
            "macd": self._create_indicator_encoder(3), # MACD, Signal, Histogram
            "bollinger_bands": self._create_indicator_encoder(3), # Upper, Lower, %B
            "atr": self._create_indicator_encoder(1),  # ATR
            "obv": self._create_indicator_encoder(1),  # OBV
            "stochastic": self._create_indicator_encoder(2) # %K, %D
        })
        
        # 融合层
        total_features = sum(encoder[-1].out_features for encoder in self.indicator_encoders.values())
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
    def _create_indicator_encoder(self, input_dim: int) -> nn.Module:
        """创建单个指标编码器"""
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128)
        )
    
    def forward(self, technical_indicators: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            technical_indicators: dict of indicator tensors
        Returns:
            encoded_features: (batch_size, hidden_size)
        """
        encoded_features = []
        
        for indicator_name, encoder in self.indicator_encoders.items():
            if indicator_name in technical_indicators:
                indicator_data = technical_indicators[indicator_name]
                encoded = encoder(indicator_data)
                encoded_features.append(encoded)
            else:
                # 如果缺少某个指标，用零填充
                batch_size = list(technical_indicators.values())[0].shape[0]
                zeros = torch.zeros(batch_size, encoder[-1].out_features, 
                                  device=next(self.parameters()).device)
                encoded_features.append(zeros)
        
        # 融合所有编码特征
        concatenated = torch.cat(encoded_features, dim=-1)
        fused_features = self.fusion_layer(concatenated)
        
        return fused_features


class CandlestickPatternRecognizer(nn.Module):
    """烛台图案识别器"""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        
        # 烛台图案列表 (32种常见模式)
        self.patterns = [
            "doji", "hammer", "shooting_star", "spinning_top",
            "marubozu", "hanging_man", "inverted_hammer", "dragonfly_doji",
            "gravestone_doji", "engulfing_bullish", "engulfing_bearish",
            "harami_bullish", "harami_bearish", "piercing_pattern",
            "dark_cloud_cover", "morning_star", "evening_star",
            "three_white_soldiers", "three_black_crows", "rising_three_methods",
            "falling_three_methods", "tweezer_top", "tweezer_bottom",
            "gap_up", "gap_down", "inside_day", "outside_day",
            "key_reversal_up", "key_reversal_down", "breakaway_gap",
            "exhaustion_gap", "continuation_pattern"
        ]
        
        # 模式识别网络
        self.pattern_detector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 4, len(self.patterns))
        )
        
        # 可靠性评估网络
        self.reliability_estimator = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, len(self.patterns))
        )
        
    def forward(self, vit_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            vit_features: (batch_size, hidden_size)
        Returns:
            Dict containing pattern predictions and reliability scores
        """
        # 模式概率
        pattern_logits = self.pattern_detector(vit_features)
        pattern_probs = torch.sigmoid(pattern_logits)
        
        # 可靠性分数
        reliability_scores = torch.sigmoid(self.reliability_estimator(vit_features))
        
        # 加权模式分数
        weighted_patterns = pattern_probs * reliability_scores
        
        return {
            "pattern_probabilities": pattern_probs,
            "reliability_scores": reliability_scores,
            "weighted_patterns": weighted_patterns,
            "pattern_names": self.patterns
        }


class ViTChartAnalyzer(nn.Module):
    """完整的ViT图表分析器"""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        
        # 核心组件
        self.feature_extractor = ChartFeatureExtractor(config)
        self.technical_encoder = TechnicalIndicatorEncoder(config)
        self.pattern_recognizer = CandlestickPatternRecognizer(config)
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )
        
        # 输出头
        self.signal_head = nn.Linear(config.hidden_size, 3)  # 买入/卖出/持有
        self.confidence_head = nn.Linear(config.hidden_size, 1)  # 信心度
        
    def forward(self, chart_images: torch.Tensor,
                technical_indicators: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            chart_images: (batch_size, channels, height, width)
            technical_indicators: optional technical indicator data
        Returns:
            Comprehensive chart analysis results
        """
        # 图表特征提取
        chart_features = self.feature_extractor(chart_images)
        vit_features = chart_features["vit_features"]
        
        # 技术指标编码
        if technical_indicators is not None:
            tech_features = self.technical_encoder(technical_indicators)
        else:
            # 如果没有技术指标，使用零向量
            batch_size = chart_images.shape[0]
            tech_features = torch.zeros(batch_size, self.config.hidden_size,
                                      device=chart_images.device)
        
        # 烛台模式识别
        pattern_results = self.pattern_recognizer(vit_features)
        pattern_features = pattern_results["weighted_patterns"]
        
        # 扩展pattern_features到hidden_size维度
        if pattern_features.shape[-1] != self.config.hidden_size:
            pattern_proj = nn.Linear(pattern_features.shape[-1], self.config.hidden_size).to(pattern_features.device)
            pattern_features = pattern_proj(pattern_features)
        
        # 最终特征融合
        combined_features = torch.cat([vit_features, tech_features, pattern_features], dim=-1)
        fused_features = self.final_fusion(combined_features)
        
        # 生成交易信号
        trading_signal = self.signal_head(fused_features)
        confidence_score = torch.sigmoid(self.confidence_head(fused_features))
        
        return {
            "trading_signal": trading_signal,
            "confidence_score": confidence_score,
            "vit_features": vit_features,
            "technical_features": tech_features,
            "pattern_results": pattern_results,
            "chart_analysis": chart_features,
            "fused_features": fused_features
        }


def create_chart_image(ohlcv_data: pd.DataFrame, 
                      technical_indicators: Optional[Dict] = None,
                      image_size: int = 224) -> np.ndarray:
    """
    创建金融图表图像
    
    Args:
        ohlcv_data: OHLCV数据
        technical_indicators: 技术指标数据
        image_size: 输出图像大小
    Returns:
        chart_image: (channels, height, width) numpy array
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # 绘制烛台图
    ohlc_data = ohlcv_data[['Open', 'High', 'Low', 'Close']].values
    for i, (o, h, l, c) in enumerate(ohlc_data):
        color = 'green' if c >= o else 'red'
        # 实体
        ax1.add_patch(patches.Rectangle((i-0.3, min(o, c)), 0.6, abs(c-o), 
                                       facecolor=color, alpha=0.7))
        # 影线
        ax1.plot([i, i], [l, h], color='black', linewidth=1)
    
    # 添加技术指标
    if technical_indicators:
        if 'sma_20' in technical_indicators:
            ax1.plot(technical_indicators['sma_20'], label='SMA 20', alpha=0.7)
        if 'bollinger_upper' in technical_indicators:
            ax1.plot(technical_indicators['bollinger_upper'], 'b--', alpha=0.5)
            ax1.plot(technical_indicators['bollinger_lower'], 'b--', alpha=0.5)
    
    # 绘制成交量
    colors = ['green' if c >= o else 'red' for o, c in zip(ohlcv_data['Open'], ohlcv_data['Close'])]
    ax2.bar(range(len(ohlcv_data)), ohlcv_data['Volume'], color=colors, alpha=0.7)
    
    # 设置样式
    ax1.set_ylabel('Price')
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Time')
    plt.tight_layout()
    
    # 转换为图像数组
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    image = image.resize((image_size, image_size))
    
    # 转换为numpy数组并标准化
    image_array = np.array(image) / 255.0
    image_array = image_array.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    
    return image_array.astype(np.float32)


# 数学导入（修复missing import）
import math


if __name__ == "__main__":
    # 测试Vision Transformer
    print("=== 测试Vision Transformer图表分析 ===")
    
    config = VisionConfig()
    model = ViTChartAnalyzer(config)
    
    # 创建测试图像
    batch_size = 2
    test_images = torch.randn(batch_size, 3, config.image_size, config.image_size)
    
    # 创建测试技术指标
    test_indicators = {
        "sma": torch.randn(batch_size, 5),
        "ema": torch.randn(batch_size, 5),
        "rsi": torch.randn(batch_size, 1),
        "macd": torch.randn(batch_size, 3)
    }
    
    # 前向传播
    with torch.no_grad():
        outputs = model(test_images, test_indicators)
    
    print(f"交易信号形状: {outputs['trading_signal'].shape}")
    print(f"信心度形状: {outputs['confidence_score'].shape}")
    print(f"ViT特征形状: {outputs['vit_features'].shape}")
    print(f"技术特征形状: {outputs['technical_features'].shape}")
    print(f"模式识别结果数量: {len(outputs['pattern_results']['pattern_names'])}")
    
    # 测试图表图像生成
    print("\n=== 测试图表图像生成 ===")
    
    # 创建模拟OHLCV数据
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    np.random.seed(42)
    ohlcv_data = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.randn(50) * 0.5),
        'High': 100 + np.cumsum(np.random.randn(50) * 0.5) + np.random.rand(50) * 2,
        'Low': 100 + np.cumsum(np.random.randn(50) * 0.5) - np.random.rand(50) * 2,
        'Close': 100 + np.cumsum(np.random.randn(50) * 0.5),
        'Volume': np.random.randint(1000000, 5000000, 50)
    }, index=dates)
    
    # 确保High是最高价，Low是最低价
    ohlcv_data['High'] = ohlcv_data[['Open', 'High', 'Close']].max(axis=1) + 0.5
    ohlcv_data['Low'] = ohlcv_data[['Open', 'Low', 'Close']].min(axis=1) - 0.5
    
    try:
        chart_image = create_chart_image(ohlcv_data)
        print(f"图表图像形状: {chart_image.shape}")
        print(f"图像数值范围: [{chart_image.min():.3f}, {chart_image.max():.3f}]")
    except Exception as e:
        print(f"图表生成测试跳过 (需要matplotlib): {e}")
    
    print("\n=== Vision Transformer测试完成 ===")