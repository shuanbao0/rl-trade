"""
多模态奖励函数配置模块

定义多模态融合的配置参数和数据类型
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import torch


class DataModalityType(Enum):
    """数据模态类型枚举"""
    PRICE_CHART = "price_chart"
    TECHNICAL_INDICATORS = "technical_indicators"
    NEWS_TEXT = "news_text"
    FUNDAMENTAL_DATA = "fundamental_data"
    MARKET_SENTIMENT = "market_sentiment"
    MACROECONOMIC_DATA = "macroeconomic_data"
    EARNINGS_CALLS = "earnings_calls"
    SOCIAL_MEDIA = "social_media"


class FusionStrategy(Enum):
    """融合策略枚举"""
    EARLY_FUSION = "early_fusion"
    INTERMEDIATE_FUSION = "intermediate_fusion"
    LATE_FUSION = "late_fusion"
    HIERARCHICAL_FUSION = "hierarchical_fusion"
    ADAPTIVE_FUSION = "adaptive_fusion"


class AttentionType(Enum):
    """注意力机制类型"""
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    MULTI_HEAD_ATTENTION = "multi_head_attention"
    GATED_ATTENTION = "gated_attention"


@dataclass
class VisionConfig:
    """Vision Transformer配置"""
    # ViT架构参数
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # 图表分析特定参数
    chart_types: List[str] = field(default_factory=lambda: [
        "candlestick", "line", "volume", "rsi", "macd", "bollinger_bands"
    ])
    technical_indicators: List[str] = field(default_factory=lambda: [
        "sma", "ema", "rsi", "macd", "bollinger_bands", "atr", "obv", "stochastic"
    ])  
    time_horizons: List[str] = field(default_factory=lambda: [
        "1min", "5min", "15min", "30min", "1hour", "4hour", "1day"
    ])
    
    # 预处理参数
    normalize_prices: bool = True
    augmentation_enabled: bool = True
    noise_reduction: bool = True


@dataclass
class TextConfig:
    """文本分析配置"""
    # BERT/GPT模型参数
    model_name: str = "ProsusAI/finbert"
    max_sequence_length: int = 512
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    hidden_dropout_prob: float = 0.1
    
    # GPT生成参数
    gpt_model_name: str = "gpt-3.5-turbo"
    max_tokens: int = 150
    temperature: float = 0.3
    
    # 文本源配置
    news_sources: List[str] = field(default_factory=lambda: [
        "reuters", "bloomberg", "wsj", "cnbc", "marketwatch", "seeking_alpha"
    ])
    social_media_sources: List[str] = field(default_factory=lambda: [
        "twitter", "reddit", "stocktwits", "discord"
    ])
    
    # 情感分析参数
    sentiment_threshold: float = 0.1
    confidence_threshold: float = 0.8
    batch_size: int = 32
    
    # 语言处理参数
    languages: List[str] = field(default_factory=lambda: ["en", "zh"])
    text_preprocessing: bool = True
    remove_stopwords: bool = True


@dataclass
class FusionConfig:
    """跨模态融合配置"""
    # 融合策略
    fusion_strategy: FusionStrategy = FusionStrategy.HIERARCHICAL_FUSION
    attention_type: AttentionType = AttentionType.CROSS_ATTENTION
    
    # 注意力机制参数
    num_attention_heads: int = 8
    hidden_size: int = 512
    intermediate_size: int = 2048
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    
    # 跨模态对齐参数
    projection_dim: int = 256
    temperature: float = 0.07
    margin: float = 0.2
    
    # 时序融合参数
    temporal_window: int = 60
    temporal_stride: int = 1
    use_temporal_attention: bool = True
    
    # 自适应融合参数
    use_adaptive_weights: bool = True
    weight_learning_rate: float = 0.001
    weight_regularization: float = 0.01
    
    # 特征融合维度
    vision_feature_dim: int = 768
    text_feature_dim: int = 768
    fused_feature_dim: int = 1024
    
    # 融合层配置
    num_fusion_layers: int = 3
    residual_connections: bool = True
    layer_norm: bool = True


@dataclass
class MultimodalConfig:
    """多模态奖励函数总配置"""
    # 子配置
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    text_config: TextConfig = field(default_factory=TextConfig)
    fusion_config: FusionConfig = field(default_factory=FusionConfig)
    
    # 数据模态权重
    modality_weights: Dict[str, float] = field(default_factory=lambda: {
        "price_chart": 0.3,
        "technical_indicators": 0.2,
        "news_text": 0.2,
        "fundamental_data": 0.15,
        "market_sentiment": 0.1,
        "macroeconomic_data": 0.05
    })
    
    # 启用的数据模态
    enabled_modalities: List[DataModalityType] = field(default_factory=lambda: [
        DataModalityType.PRICE_CHART,
        DataModalityType.TECHNICAL_INDICATORS,
        DataModalityType.NEWS_TEXT,
        DataModalityType.MARKET_SENTIMENT
    ])
    
    # 训练参数
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 50
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    compile_model: bool = True
    
    # 缓存和性能
    cache_size: int = 1000
    prefetch_factor: int = 2
    num_workers: int = 4
    persistent_workers: bool = True
    
    # 模型保存和加载
    model_save_path: str = "models/multimodal/"
    checkpoint_frequency: int = 100
    early_stopping_patience: int = 10
    
    # 评估指标
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1_score", "auc_roc",
        "sharpe_ratio", "max_drawdown", "profit_factor"
    ])
    
    # 实时数据处理
    real_time_processing: bool = True
    data_refresh_interval: int = 60  # 秒
    max_data_latency: int = 300  # 秒
    
    # 风险管理
    max_position_size: float = 0.1
    risk_free_rate: float = 0.02
    volatility_window: int = 20
    
    # 调试和日志
    debug_mode: bool = False
    log_level: str = "INFO"
    save_intermediate_outputs: bool = False
    
    def __post_init__(self):
        """配置后处理"""
        # 验证权重总和
        total_weight = sum(self.modality_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            # 归一化权重
            self.modality_weights = {
                k: v / total_weight for k, v in self.modality_weights.items()
            }
        
        # 验证设备可用性
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            self.mixed_precision = False
    
    def get_enabled_modality_weights(self) -> Dict[str, float]:
        """获取启用模态的权重"""
        enabled_names = [mod.value for mod in self.enabled_modalities]
        enabled_weights = {
            name: weight for name, weight in self.modality_weights.items()
            if name in enabled_names
        }
        
        # 重新归一化
        total_weight = sum(enabled_weights.values())
        if total_weight > 0:
            enabled_weights = {
                k: v / total_weight for k, v in enabled_weights.items()
            }
        
        return enabled_weights
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "vision_config": self.vision_config.__dict__,
            "text_config": self.text_config.__dict__,
            "fusion_config": self.fusion_config.__dict__,
            "modality_weights": self.modality_weights,
            "enabled_modalities": [mod.value for mod in self.enabled_modalities],
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "device": self.device,
            "real_time_processing": self.real_time_processing
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MultimodalConfig':
        """从字典创建配置"""
        config = cls()
        
        # 更新子配置
        if "vision_config" in config_dict:
            for key, value in config_dict["vision_config"].items():
                setattr(config.vision_config, key, value)
        
        if "text_config" in config_dict:
            for key, value in config_dict["text_config"].items():
                setattr(config.text_config, key, value)
                
        if "fusion_config" in config_dict:
            for key, value in config_dict["fusion_config"].items():
                setattr(config.fusion_config, key, value)
        
        # 更新主配置
        for key, value in config_dict.items():
            if key not in ["vision_config", "text_config", "fusion_config"]:
                if key == "enabled_modalities":
                    config.enabled_modalities = [
                        DataModalityType(mod) for mod in value
                    ]
                elif hasattr(config, key):
                    setattr(config, key, value)
        
        return config


def create_lightweight_config() -> MultimodalConfig:
    """创建轻量级配置（用于资源受限环境）"""
    config = MultimodalConfig()
    
    # 减少模型大小
    config.vision_config.hidden_size = 384
    config.vision_config.num_hidden_layers = 6
    config.vision_config.num_attention_heads = 6
    config.vision_config.intermediate_size = 1536
    
    config.text_config.max_sequence_length = 256
    config.text_config.hidden_size = 384
    config.text_config.num_hidden_layers = 6
    
    config.fusion_config.hidden_size = 256
    config.fusion_config.num_fusion_layers = 2
    config.fusion_config.fused_feature_dim = 512
    
    # 减少批次大小
    config.batch_size = 8
    config.cache_size = 500
    
    return config


def create_production_config() -> MultimodalConfig:
    """创建生产环境配置"""
    config = MultimodalConfig()
    
    # 增强模型能力
    config.vision_config.hidden_size = 1024
    config.vision_config.num_hidden_layers = 16
    config.vision_config.num_attention_heads = 16
    config.vision_config.intermediate_size = 4096
    
    config.text_config.max_sequence_length = 1024
    config.text_config.hidden_size = 1024
    config.text_config.num_hidden_layers = 16
    
    config.fusion_config.hidden_size = 768
    config.fusion_config.num_fusion_layers = 4
    config.fusion_config.fused_feature_dim = 1536
    
    # 优化性能
    config.batch_size = 32
    config.cache_size = 2000
    config.num_workers = 8
    config.compile_model = True
    
    return config


if __name__ == "__main__":
    # 测试配置创建
    print("=== 测试多模态配置 ===")
    
    # 标准配置
    config = MultimodalConfig()
    print(f"标准配置设备: {config.device}")
    print(f"启用的模态: {[mod.value for mod in config.enabled_modalities]}")
    print(f"模态权重: {config.get_enabled_modality_weights()}")
    
    # 轻量级配置
    lightweight_config = create_lightweight_config()
    print(f"\n轻量级配置 - Vision隐藏层大小: {lightweight_config.vision_config.hidden_size}")
    print(f"轻量级配置 - 批次大小: {lightweight_config.batch_size}")
    
    # 生产配置
    production_config = create_production_config()
    print(f"\n生产配置 - Vision隐藏层大小: {production_config.vision_config.hidden_size}")
    print(f"生产配置 - 工作进程数: {production_config.num_workers}")
    
    # 配置序列化测试
    config_dict = config.to_dict()
    restored_config = MultimodalConfig.from_dict(config_dict)
    print(f"\n配置序列化测试 - 成功: {config.device == restored_config.device}")
    
    print("\n=== 多模态配置测试完成 ===")