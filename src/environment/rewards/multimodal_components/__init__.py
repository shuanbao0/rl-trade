"""
多模态奖励函数组件模块

集成Vision Transformer图表分析、BERT/GPT文本分析和跨模态注意力机制
"""

from .vision_transformer import (
    ViTChartAnalyzer,
    ChartFeatureExtractor,
    TechnicalIndicatorEncoder,
    CandlestickPatternRecognizer,
    create_chart_image
)

from .text_analysis import (
    FinancialBERTAnalyzer,
    NewssentimentAnalyzer, 
    GPTFundamentalAnalyzer,
    MarketContextExtractor,
    create_sample_news_articles,
    NewsArticle
)

from .cross_modal_fusion import (
    CrossModalAttention,
    MultimodalFusionTransformer,
    TemporalFusionLayer,
    AdaptiveFusionGate,
    create_multimodal_fusion_model
)

from .multimodal_config import (
    MultimodalConfig,
    VisionConfig,
    TextConfig,
    FusionConfig,
    DataModalityType,
    create_lightweight_config,
    create_production_config
)

__all__ = [
    # Vision Transformer Components
    'ViTChartAnalyzer',
    'ChartFeatureExtractor', 
    'TechnicalIndicatorEncoder',
    'CandlestickPatternRecognizer',
    'create_chart_image',
    
    # Text Analysis Components
    'FinancialBERTAnalyzer',
    'NewssentimentAnalyzer',
    'GPTFundamentalAnalyzer', 
    'MarketContextExtractor',
    'create_sample_news_articles',
    'NewsArticle',
    
    # Cross-Modal Fusion Components
    'CrossModalAttention',
    'MultimodalFusionTransformer',
    'TemporalFusionLayer',
    'AdaptiveFusionGate',
    'create_multimodal_fusion_model',
    
    # Configuration
    'MultimodalConfig',
    'VisionConfig',
    'TextConfig',
    'FusionConfig',
    'DataModalityType',
    'create_lightweight_config',
    'create_production_config'
]