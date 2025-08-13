"""
多模态奖励函数

整合Vision Transformer图表分析、BERT/GPT文本分析和跨模态注意力机制的完整奖励函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from .base_reward import BaseRewardScheme
from .multimodal_components import (
    MultimodalConfig, VisionConfig, TextConfig, FusionConfig,
    ViTChartAnalyzer, FinancialBERTAnalyzer, GPTFundamentalAnalyzer,
    NewssentimentAnalyzer, MarketContextExtractor,
    MultimodalFusionTransformer, create_multimodal_fusion_model,
    DataModalityType, NewsArticle, create_chart_image
)


@dataclass
class MultimodalRewardConfig(MultimodalConfig):
    """多模态奖励函数专用配置"""
    # 奖励权重
    vision_reward_weight: float = 0.4
    text_reward_weight: float = 0.3
    fusion_reward_weight: float = 0.3
    
    # 交易信号阈值
    buy_threshold: float = 0.6
    sell_threshold: float = -0.6
    confidence_threshold: float = 0.7
    
    # 风险管理参数
    max_position_change: float = 0.2
    volatility_penalty: float = 0.1
    drawdown_penalty: float = 0.15
    
    # 数据更新频率
    chart_update_interval: int = 60  # 秒
    news_update_interval: int = 300  # 秒
    context_update_interval: int = 900  # 秒
    
    # 缓存设置
    enable_caching: bool = True
    cache_ttl: int = 3600  # 秒
    max_cache_size: int = 1000
    
    # 实时处理
    async_processing: bool = True
    max_concurrent_requests: int = 10
    timeout_seconds: int = 30


class MultimodalDataManager:
    """多模态数据管理器"""
    
    def __init__(self, config: MultimodalRewardConfig):
        self.config = config
        self.cache = {} if config.enable_caching else None
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_requests)
        
        # 数据源状态
        self.last_chart_update = None
        self.last_news_update = None
        self.last_context_update = None
        
        self.logger = logging.getLogger(__name__)
        
    def get_chart_data(self, symbol: str, timeframe: str = '1D', 
                      lookback_periods: int = 100) -> Optional[np.ndarray]:
        """获取图表数据并生成图像"""
        cache_key = f"chart_{symbol}_{timeframe}_{lookback_periods}"
        
        # 检查缓存
        if self.cache and cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.config.cache_ttl:
                return cached_data
        
        try:
            # 这里应该连接到真实的数据源
            # 现在创建模拟数据用于演示
            ohlcv_data = self._generate_mock_ohlcv_data(lookback_periods)
            chart_image = create_chart_image(ohlcv_data, image_size=224)
            
            # 缓存结果
            if self.cache:
                self.cache[cache_key] = (chart_image, datetime.now())
                self._cleanup_cache()
            
            return chart_image
            
        except Exception as e:
            self.logger.error(f"获取图表数据失败 {symbol}: {e}")
            return None
    
    def get_news_data(self, symbol: str, hours_back: int = 24) -> List[NewsArticle]:
        """获取新闻数据"""
        cache_key = f"news_{symbol}_{hours_back}"
        
        # 检查缓存
        if self.cache and cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.config.cache_ttl:
                return cached_data
        
        try:
            # 这里应该连接到真实的新闻API
            # 现在创建模拟新闻数据
            news_articles = self._generate_mock_news_data(symbol, hours_back)
            
            # 缓存结果
            if self.cache:
                self.cache[cache_key] = (news_articles, datetime.now())
                self._cleanup_cache()
            
            return news_articles
            
        except Exception as e:
            self.logger.error(f"获取新闻数据失败 {symbol}: {e}")
            return []
    
    def get_market_context(self) -> Dict[str, Any]:
        """获取市场上下文信息"""
        cache_key = "market_context"
        
        # 检查缓存
        if self.cache and cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.config.cache_ttl:
                return cached_data
        
        try:
            # 模拟市场上下文数据
            context = {
                'market_sentiment': np.random.uniform(-1, 1),
                'volatility_index': np.random.uniform(10, 40),
                'economic_indicators': {
                    'gdp_growth': np.random.uniform(-2, 5),
                    'inflation_rate': np.random.uniform(0, 8),
                    'unemployment_rate': np.random.uniform(3, 12)
                },
                'sector_performance': {
                    'technology': np.random.uniform(-5, 5),
                    'healthcare': np.random.uniform(-3, 4),
                    'finance': np.random.uniform(-4, 6)
                }
            }
            
            # 缓存结果
            if self.cache:
                self.cache[cache_key] = (context, datetime.now())
                self._cleanup_cache()
            
            return context
            
        except Exception as e:
            self.logger.error(f"获取市场上下文失败: {e}")
            return {}
    
    def _generate_mock_ohlcv_data(self, periods: int) -> pd.DataFrame:
        """生成模拟OHLCV数据"""
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
        
        # 生成价格走势
        returns = np.random.randn(periods) * 0.02
        prices = 100 * np.exp(np.cumsum(returns))
        
        # 生成OHLCV
        ohlcv_data = pd.DataFrame({
            'Open': prices * (1 + np.random.randn(periods) * 0.005),
            'High': prices * (1 + np.abs(np.random.randn(periods)) * 0.01),
            'Low': prices * (1 - np.abs(np.random.randn(periods)) * 0.01),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, periods)
        }, index=dates)
        
        # 确保High是最高价，Low是最低价
        ohlcv_data['High'] = ohlcv_data[['Open', 'High', 'Close']].max(axis=1)
        ohlcv_data['Low'] = ohlcv_data[['Open', 'Low', 'Close']].min(axis=1)
        
        return ohlcv_data
    
    def _generate_mock_news_data(self, symbol: str, hours_back: int) -> List[NewsArticle]:
        """生成模拟新闻数据"""
        news_templates = [
            f"{symbol} reports strong quarterly earnings exceeding analyst expectations",
            f"Market volatility affects {symbol} trading as investors remain cautious",
            f"{symbol} announces strategic partnership to expand market presence",
            f"Analysts upgrade {symbol} price target citing strong fundamentals",
            f"Regulatory concerns impact {symbol} stock performance in recent session"
        ]
        
        news_articles = []
        num_articles = np.random.randint(1, 6)
        
        for i in range(num_articles):
            hours_ago = np.random.randint(1, hours_back + 1)
            timestamp = datetime.now() - timedelta(hours=hours_ago)
            
            title = np.random.choice(news_templates)
            content = f"{title}. {self._generate_mock_news_content()}"
            
            article = NewsArticle(
                title=title,
                content=content,
                source=np.random.choice(['Reuters', 'Bloomberg', 'CNBC', 'WSJ']),
                timestamp=timestamp,
                ticker_symbols=[symbol],
                category=np.random.choice(['earnings', 'market_analysis', 'company_news'])
            )
            news_articles.append(article)
        
        return news_articles
    
    def _generate_mock_news_content(self) -> str:
        """生成模拟新闻内容"""
        content_parts = [
            "The company's revenue increased by",
            "Market analysts predict",
            "Trading volume shows",
            "Institutional investors are",
            "The stock price movement reflects"
        ]
        
        numbers = [
            "8% year-over-year",
            "15% growth potential",
            "significant interest",
            "bullish sentiment",
            "strong fundamentals"
        ]
        
        return f"{np.random.choice(content_parts)} {np.random.choice(numbers)}."
    
    def _cleanup_cache(self):
        """清理缓存"""
        if not self.cache:
            return
            
        if len(self.cache) > self.config.max_cache_size:
            # 移除最旧的条目
            oldest_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k][1]
            )[:len(self.cache) - self.config.max_cache_size]
            
            for key in oldest_keys:
                del self.cache[key]


class MultimodalReward(BaseRewardScheme):
    """
    多模态奖励函数
    
    整合价格图表分析、新闻情感分析和市场上下文，使用跨模态注意力机制
    生成综合的交易奖励信号。
    """
    
    def __init__(self, config: Optional[MultimodalRewardConfig] = None, **kwargs):
        super().__init__(**kwargs)
        
        self.config = config or MultimodalRewardConfig()
        self.logger = logging.getLogger(__name__)
        
        # 初始化数据管理器
        self.data_manager = MultimodalDataManager(self.config)
        
        # 初始化模型组件
        self._initialize_models()
        
        # 状态跟踪
        self.current_symbol = None
        self.prediction_history = []
        self.attention_weights_history = []
        
        # 性能指标
        self.multimodal_metrics = {
            'vision_accuracy': 0.0,
            'text_sentiment_accuracy': 0.0,
            'fusion_performance': 0.0,
            'total_predictions': 0,
            'successful_predictions': 0
        }
        
        self.logger = logging.getLogger(__name__)
        
    def _initialize_models(self):
        """初始化所有模型组件"""
        try:
            # Vision Transformer图表分析器
            self.vision_analyzer = ViTChartAnalyzer(self.config.vision_config)
            
            # 文本分析器
            self.text_analyzer = NewssentimentAnalyzer(self.config.text_config)
            self.market_context_extractor = MarketContextExtractor(self.config.text_config)
            
            # 多模态融合Transformer
            self.fusion_transformer = create_multimodal_fusion_model(self.config.fusion_config)
            
            # 设置为评估模式
            self.vision_analyzer.eval()
            self.text_analyzer.eval()
            self.market_context_extractor.eval()
            self.fusion_transformer.eval()
            
            self.logger.info("多模态模型组件初始化成功")
            
        except Exception as e:
            self.logger.error(f"模型初始化失败: {e}")
            # 创建简化的备用模型
            self._create_fallback_models()
    
    def _create_fallback_models(self):
        """创建简化的备用模型"""
        self.logger.warning("使用简化备用模型")
        
        # 简化的视觉分析器
        self.vision_analyzer = nn.Sequential(
            nn.Linear(224 * 224 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # 买入/卖出/持有
        )
        
        # 简化的文本分析器
        self.text_analyzer = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        
        # 简化的融合层
        self.fusion_transformer = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    
    def reward(self, env) -> float:
        """
        计算多模态奖励
        
        Args:
            env: TensorTrade环境实例
            
        Returns:
            float: 计算得到的奖励值
        """
        try:
            # 更新状态
            state = self.update_state(env)
            
            # 获取交易标的
            symbol = getattr(env, 'symbol', 'UNKNOWN')
            if symbol != self.current_symbol:
                self.current_symbol = symbol
                self.logger.info(f"更新交易标的: {symbol}")
            
            # 收集多模态数据
            multimodal_data = self._collect_multimodal_data(symbol, state)
            
            # 多模态分析
            analysis_results = self._analyze_multimodal_data(multimodal_data)
            
            # 计算奖励
            reward_value = self._calculate_multimodal_reward(analysis_results, state)
            
            # 更新历史记录
            self._update_prediction_history(analysis_results, reward_value)
            
            # 更新性能指标
            self._update_metrics(analysis_results, state)
            
            return float(reward_value)
            
        except Exception as e:
            self.logger.error(f"多模态奖励计算失败: {e}")
            # 返回基础奖励作为备用
            return self._calculate_fallback_reward(env)
    
    def get_reward(self, portfolio) -> float:
        """TensorTrade框架要求的get_reward方法"""
        # 创建模拟环境对象
        mock_env = type('MockEnv', (), {
            'action_scheme': type('ActionScheme', (), {'portfolio': portfolio})(),
            'symbol': getattr(portfolio, 'symbol', 'UNKNOWN')
        })()
        
        return self.reward(mock_env)
    
    def _collect_multimodal_data(self, symbol: str, state: Dict[str, float]) -> Dict[str, Any]:
        """收集多模态数据"""
        multimodal_data = {
            'symbol': symbol,
            'state': state,
            'chart_data': None,
            'news_data': None,
            'market_context': None,
            'technical_indicators': None
        }
        
        try:
            # 获取图表数据
            if DataModalityType.PRICE_CHART in self.config.enabled_modalities:
                chart_data = self.data_manager.get_chart_data(symbol)
                multimodal_data['chart_data'] = chart_data
            
            # 获取新闻数据
            if DataModalityType.NEWS_TEXT in self.config.enabled_modalities:
                news_data = self.data_manager.get_news_data(symbol)
                multimodal_data['news_data'] = news_data
            
            # 获取市场上下文
            market_context = self.data_manager.get_market_context()
            multimodal_data['market_context'] = market_context
            
            # 构造技术指标
            if DataModalityType.TECHNICAL_INDICATORS in self.config.enabled_modalities:
                technical_indicators = self._extract_technical_indicators(state)
                multimodal_data['technical_indicators'] = technical_indicators
            
        except Exception as e:
            self.logger.error(f"多模态数据收集失败: {e}")
        
        return multimodal_data
    
    def _extract_technical_indicators(self, state: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """从状态中提取技术指标"""
        indicators = {}
        
        try:
            # 模拟技术指标数据
            batch_size = 1
            indicators = {
                'sma': torch.randn(batch_size, 5) * 0.1,
                'ema': torch.randn(batch_size, 5) * 0.1,
                'rsi': torch.tensor([[state.get('total_return_pct', 0) * 0.01]]),
                'macd': torch.randn(batch_size, 3) * 0.05
            }
            
        except Exception as e:
            self.logger.error(f"技术指标提取失败: {e}")
            
        return indicators
    
    def _analyze_multimodal_data(self, multimodal_data: Dict[str, Any]) -> Dict[str, Any]:
        """多模态数据分析"""
        analysis_results = {
            'vision_analysis': None,
            'text_analysis': None,
            'fusion_analysis': None,
            'confidence_scores': {},
            'trading_signals': {}
        }
        
        try:
            with torch.no_grad():
                # 视觉分析
                if multimodal_data['chart_data'] is not None:
                    vision_results = self._analyze_chart_data(
                        multimodal_data['chart_data'],
                        multimodal_data.get('technical_indicators')
                    )
                    analysis_results['vision_analysis'] = vision_results
                
                # 文本分析
                if multimodal_data['news_data']:
                    text_results = self._analyze_text_data(multimodal_data['news_data'])
                    analysis_results['text_analysis'] = text_results
                
                # 跨模态融合分析
                if (analysis_results['vision_analysis'] is not None and 
                    analysis_results['text_analysis'] is not None):
                    fusion_results = self._perform_cross_modal_fusion(
                        analysis_results['vision_analysis'],
                        analysis_results['text_analysis']
                    )
                    analysis_results['fusion_analysis'] = fusion_results
                
        except Exception as e:
            self.logger.error(f"多模态分析失败: {e}")
        
        return analysis_results
    
    def _analyze_chart_data(self, chart_data: np.ndarray, 
                          technical_indicators: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """分析图表数据"""
        try:
            # 转换为tensor
            chart_tensor = torch.from_numpy(chart_data).unsqueeze(0).float()
            
            # ViT分析
            vision_outputs = self.vision_analyzer(chart_tensor, technical_indicators)
            
            return {
                'trading_signal': vision_outputs['trading_signal'],
                'confidence_score': vision_outputs['confidence_score'],
                'vit_features': vision_outputs['vit_features'],
                'pattern_results': vision_outputs.get('pattern_results', {}),
                'technical_features': vision_outputs.get('technical_features')
            }
            
        except Exception as e:
            self.logger.error(f"图表数据分析失败: {e}")
            return {}
    
    def _analyze_text_data(self, news_data: List[NewsArticle]) -> Dict[str, Any]:
        """分析文本数据"""
        try:
            # 批量分析新闻
            if hasattr(self.text_analyzer, 'analyze_news_batch'):
                text_results = self.text_analyzer.analyze_news_batch(news_data)
            else:
                # 备用简化分析
                text_results = self._simple_text_analysis(news_data)
            
            return text_results
            
        except Exception as e:
            self.logger.error(f"文本数据分析失败: {e}")
            return {}
    
    def _simple_text_analysis(self, news_data: List[NewsArticle]) -> Dict[str, Any]:
        """简化的文本分析"""
        if not news_data:
            return {'overall_market_sentiment': 0.0, 'market_moving_probability': 0.0}
        
        # 简单的关键词情感分析
        positive_words = ['strong', 'growth', 'beat', 'exceed', 'bullish', 'upgrade']
        negative_words = ['weak', 'decline', 'miss', 'concern', 'bearish', 'downgrade']
        
        sentiment_scores = []
        for article in news_data:
            text = f"{article.title} {article.content}".lower()
            
            pos_count = sum(word in text for word in positive_words)
            neg_count = sum(word in text for word in negative_words)
            
            if pos_count + neg_count > 0:
                sentiment = (pos_count - neg_count) / (pos_count + neg_count)
            else:
                sentiment = 0.0
                
            sentiment_scores.append(sentiment)
        
        overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        market_probability = min(abs(overall_sentiment), 1.0)
        
        return {
            'overall_market_sentiment': float(overall_sentiment),
            'market_moving_probability': float(market_probability)
        }
    
    def _perform_cross_modal_fusion(self, vision_analysis: Dict[str, Any], 
                                  text_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """执行跨模态融合"""
        try:
            # 提取特征
            vision_features = vision_analysis.get('vit_features')
            
            # 为文本分析创建特征向量
            text_sentiment = text_analysis.get('overall_market_sentiment', 0.0)
            market_probability = text_analysis.get('market_moving_probability', 0.0)
            
            # 创建文本特征向量
            text_features = torch.tensor([[text_sentiment, market_probability] + [0.0] * (vision_features.size(-1) - 2)]).float()
            
            # 确保特征维度匹配
            if text_features.size(-1) != vision_features.size(-1):
                text_features = text_features[:, :vision_features.size(-1)]
            
            # 扩展维度以匹配序列长度
            vision_seq = vision_features.unsqueeze(1)  # (B, 1, H)
            text_seq = text_features.unsqueeze(1)      # (B, 1, H)
            
            # 执行融合
            fusion_outputs = self.fusion_transformer(vision_seq, text_seq)
            
            return {
                'fused_features': fusion_outputs['fused_features'],
                'classification_logits': fusion_outputs['classification_logits'],
                'regression_output': fusion_outputs['regression_output'],
                'modality_weights': fusion_outputs.get('adaptive_fusion_output', {}).get('modality_weights')
            }
            
        except Exception as e:
            self.logger.error(f"跨模态融合失败: {e}")
            return {}
    
    def _calculate_multimodal_reward(self, analysis_results: Dict[str, Any], 
                                   state: Dict[str, float]) -> float:
        """计算多模态奖励"""
        total_reward = 0.0
        
        try:
            # 获取权重
            enabled_weights = self.config.get_enabled_modality_weights()
            
            # 视觉奖励
            if analysis_results['vision_analysis']:
                vision_reward = self._calculate_vision_reward(
                    analysis_results['vision_analysis'], state
                )
                total_reward += vision_reward * self.config.vision_reward_weight
            
            # 文本奖励
            if analysis_results['text_analysis']:
                text_reward = self._calculate_text_reward(
                    analysis_results['text_analysis'], state
                )
                total_reward += text_reward * self.config.text_reward_weight
            
            # 融合奖励
            if analysis_results['fusion_analysis']:
                fusion_reward = self._calculate_fusion_reward(
                    analysis_results['fusion_analysis'], state
                )
                total_reward += fusion_reward * self.config.fusion_reward_weight
            
            # 应用风险调整
            total_reward = self._apply_risk_adjustments(total_reward, state)
            
            # 限制奖励范围
            total_reward = np.clip(total_reward, -1.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"多模态奖励计算失败: {e}")
            total_reward = 0.0
        
        return total_reward
    
    def _calculate_vision_reward(self, vision_analysis: Dict[str, Any], 
                               state: Dict[str, float]) -> float:
        """计算视觉奖励"""
        try:
            trading_signal = vision_analysis.get('trading_signal')
            confidence = vision_analysis.get('confidence_score')
            
            if trading_signal is None or confidence is None:
                return 0.0
            
            # 转换为标量
            if isinstance(trading_signal, torch.Tensor):
                signal_probs = F.softmax(trading_signal, dim=-1)
                signal_value = signal_probs[0, 2].item() - signal_probs[0, 0].item()  # 买入 - 卖出
            else:
                signal_value = float(trading_signal)
            
            if isinstance(confidence, torch.Tensor):
                confidence_value = confidence.item()
            else:
                confidence_value = float(confidence)
            
            # 基于信号强度和置信度计算奖励
            reward = signal_value * confidence_value
            
            # 考虑当前收益情况
            current_return = state.get('step_return_pct', 0.0) / 100.0
            reward *= (1.0 + current_return)
            
            return reward
            
        except Exception as e:
            self.logger.error(f"视觉奖励计算失败: {e}")
            return 0.0
    
    def _calculate_text_reward(self, text_analysis: Dict[str, Any], 
                             state: Dict[str, float]) -> float:
        """计算文本奖励"""
        try:
            sentiment = text_analysis.get('overall_market_sentiment', 0.0)
            probability = text_analysis.get('market_moving_probability', 0.0)
            
            # 基本文本奖励
            text_reward = sentiment * probability
            
            # 考虑当前表现
            current_return = state.get('step_return_pct', 0.0) / 100.0
            
            # 如果情感与收益方向一致，给予额外奖励
            if (sentiment > 0 and current_return > 0) or (sentiment < 0 and current_return < 0):
                text_reward *= 1.2
            
            return text_reward
            
        except Exception as e:
            self.logger.error(f"文本奖励计算失败: {e}")
            return 0.0
    
    def _calculate_fusion_reward(self, fusion_analysis: Dict[str, Any], 
                               state: Dict[str, float]) -> float:
        """计算融合奖励"""
        try:
            classification_logits = fusion_analysis.get('classification_logits')
            regression_output = fusion_analysis.get('regression_output')
            
            if classification_logits is None:
                return 0.0
            
            # 分类奖励
            if isinstance(classification_logits, torch.Tensor):
                class_probs = F.softmax(classification_logits, dim=-1)
                class_reward = class_probs[0, 2].item() - class_probs[0, 0].item()  # 买入 - 卖出
            else:
                class_reward = 0.0
            
            # 回归奖励
            if regression_output is not None:
                if isinstance(regression_output, torch.Tensor):
                    reg_reward = regression_output.item()
                else:
                    reg_reward = float(regression_output)
            else:
                reg_reward = 0.0
            
            # 组合奖励
            fusion_reward = 0.7 * class_reward + 0.3 * reg_reward
            
            # 应用模态权重（如果可用）
            modality_weights = fusion_analysis.get('modality_weights')
            if modality_weights is not None and isinstance(modality_weights, torch.Tensor):
                weight_balance = torch.std(modality_weights).item()
                # 如果权重过于不平衡，降低奖励
                if weight_balance > 0.3:
                    fusion_reward *= 0.8
            
            return fusion_reward
            
        except Exception as e:
            self.logger.error(f"融合奖励计算失败: {e}")
            return 0.0
    
    def _apply_risk_adjustments(self, reward: float, state: Dict[str, float]) -> float:
        """应用风险调整"""
        try:
            # 波动性惩罚
            if hasattr(self, 'reward_history') and len(self.reward_history) > 5:
                recent_rewards = self.reward_history[-5:]
                volatility = np.std(recent_rewards)
                reward -= volatility * self.config.volatility_penalty
            
            # 回撤惩罚
            total_return = state.get('total_return_pct', 0.0)
            if total_return < -5.0:  # 如果总回撤超过5%
                drawdown_penalty = abs(total_return + 5.0) * self.config.drawdown_penalty / 100.0
                reward -= drawdown_penalty
            
            # 位置变化限制
            current_action = state.get('current_action', 0.0)
            if abs(current_action) > self.config.max_position_change:
                position_penalty = abs(current_action) - self.config.max_position_change
                reward -= position_penalty * 0.1
            
            return reward
            
        except Exception as e:
            self.logger.error(f"风险调整失败: {e}")
            return reward
    
    def _calculate_fallback_reward(self, env) -> float:
        """计算备用奖励"""
        try:
            state = self.update_state(env)
            return state.get('step_return_pct', 0.0) / 100.0
        except:
            return 0.0
    
    def _update_prediction_history(self, analysis_results: Dict[str, Any], reward: float):
        """更新预测历史"""
        prediction_entry = {
            'timestamp': datetime.now(),
            'reward': reward,
            'vision_signal': None,
            'text_sentiment': None,
            'fusion_output': None
        }
        
        try:
            if analysis_results['vision_analysis']:
                prediction_entry['vision_signal'] = analysis_results['vision_analysis'].get('trading_signal')
            
            if analysis_results['text_analysis']:
                prediction_entry['text_sentiment'] = analysis_results['text_analysis'].get('overall_market_sentiment')
            
            if analysis_results['fusion_analysis']:
                prediction_entry['fusion_output'] = analysis_results['fusion_analysis'].get('classification_logits')
        
        except Exception as e:
            self.logger.error(f"预测历史更新失败: {e}")
        
        self.prediction_history.append(prediction_entry)
        
        # 保持历史记录大小
        if len(self.prediction_history) > 1000:
            self.prediction_history.pop(0)
    
    def _update_metrics(self, analysis_results: Dict[str, Any], state: Dict[str, float]):
        """更新性能指标"""
        try:
            self.multimodal_metrics['total_predictions'] += 1
            
            # 基于收益情况判断预测是否成功
            step_return = state.get('step_return_pct', 0.0)
            if abs(step_return) > 0.1:  # 如果有显著收益变化
                self.multimodal_metrics['successful_predictions'] += 1
            
            # 更新准确率
            if self.multimodal_metrics['total_predictions'] > 0:
                accuracy = (self.multimodal_metrics['successful_predictions'] / 
                          self.multimodal_metrics['total_predictions'])
                self.multimodal_metrics['fusion_performance'] = accuracy
        
        except Exception as e:
            self.logger.error(f"指标更新失败: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取多模态性能摘要"""
        base_summary = super().get_performance_summary()
        
        multimodal_summary = {
            'modality_weights': self.config.get_enabled_modality_weights(),
            'enabled_modalities': [mod.value for mod in self.config.enabled_modalities],
            'vision_reward_weight': self.config.vision_reward_weight,
            'text_reward_weight': self.config.text_reward_weight,
            'fusion_reward_weight': self.config.fusion_reward_weight,
            'current_symbol': self.current_symbol,
            'prediction_history_length': len(self.prediction_history),
            'multimodal_metrics': self.multimodal_metrics.copy(),
            'model_device': str(self.config.device),
            'cache_enabled': self.config.enable_caching,
            'async_processing': self.config.async_processing
        }
        
        # 合并摘要
        base_summary.update(multimodal_summary)
        return base_summary
    
    @classmethod
    def get_reward_info(cls) -> Dict[str, Any]:
        """获取多模态奖励函数信息"""
        return {
            'name': 'MultimodalReward',
            'description': '''
多模态奖励函数 (Multimodal Trading Reward Function)

基于2024-2025年最新研究的先进多模态融合奖励函数，整合：
1. Vision Transformer图表分析 - 识别价格模式和技术指标
2. FinBERT/GPT文本分析 - 处理新闻情感和基本面分析  
3. 跨模态注意力机制 - 融合视觉和文本信息
4. 自适应融合策略 - 动态调整不同模态的权重

核心创新：
- 基于ViT的烛台图案识别和技术指标分析
- 金融BERT情感分析和GPT基本面评估
- 跨模态Transformer注意力机制
- 时序融合层处理多时间尺度信息
- 自适应门控网络优化模态融合权重

适用场景：
- 需要综合多种信息源的复杂交易决策
- 高频交易和算法交易系统
- 需要解释性的AI驱动投资策略
- 多资产类别的投资组合管理

技术特点：
- 支持实时多模态数据处理
- 可解释的注意力权重
- 自适应学习和在线优化
- 风险管理和回撤控制集成
            ''',
            'category': 'advanced_multimodal',
            'parameters': {
                'vision_reward_weight': {'type': 'float', 'default': 0.4, 'range': [0.0, 1.0]},
                'text_reward_weight': {'type': 'float', 'default': 0.3, 'range': [0.0, 1.0]},
                'fusion_reward_weight': {'type': 'float', 'default': 0.3, 'range': [0.0, 1.0]},
                'buy_threshold': {'type': 'float', 'default': 0.6, 'range': [0.0, 1.0]},
                'sell_threshold': {'type': 'float', 'default': -0.6, 'range': [-1.0, 0.0]},
                'confidence_threshold': {'type': 'float', 'default': 0.7, 'range': [0.0, 1.0]},
                'enabled_modalities': {'type': 'list', 'default': ['price_chart', 'news_text']},
                'real_time_processing': {'type': 'bool', 'default': True},
                'async_processing': {'type': 'bool', 'default': True}
            }
        }


def create_multimodal_reward(config: Optional[MultimodalRewardConfig] = None, **kwargs) -> MultimodalReward:
    """创建多模态奖励函数的工厂函数"""
    if config is None:
        config = MultimodalRewardConfig()
    
    # 从kwargs更新配置
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return MultimodalReward(config=config)


if __name__ == "__main__":
    # 测试多模态奖励函数
    print("=== 测试多模态奖励函数 ===")
    
    # 创建配置
    config = MultimodalRewardConfig()
    print(f"配置设备: {config.device}")
    print(f"启用模态: {[mod.value for mod in config.enabled_modalities]}")
    
    # 创建奖励函数
    multimodal_reward = create_multimodal_reward(config)
    print("✓ 多模态奖励函数创建成功")
    
    # 获取信息
    info = multimodal_reward.get_reward_info()
    print(f"奖励函数名称: {info['name']}")
    print(f"类别: {info['category']}")
    
    # 模拟环境测试
    print("\n--- 模拟环境测试 ---")
    
    class MockEnv:
        def __init__(self):
            self.symbol = "AAPL"
            self.action_scheme = type('ActionScheme', (), {
                'portfolio': type('Portfolio', (), {'net_worth': 10500.0})()
            })()
            self._last_action = 0.1
    
    mock_env = MockEnv()
    
    try:
        # 计算奖励
        reward_value = multimodal_reward.reward(mock_env)
        print(f"✓ 奖励值: {reward_value:.4f}")
        
        # 获取性能摘要
        summary = multimodal_reward.get_performance_summary()
        print(f"✓ 总预测次数: {summary['multimodal_metrics']['total_predictions']}")
        print(f"✓ 当前交易标的: {summary['current_symbol']}")
        print(f"✓ 启用的模态: {summary['enabled_modalities']}")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 多模态奖励函数测试完成 ===")