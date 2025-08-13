"""
BERT/GPT文本分析模块

实现基于BERT和GPT的金融新闻情感分析、基本面分析和市场上下文提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import re
import json
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from .multimodal_config import TextConfig, DataModalityType

# 尝试导入transformers库
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoConfig,
        BertTokenizer, BertModel, BertConfig,
        GPT2Tokenizer, GPT2LMHeadModel, GPT2Config,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers library not available. Using mock implementations.")
    TRANSFORMERS_AVAILABLE = False
    
    # Mock classes for testing when transformers is not available
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name):
            return MockTokenizer()
    
    class AutoModel:
        @staticmethod
        def from_pretrained(model_name):
            return MockModel()
    
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.randint(0, 1000, (1, 50)), 
                   "attention_mask": torch.ones(1, 50)}
        
        def decode(self, ids, **kwargs):
            return "Mock decoded text"
    
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type('Config', (), {'hidden_size': 768})()
        
        def forward(self, input_ids, attention_mask=None, **kwargs):
            batch_size, seq_len = input_ids.shape
            return type('Output', (), {
                'last_hidden_state': torch.randn(batch_size, seq_len, 768),
                'pooler_output': torch.randn(batch_size, 768)
            })()


@dataclass
class NewsArticle:
    """新闻文章数据结构"""
    title: str
    content: str
    source: str
    timestamp: datetime
    url: Optional[str] = None
    category: Optional[str] = None
    ticker_symbols: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SentimentAnalysisResult:
    """情感分析结果"""
    sentiment: str  # positive, negative, neutral
    confidence: float
    scores: Dict[str, float]  # {'positive': 0.8, 'negative': 0.1, 'neutral': 0.1}
    key_phrases: List[str]
    market_relevance: float


@dataclass
class FundamentalAnalysisResult:
    """基本面分析结果"""
    financial_health: str  # strong, moderate, weak
    growth_prospects: str  # high, medium, low
    risk_assessment: str   # low, medium, high
    key_metrics: Dict[str, float]
    analyst_recommendations: List[str]
    market_context: Dict[str, Any]


class FinancialBERTAnalyzer(nn.Module):
    """基于FinBERT的金融情感分析器"""
    
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        
        # 加载预训练FinBERT模型
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                self.bert_model = AutoModel.from_pretrained(config.model_name)
            except Exception as e:
                logging.warning(f"Failed to load {config.model_name}, using default BERT: {e}")
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self.bert_model = AutoModel.from_pretrained("bert-base-uncased")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.bert_model = AutoModel.from_pretrained(config.model_name)
            
        # 冻结BERT参数（可选）
        for param in self.bert_model.parameters():
            param.requires_grad = False
            
        # 情感分类头
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size // 2, 3)  # positive, negative, neutral
        )
        
        # 市场相关性预测头
        self.market_relevance_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1)
        )
        
        # 关键词提取头
        self.keyword_extractor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.max_sequence_length)
        )
        
        # 情感强度预测
        self.sentiment_intensity = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1)
        )
        
    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 清理文本
        text = re.sub(r'http\S+', '', text)  # 移除URL
        text = re.sub(r'@\w+', '', text)     # 移除@用户名
        text = re.sub(r'#\w+', '', text)     # 移除hashtag
        text = re.sub(r'\s+', ' ', text)     # 统一空格
        text = text.strip()
        
        # 转换金融术语
        financial_terms = {
            'bull market': 'bullish trend',
            'bear market': 'bearish trend',
            'IPO': 'initial public offering',
            'M&A': 'mergers and acquisitions',
            'P/E': 'price to earnings ratio'
        }
        
        for term, replacement in financial_terms.items():
            text = text.replace(term, replacement)
            
        return text
    
    def forward(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Args:
            texts: list of text strings
        Returns:
            Dict containing analysis results
        """
        batch_size = len(texts)
        
        # 预处理文本
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 分词
        encoded = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_tensors='pt'
        )
        
        # 移动到正确的设备
        device = next(self.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # BERT编码
        with torch.no_grad():
            bert_outputs = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # 获取[CLS]表示
        cls_output = bert_outputs.last_hidden_state[:, 0]  # [CLS] token
        sequence_output = bert_outputs.last_hidden_state
        
        # 情感分类
        sentiment_logits = self.sentiment_classifier(cls_output)
        sentiment_probs = F.softmax(sentiment_logits, dim=-1)
        
        # 市场相关性
        market_relevance = torch.sigmoid(self.market_relevance_predictor(cls_output))
        
        # 情感强度
        intensity_scores = torch.sigmoid(self.sentiment_intensity(cls_output))
        
        # 关键词重要性
        keyword_scores = self.keyword_extractor(cls_output)
        keyword_attention = F.softmax(keyword_scores, dim=-1)
        
        return {
            'sentiment_logits': sentiment_logits,
            'sentiment_probs': sentiment_probs,
            'market_relevance': market_relevance,
            'intensity_scores': intensity_scores,
            'keyword_attention': keyword_attention,
            'bert_features': cls_output,
            'sequence_features': sequence_output,
            'attention_mask': attention_mask
        }
    
    def analyze_sentiment(self, texts: List[str]) -> List[SentimentAnalysisResult]:
        """分析情感并返回结构化结果"""
        with torch.no_grad():
            outputs = self.forward(texts)
            
        results = []
        sentiment_labels = ['negative', 'neutral', 'positive']
        
        batch_size = len(texts)
        for i in range(batch_size):
            # 获取情感预测  
            if i < outputs['sentiment_probs'].shape[0]:
                sentiment_probs = outputs['sentiment_probs'][i].cpu().numpy()
            else:
                sentiment_probs = outputs['sentiment_probs'][0].cpu().numpy()
            predicted_sentiment = sentiment_labels[np.argmax(sentiment_probs)]
            confidence = np.max(sentiment_probs)
            
            # 创建分数字典
            scores = {
                label: float(prob) for label, prob in zip(sentiment_labels, sentiment_probs)
            }
            
            # 市场相关性
            market_relevance = float(outputs['market_relevance'][i].cpu().numpy())
            
            # 提取关键短语（简化实现）
            key_phrases = self._extract_key_phrases(texts[i])
            
            result = SentimentAnalysisResult(
                sentiment=predicted_sentiment,
                confidence=confidence,
                scores=scores,
                key_phrases=key_phrases,
                market_relevance=market_relevance
            )
            results.append(result)
            
        return results
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """提取关键短语（简化实现）"""
        # 金融关键词
        financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline',
            'bullish', 'bearish', 'rally', 'crash', 'volatility', 'volume',
            'dividend', 'acquisition', 'merger', 'IPO', 'buyback', 'split',
            'upgrade', 'downgrade', 'target price', 'analyst', 'forecast'
        ]
        
        text_lower = text.lower()
        found_phrases = []
        
        for keyword in financial_keywords:
            if keyword in text_lower:
                found_phrases.append(keyword)
                
        return found_phrases[:10]  # 返回前10个关键短语


class GPTFundamentalAnalyzer(nn.Module):
    """基于GPT的基本面分析器"""
    
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        
        # GPT模型设置
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                logging.warning(f"Failed to load GPT model: {e}")
                self.tokenizer = MockTokenizer()
                self.gpt_model = MockModel()
        else:
            self.tokenizer = MockTokenizer()
            self.gpt_model = MockModel()
        
        # 基本面分析分类器
        hidden_size = getattr(self.gpt_model.config, 'hidden_size', 768)
        
        self.financial_health_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 3)  # strong, moderate, weak
        )
        
        self.growth_prospects_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 3)  # high, medium, low
        )
        
        self.risk_assessment_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 3)  # low, medium, high
        )
        
    def generate_analysis_prompt(self, company: str, news_text: str, 
                               financial_data: Optional[Dict] = None) -> str:
        """生成基本面分析提示"""
        prompt = f"""
        Analyze the fundamental prospects of {company} based on the following information:
        
        Recent News: {news_text[:500]}
        
        Please evaluate:
        1. Financial Health (Strong/Moderate/Weak)
        2. Growth Prospects (High/Medium/Low) 
        3. Risk Assessment (Low/Medium/High)
        4. Key Financial Metrics
        5. Market Position and Competitive Advantages
        
        Analysis:
        """
        
        if financial_data:
            prompt += f"\nFinancial Data: {str(financial_data)[:200]}"
            
        return prompt
    
    def forward(self, texts: List[str], companies: List[str] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            texts: list of text for analysis
            companies: list of company names (optional)
        Returns:
            Dict containing fundamental analysis results
        """
        if companies is None:
            companies = ["Unknown Company"] * len(texts)
            
        batch_size = len(texts)
        
        # 生成分析提示
        prompts = [
            self.generate_analysis_prompt(company, text)
            for company, text in zip(companies, texts)
        ]
        
        # 分词
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_tensors='pt'
        )
        
        # 移动到正确设备
        device = next(self.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # GPT编码
        with torch.no_grad():
            gpt_outputs = self.gpt_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # 使用最后一层隐藏状态的平均池化
        hidden_states = gpt_outputs.hidden_states[-1]
        pooled_output = hidden_states.mean(dim=1)
        
        # 基本面分析预测
        financial_health_logits = self.financial_health_classifier(pooled_output)
        growth_prospects_logits = self.growth_prospects_classifier(pooled_output)
        risk_assessment_logits = self.risk_assessment_classifier(pooled_output)
        
        return {
            'financial_health_logits': financial_health_logits,
            'growth_prospects_logits': growth_prospects_logits,
            'risk_assessment_logits': risk_assessment_logits,
            'gpt_features': pooled_output,
            'hidden_states': hidden_states
        }
    
    def analyze_fundamentals(self, texts: List[str], 
                           companies: List[str] = None) -> List[FundamentalAnalysisResult]:
        """进行基本面分析"""
        with torch.no_grad():
            outputs = self.forward(texts, companies)
            
        results = []
        health_labels = ['weak', 'moderate', 'strong']
        growth_labels = ['low', 'medium', 'high']
        risk_labels = ['low', 'medium', 'high']
        
        for i in range(len(texts)):
            # 获取预测结果
            health_probs = F.softmax(outputs['financial_health_logits'][i], dim=0)
            growth_probs = F.softmax(outputs['growth_prospects_logits'][i], dim=0)
            risk_probs = F.softmax(outputs['risk_assessment_logits'][i], dim=0)
            
            financial_health = health_labels[torch.argmax(health_probs).item()]
            growth_prospects = growth_labels[torch.argmax(growth_probs).item()]
            risk_assessment = risk_labels[torch.argmax(risk_probs).item()]
            
            # 计算关键指标
            key_metrics = {
                'health_confidence': float(torch.max(health_probs)),
                'growth_confidence': float(torch.max(growth_probs)),
                'risk_confidence': float(torch.max(risk_probs)),
                'overall_score': float((torch.max(health_probs) + torch.max(growth_probs) + torch.max(risk_probs)) / 3)
            }
            
            # 生成分析师建议（简化实现）
            recommendations = self._generate_recommendations(
                financial_health, growth_prospects, risk_assessment
            )
            
            result = FundamentalAnalysisResult(
                financial_health=financial_health,
                growth_prospects=growth_prospects,
                risk_assessment=risk_assessment,
                key_metrics=key_metrics,
                analyst_recommendations=recommendations,
                market_context={}
            )
            results.append(result)
            
        return results
    
    def _generate_recommendations(self, health: str, growth: str, risk: str) -> List[str]:
        """生成投资建议"""
        recommendations = []
        
        # 基于三个维度生成建议
        if health == 'strong' and growth == 'high' and risk == 'low':
            recommendations.append("Strong Buy - Excellent fundamentals across all metrics")
        elif health == 'strong' and growth == 'high':
            recommendations.append("Buy - Strong fundamentals with high growth potential")
        elif health == 'strong' and risk == 'low':
            recommendations.append("Buy - Stable investment with strong financials")
        elif health == 'moderate' and growth == 'high':
            recommendations.append("Hold/Buy - Moderate health but strong growth prospects")
        elif risk == 'high':
            recommendations.append("Caution - High risk profile requires careful monitoring")
        else:
            recommendations.append("Hold - Mixed fundamentals suggest neutral position")
            
        return recommendations


class NewssentimentAnalyzer(nn.Module):
    """新闻情感分析器（整合多个模型）"""
    
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        
        # 初始化子模块
        self.bert_analyzer = FinancialBERTAnalyzer(config)
        self.gpt_analyzer = GPTFundamentalAnalyzer(config)
        
        # 融合层
        combined_size = config.hidden_size * 2  # BERT + GPT features
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1)  # 综合得分
        )
        
        # 新闻分类器
        self.news_category_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 8)  # 8个新闻类别
        )
        
        self.news_categories = [
            'earnings', 'market_analysis', 'company_news', 'economic_data',
            'regulatory', 'mergers_acquisitions', 'analyst_reports', 'other'
        ]
        
    def forward(self, texts: List[str], companies: List[str] = None) -> Dict[str, torch.Tensor]:
        """综合新闻分析"""
        # BERT情感分析
        bert_outputs = self.bert_analyzer(texts)
        bert_features = bert_outputs['bert_features']
        
        # GPT基本面分析
        gpt_outputs = self.gpt_analyzer(texts, companies)
        gpt_features = gpt_outputs['gpt_features']
        
        # 特征融合
        combined_features = torch.cat([bert_features, gpt_features], dim=-1)
        fusion_score = torch.sigmoid(self.fusion_layer(combined_features))
        
        # 新闻分类
        category_logits = self.news_category_classifier(bert_features)
        category_probs = F.softmax(category_logits, dim=-1)
        
        return {
            'bert_outputs': bert_outputs,
            'gpt_outputs': gpt_outputs,
            'fusion_score': fusion_score,
            'category_logits': category_logits,
            'category_probs': category_probs,
            'combined_features': combined_features
        }
    
    def analyze_news_batch(self, news_articles: List[NewsArticle]) -> Dict[str, Any]:
        """批量分析新闻文章"""
        texts = [f"{article.title}. {article.content}" for article in news_articles]
        companies = [article.ticker_symbols[0] if article.ticker_symbols else "Unknown" 
                    for article in news_articles]
        
        with torch.no_grad():
            outputs = self.forward(texts, companies)
            
        # 整理结果
        sentiment_results = self.bert_analyzer.analyze_sentiment(texts)
        fundamental_results = self.gpt_analyzer.analyze_fundamentals(texts, companies)
        
        batch_results = {
            'sentiment_analysis': sentiment_results,
            'fundamental_analysis': fundamental_results,
            'fusion_scores': outputs['fusion_score'].cpu().numpy().tolist(),
            'news_categories': [],
            'overall_market_sentiment': 0.0,
            'market_moving_probability': 0.0
        }
        
        # 新闻分类
        category_probs = outputs['category_probs'].cpu().numpy()
        for i, probs in enumerate(category_probs):
            predicted_category = self.news_categories[np.argmax(probs)]
            confidence = float(np.max(probs))
            batch_results['news_categories'].append({
                'category': predicted_category,
                'confidence': confidence,
                'all_probs': {cat: float(prob) for cat, prob in zip(self.news_categories, probs)}
            })
        
        # 计算整体市场情绪
        sentiment_scores = [result.scores['positive'] - result.scores['negative'] 
                          for result in sentiment_results]
        batch_results['overall_market_sentiment'] = float(np.mean(sentiment_scores))
        
        # 计算市场影响概率
        market_relevance_scores = [result.market_relevance for result in sentiment_results]
        batch_results['market_moving_probability'] = float(np.mean(market_relevance_scores))
        
        return batch_results


class MarketContextExtractor(nn.Module):
    """市场上下文提取器"""
    
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        
        # 上下文类型分类器
        self.context_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 6)  # 6种市场上下文
        )
        
        self.context_types = [
            'bull_market', 'bear_market', 'volatile_market',
            'earnings_season', 'economic_uncertainty', 'stable_growth'
        ]
        
        # 时间敏感性预测器
        self.time_sensitivity_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1)
        )
        
        # 影响范围预测器
        self.impact_scope_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 3)  # sector, market, global
        )
        
    def forward(self, text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_features: (batch_size, hidden_size)
        Returns:
            Dict containing market context analysis
        """
        # 市场上下文分类
        context_logits = self.context_classifier(text_features)
        context_probs = F.softmax(context_logits, dim=-1)
        
        # 时间敏感性
        time_sensitivity = torch.sigmoid(self.time_sensitivity_predictor(text_features))
        
        # 影响范围
        scope_logits = self.impact_scope_predictor(text_features)
        scope_probs = F.softmax(scope_logits, dim=-1)
        
        return {
            'context_logits': context_logits,
            'context_probs': context_probs,
            'time_sensitivity': time_sensitivity,
            'scope_logits': scope_logits,
            'scope_probs': scope_probs
        }
    
    def extract_market_context(self, text_features: torch.Tensor) -> Dict[str, Any]:
        """提取市场上下文信息"""
        with torch.no_grad():
            outputs = self.forward(text_features)
            
        batch_size = text_features.shape[0]
        results = []
        
        scope_labels = ['sector', 'market', 'global']
        
        for i in range(batch_size):
            # 市场上下文
            context_probs = outputs['context_probs'][i].cpu().numpy()
            predicted_context = self.context_types[np.argmax(context_probs)]
            context_confidence = float(np.max(context_probs))
            
            # 时间敏感性
            time_sensitivity = float(outputs['time_sensitivity'][i].cpu().numpy())
            
            # 影响范围
            scope_probs = outputs['scope_probs'][i].cpu().numpy()
            predicted_scope = scope_labels[np.argmax(scope_probs)]
            scope_confidence = float(np.max(scope_probs))
            
            result = {
                'market_context': predicted_context,
                'context_confidence': context_confidence,
                'time_sensitivity': time_sensitivity,
                'impact_scope': predicted_scope,
                'scope_confidence': scope_confidence,
                'context_distribution': {
                    ctx: float(prob) for ctx, prob in zip(self.context_types, context_probs)
                }
            }
            results.append(result)
            
        return results


def create_sample_news_articles() -> List[NewsArticle]:
    """创建示例新闻文章用于测试"""
    articles = [
        NewsArticle(
            title="Apple Reports Strong Q4 Earnings Beat Expectations",
            content="Apple Inc. reported quarterly earnings that exceeded analyst expectations, driven by strong iPhone sales and services revenue growth. The company's revenue increased 8% year-over-year to $94.9 billion.",
            source="Reuters",
            timestamp=datetime.now() - timedelta(hours=2),
            ticker_symbols=["AAPL"],
            category="earnings"
        ),
        NewsArticle(
            title="Federal Reserve Signals Potential Rate Cuts Amid Economic Concerns",
            content="The Federal Reserve indicated it may consider interest rate cuts in response to growing economic uncertainties and inflation concerns. Markets responded positively to the dovish tone.",
            source="Bloomberg",
            timestamp=datetime.now() - timedelta(hours=6),
            ticker_symbols=["SPY", "QQQ"],
            category="economic_data"
        ),
        NewsArticle(
            title="Tesla Stock Volatile After CEO Comments on Production",
            content="Tesla shares experienced high volatility following CEO Elon Musk's comments about production challenges at the new Gigafactory. Analysts remain divided on the company's near-term prospects.",
            source="CNBC",
            timestamp=datetime.now() - timedelta(hours=4),
            ticker_symbols=["TSLA"],
            category="company_news"
        )
    ]
    return articles


if __name__ == "__main__":
    # 测试文本分析模块
    print("=== 测试BERT/GPT文本分析模块 ===")
    
    config = TextConfig()
    
    # 测试FinBERT分析器
    print("\n--- 测试FinBERT情感分析 ---")
    bert_analyzer = FinancialBERTAnalyzer(config)
    
    test_texts = [
        "Apple reported strong quarterly earnings with revenue growth exceeding expectations.",
        "Market volatility increased due to concerns about inflation and economic uncertainty.",
        "The company announced a major acquisition that could boost future growth prospects."
    ]
    
    try:
        sentiment_results = bert_analyzer.analyze_sentiment(test_texts)
        for i, result in enumerate(sentiment_results):
            print(f"Text {i+1}: {result.sentiment} (confidence: {result.confidence:.3f})")
            print(f"  Market relevance: {result.market_relevance:.3f}")
            print(f"  Key phrases: {result.key_phrases}")
    except Exception as e:
        print(f"BERT分析测试失败: {e}")
    
    # 测试GPT基本面分析
    print("\n--- 测试GPT基本面分析 ---")
    gpt_analyzer = GPTFundamentalAnalyzer(config)
    
    companies = ["Apple", "Tesla", "Microsoft"]
    
    try:
        fundamental_results = gpt_analyzer.analyze_fundamentals(test_texts, companies)
        for i, result in enumerate(fundamental_results):
            print(f"Company {companies[i]}:")
            print(f"  Financial Health: {result.financial_health}")
            print(f"  Growth Prospects: {result.growth_prospects}")
            print(f"  Risk Assessment: {result.risk_assessment}")
            print(f"  Recommendations: {result.analyst_recommendations}")
    except Exception as e:
        print(f"GPT分析测试失败: {e}")
    
    # 测试综合新闻分析
    print("\n--- 测试综合新闻分析 ---")
    news_analyzer = NewssentimentAnalyzer(config)
    
    sample_articles = create_sample_news_articles()
    
    try:
        batch_results = news_analyzer.analyze_news_batch(sample_articles)
        print(f"整体市场情绪: {batch_results['overall_market_sentiment']:.3f}")
        print(f"市场影响概率: {batch_results['market_moving_probability']:.3f}")
        
        for i, category_result in enumerate(batch_results['news_categories']):
            print(f"文章 {i+1} 分类: {category_result['category']} "
                  f"(置信度: {category_result['confidence']:.3f})")
    except Exception as e:
        print(f"综合分析测试失败: {e}")
    
    print("\n=== 文本分析模块测试完成 ===")