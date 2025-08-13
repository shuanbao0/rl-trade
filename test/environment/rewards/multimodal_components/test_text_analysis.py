"""
测试文本分析模块
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.environment.rewards.multimodal_components.text_analysis import FinancialBERTAnalyzer


class TestFinancialBERTAnalyzer:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.text_analyzer = FinancialBERTAnalyzer(
            model_name='bert-base',
            max_length=512,
            sentiment_threshold=0.5
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.text_analyzer.model_name == 'bert-base'
        assert self.text_analyzer.max_length == 512
        assert self.text_analyzer.sentiment_threshold == 0.5
    
    def test_sentiment_analysis(self):
        """测试情感分析"""
        positive_text = "The company reported strong quarterly earnings."
        negative_text = "Stock prices are falling due to market concerns."
        
        positive_score = self.text_analyzer.analyze_sentiment(positive_text)
        negative_score = self.text_analyzer.analyze_sentiment(negative_text)
        
        assert isinstance(positive_score, (int, float))
        assert isinstance(negative_score, (int, float))
        assert -1 <= positive_score <= 1
        assert -1 <= negative_score <= 1
    
    def test_feature_extraction(self):
        """测试特征提取"""
        sample_text = "Market volatility increased after the Federal Reserve announcement."
        
        features = self.text_analyzer.extract_features(sample_text)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0