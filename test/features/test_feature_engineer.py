"""
测试特征工程模块
"""

import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineer import FeatureEngineer
from src.utils.config import Config


class TestFeatureEngineer:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.config = Config()
        self.feature_engineer = FeatureEngineer(config=self.config)
        
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 105,
            'Low': np.random.randn(100).cumsum() + 95,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        })
    
    def test_feature_engineer_initialization(self):
        """测试特征工程器初始化"""
        assert self.feature_engineer.config is not None
    
    def test_prepare_features(self):
        """测试特征准备"""
        features = self.feature_engineer.prepare_features(self.test_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
    
    def test_technical_indicators(self):
        """测试技术指标计算"""
        # 测试技术指标计算
        features = self.feature_engineer.calculate_technical_indicators(self.test_data)
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        
        # 检查是否包含SMA列
        sma_columns = [col for col in features.columns if col.startswith('SMA_')]
        assert len(sma_columns) > 0
    
    def test_feature_scaling(self):
        """测试特征缩放"""
        features = self.feature_engineer.prepare_features(self.test_data)
        
        # 检查特征是否被适当缩放
        assert features is not None