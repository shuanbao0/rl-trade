"""
测试前向验证器模块  
"""

import pytest
import pandas as pd
import numpy as np
from src.validation.walk_forward_validator import WalkForwardValidator
from src.utils.config import Config


class TestWalkForwardValidator:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.config = Config()
        
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
        self.test_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.randn(365).cumsum() + 100,
            'High': np.random.randn(365).cumsum() + 105,
            'Low': np.random.randn(365).cumsum() + 95, 
            'Close': np.random.randn(365).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 365)
        })
        
        self.validator = WalkForwardValidator(
            config=self.config,
            data=self.test_data
        )
    
    def test_validator_initialization(self):
        """测试验证器初始化"""
        assert self.validator.config is not None
        assert self.validator.data is not None
    
    def test_data_split_generation(self):
        """测试数据分割生成"""
        splits = self.validator.generate_splits(
            train_size=180,  # 6个月训练
            test_size=30,    # 1个月测试
            step_size=30     # 1个月步长
        )
        
        assert len(splits) > 0
        
        # 验证每个分割的结构
        for split in splits[:2]:  # 只检查前2个
            assert 'train_start' in split
            assert 'train_end' in split
            assert 'test_start' in split
            assert 'test_end' in split
    
    def test_cross_validation(self):
        """测试交叉验证"""
        # 模拟一个简单的模型函数
        def mock_model_train(train_data):
            return {'model': 'trained'}
        
        def mock_model_evaluate(model, test_data):
            return {'score': np.random.rand()}
        
        results = self.validator.validate(
            model_train_func=mock_model_train,
            model_evaluate_func=mock_model_evaluate,
            n_splits=3
        )
        
        assert isinstance(results, list)
        assert len(results) == 3
        
        # 验证结果结构
        for result in results:
            assert 'score' in result
    
    def test_performance_metrics(self):
        """测试性能指标计算"""
        # 模拟验证结果
        mock_results = [
            {'score': 0.8, 'returns': [0.1, 0.05, -0.02]},
            {'score': 0.75, 'returns': [0.08, 0.03, 0.01]},
            {'score': 0.85, 'returns': [0.12, -0.01, 0.04]}
        ]
        
        metrics = self.validator.calculate_metrics(mock_results)
        
        assert isinstance(metrics, dict)
        assert 'mean_score' in metrics
        assert 'std_score' in metrics