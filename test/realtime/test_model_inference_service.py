"""
测试模型推理服务模块
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.realtime.model_inference_service import ModelInferenceService
from src.utils.config import Config


class TestModelInferenceService:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.config = Config()
        self.model_path = "test_model.pkl"
        
    def test_service_initialization(self):
        """测试服务初始化"""
        service = ModelInferenceService(
            model_path=self.model_path,
            config=self.config
        )
        
        assert service.model_path == self.model_path
        assert service.config is not None
    
    @patch('pickle.load')
    @patch('builtins.open')
    def test_model_loading(self, mock_open, mock_pickle_load):
        """测试模型加载"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5])
        mock_pickle_load.return_value = mock_model
        
        service = ModelInferenceService(
            model_path=self.model_path,
            config=self.config
        )
        
        service.load_model()
        
        assert service.model is not None
    
    def test_prediction(self):
        """测试预测功能"""
        service = ModelInferenceService(
            model_path=self.model_path,
            config=self.config
        )
        
        # 模拟模型
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.3, 0.8]])
        service.model = mock_model
        
        # 测试数据
        features = np.random.randn(10)
        
        result = service.predict(features, symbol="AAPL")
        
        assert result is not None
        assert hasattr(result, 'action')
        assert hasattr(result, 'confidence')
    
    @pytest.mark.asyncio
    async def test_async_prediction(self):
        """测试异步预测"""
        service = ModelInferenceService(
            model_path=self.model_path,
            config=self.config
        )
        
        # 模拟模型
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.2, 0.9]])
        service.model = mock_model
        
        features = np.random.randn(10)
        
        result = await service.predict_async(features, symbol="AAPL")
        
        assert result is not None
        assert hasattr(result, 'action')
        assert hasattr(result, 'confidence')
    
    def test_batch_prediction(self):
        """测试批量预测"""
        service = ModelInferenceService(
            model_path=self.model_path,
            config=self.config,
            batch_size=2
        )
        
        # 模拟模型
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.1, 0.7], [0.4, 0.6]])
        service.model = mock_model
        
        batch_features = [np.random.randn(10), np.random.randn(10)]
        symbols = ["AAPL", "MSFT"]
        
        results = service.predict_batch(batch_features, symbols)
        
        assert len(results) == 2
        assert all(hasattr(r, 'action') for r in results)
        assert all(hasattr(r, 'confidence') for r in results)