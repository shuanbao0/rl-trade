"""
测试多模态配置模块
"""

import pytest
from src.environment.rewards.multimodal_components.multimodal_config import MultimodalConfig


class TestMultimodalConfig:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.config = MultimodalConfig(
            modalities=['price', 'news', 'volume'],
            fusion_method='attention',
            feature_dims={'price': 128, 'news': 256, 'volume': 64}
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.config.modalities == ['price', 'news', 'volume']
        assert self.config.fusion_method == 'attention'
        assert self.config.feature_dims == {'price': 128, 'news': 256, 'volume': 64}
    
    def test_config_validation(self):
        """测试配置验证"""
        assert self.config.validate_config() == True
        
        # 测试无效配置
        invalid_config = MultimodalConfig(
            modalities=[],
            fusion_method='invalid_method'
        )
        assert invalid_config.validate_config() == False
    
    def test_get_total_feature_dim(self):
        """测试获取总特征维度"""
        total_dim = self.config.get_total_feature_dim()
        expected_total = 128 + 256 + 64  # 448
        
        assert total_dim == expected_total