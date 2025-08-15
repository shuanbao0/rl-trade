"""
测试跨模态融合模块
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.environment.rewards.multimodal_components.cross_modal_fusion import MultimodalFusionTransformer


class TestMultimodalFusionTransformer:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.fusion = MultimodalFusionTransformer(
            fusion_method='attention',
            num_modalities=3,
            feature_dim=128
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.fusion.fusion_method == 'attention'
        assert self.fusion.num_modalities == 3
        assert self.fusion.feature_dim == 128
    
    def test_attention_fusion(self):
        """测试注意力融合"""
        modality_features = [
            np.random.rand(128),  # Price features
            np.random.rand(128),  # News features  
            np.random.rand(128)   # Volume features
        ]
        
        fused_features = self.fusion.fuse_features(modality_features)
        
        assert isinstance(fused_features, np.ndarray)
        assert fused_features.shape[0] == 128
    
    def test_concatenation_fusion(self):
        """测试连接融合"""
        concat_fusion = CrossModalFusion(fusion_method='concat')
        
        modality_features = [
            np.random.rand(64),
            np.random.rand(64)
        ]
        
        fused_features = concat_fusion.fuse_features(modality_features)
        
        assert isinstance(fused_features, np.ndarray)
        assert fused_features.shape[0] == 128  # 64 + 64