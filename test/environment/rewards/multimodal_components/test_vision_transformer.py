"""
测试视觉变换器模块
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.environment.rewards.multimodal_components.vision_transformer import VisionTransformer


class TestVisionTransformer:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.vit = VisionTransformer(
            patch_size=16,
            num_patches=196,
            embed_dim=768,
            num_heads=12
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.vit.patch_size == 16
        assert self.vit.num_patches == 196
        assert self.vit.embed_dim == 768
        assert self.vit.num_heads == 12
    
    def test_patch_embedding(self):
        """测试补丁嵌入"""
        # 模拟图像数据 (batch_size=1, channels=3, height=224, width=224)
        mock_image = np.random.rand(1, 3, 224, 224)
        
        patches = self.vit.create_patches(mock_image)
        
        assert isinstance(patches, np.ndarray)
        assert patches.shape[1] == self.vit.num_patches
    
    def test_forward_pass(self):
        """测试前向传播"""
        mock_image = np.random.rand(1, 3, 224, 224)
        
        features = self.vit.forward(mock_image)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[-1] == self.vit.embed_dim