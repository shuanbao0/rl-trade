#!/usr/bin/env python
"""
BaseVisualizer测试模块
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import warnings

# 抑制matplotlib警告
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.base_visualizer import BaseVisualizer


class TestBaseVisualizer(unittest.TestCase):
    """BaseVisualizer测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.test_dir = tempfile.mkdtemp()
        self.visualizer = BaseVisualizer(output_dir=self.test_dir)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.visualizer)
        self.assertEqual(str(self.visualizer.output_dir), self.test_dir)
    
    def test_create_figure(self):
        """测试创建图形"""
        fig, axes = self.visualizer.create_figure(figsize=(10, 8), subplots=(2, 2))
        self.assertIsNotNone(fig)
        self.assertEqual(len(axes), 4)  # 2x2 = 4 axes
    
    def test_color_scheme(self):
        """测试颜色方案"""
        colors = self.visualizer.colors
        self.assertIn('primary', colors)
        self.assertIn('profit', colors)
        self.assertIn('loss', colors)
    
    def test_save_figure(self):
        """测试保存图形"""
        fig, _ = self.visualizer.create_figure()
        files = self.visualizer.save_figure(fig, 'test_chart', 'test_category')
        self.assertTrue(len(files) > 0)
        
        # 验证文件存在
        for file_path in files:
            self.assertTrue(Path(file_path).exists())


if __name__ == '__main__':
    unittest.main()