#!/usr/bin/env python
"""
VisualizationManager测试模块
"""

import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings

# 抑制matplotlib警告
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.visualization_manager import VisualizationManager


class TestVisualizationManager(unittest.TestCase):
    """VisualizationManager测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.test_dir = tempfile.mkdtemp()
        self.viz_manager = VisualizationManager(
            output_dir=self.test_dir,
            config={
                'save_formats': ['png'],
                'dpi': 100  # 降低DPI提高测试速度
            }
        )
        
        # 准备测试数据
        self._prepare_test_data()
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _prepare_test_data(self):
        """准备测试数据"""
        # 训练数据
        self.training_data = {
            'episode_rewards': np.random.normal(0.5, 0.2, 20).tolist(),
            'portfolio_values': (10000 * (1 + np.cumsum(np.random.normal(0.001, 0.05, 20)))).tolist(),
            'actions_history': np.random.uniform(-1, 1, 20).tolist(),
            'initial_balance': 10000.0
        }
        
        # 评估数据
        self.evaluation_data = {
            'episode_data': [{
                'episode': i + 1,
                'reward': np.random.normal(0.5, 0.3),
                'return': np.random.normal(2.0, 5.0),
                'steps': np.random.randint(200, 500)
            } for i in range(5)]
        }
        
        # Forex数据
        prices = 1.1000 + np.cumsum(np.random.normal(0, 0.0001, 50))
        self.forex_data = {
            'ohlc_data': pd.DataFrame({
                'Open': prices[:-1],
                'High': prices[:-1] + abs(np.random.normal(0, 0.0005, 49)),
                'Low': prices[:-1] - abs(np.random.normal(0, 0.0005, 49)), 
                'Close': prices[1:],
                'Volume': np.random.randint(1000, 10000, 49)
            }),
            'prices': prices.tolist(),
            'actions': np.random.uniform(-1, 1, 50).tolist()
        }
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.viz_manager)
        self.assertTrue(Path(self.test_dir).exists())
    
    def test_generate_training_visualizations(self):
        """测试训练可视化生成"""
        result = self.viz_manager.generate_training_visualizations(
            training_data=self.training_data,
            experiment_name="Test_Training",
            detailed=False  # 简化测试
        )
        self.assertTrue(len(result) > 0)
    
    def test_generate_evaluation_visualizations(self):
        """测试评估可视化生成"""
        result = self.viz_manager.generate_evaluation_visualizations(
            evaluation_data=self.evaluation_data,
            model_info={'algorithm': 'PPO'},
            detailed=False
        )
        self.assertTrue(len(result) > 0)
    
    def test_generate_forex_visualizations(self):
        """测试Forex可视化生成"""
        result = self.viz_manager.generate_forex_visualizations(
            forex_data=self.forex_data,
            currency_pair="EURUSD",
            detailed=False
        )
        self.assertTrue(len(result) > 0)
    
    def test_get_session_summary(self):
        """测试会话摘要"""
        # 先生成一些可视化
        self.viz_manager.generate_training_visualizations(
            self.training_data, "Test", detailed=False
        )
        
        summary = self.viz_manager.get_session_summary()
        self.assertIn('session_id', summary)
        self.assertIn('total_charts_generated', summary)
        self.assertTrue(summary['total_charts_generated'] > 0)


if __name__ == '__main__':
    unittest.main()