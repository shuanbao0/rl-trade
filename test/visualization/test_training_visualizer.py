#!/usr/bin/env python
"""
TrainingVisualizer测试模块
"""

import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path
import sys
import warnings

# 抑制matplotlib警告
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径  
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.training_visualizer import TrainingVisualizer


class TestTrainingVisualizer(unittest.TestCase):
    """TrainingVisualizer测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.test_dir = tempfile.mkdtemp()
        self.visualizer = TrainingVisualizer(output_dir=self.test_dir)
        
        # 创建测试数据
        self.episode_rewards = np.random.normal(0.5, 0.2, 50).tolist()
        self.portfolio_values = (10000 * (1 + np.cumsum(np.random.normal(0.001, 0.05, 50)))).tolist()
        self.actions_history = np.random.uniform(-1, 1, 50).tolist()
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.visualizer)
    
    def test_plot_reward_curves(self):
        """测试奖励曲线绘制"""
        files = self.visualizer.plot_reward_curves(self.episode_rewards)
        self.assertTrue(len(files) > 0)
        
        # 验证文件存在
        for file_path in files:
            self.assertTrue(Path(file_path).exists())
    
    def test_plot_portfolio_evolution(self):
        """测试投资组合演化图"""
        files = self.visualizer.plot_portfolio_evolution(
            self.portfolio_values, initial_balance=10000
        )
        self.assertTrue(len(files) > 0)
    
    def test_plot_action_analysis(self):
        """测试交易动作分析"""
        files = self.visualizer.plot_action_analysis(
            self.actions_history, self.portfolio_values
        )
        self.assertTrue(len(files) > 0)
    
    def test_create_training_dashboard(self):
        """测试训练仪表板创建"""
        files = self.visualizer.create_training_dashboard(
            episode_rewards=self.episode_rewards,
            portfolio_values=self.portfolio_values,
            actions_history=self.actions_history,
            initial_balance=10000.0
        )
        self.assertTrue(len(files) > 0)


if __name__ == '__main__':
    unittest.main()