#!/usr/bin/env python
"""
EvaluationVisualizer测试模块
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

from src.visualization.evaluation_visualizer import EvaluationVisualizer


class TestEvaluationVisualizer(unittest.TestCase):
    """EvaluationVisualizer测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.test_dir = tempfile.mkdtemp()
        self.visualizer = EvaluationVisualizer(output_dir=self.test_dir)
        
        # 创建测试episode数据
        self.episode_data = []
        for i in range(10):
            self.episode_data.append({
                'episode': i + 1,
                'reward': np.random.normal(0.5, 0.3),
                'return': np.random.normal(2.0, 5.0),
                'steps': np.random.randint(200, 500),
                'final_value': 10000 + np.random.normal(200, 1000),
                'max_drawdown': abs(np.random.normal(0, 5)),
                'volatility': abs(np.random.normal(0.05, 0.02))
            })
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.visualizer)
    
    def test_plot_episode_performance(self):
        """测试episode性能图表"""
        files = self.visualizer.plot_episode_performance(self.episode_data)
        self.assertTrue(len(files) > 0)
        
        # 验证文件存在
        for file_path in files:
            self.assertTrue(Path(file_path).exists())
    
    def test_plot_trading_signals(self):
        """测试交易信号图表"""
        prices = [1.1 + i * 0.001 for i in range(100)]
        actions = np.random.uniform(-1, 1, 100).tolist()
        portfolio_values = (10000 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 100)))).tolist()
        
        files = self.visualizer.plot_trading_signals(prices, actions, portfolio_values)
        self.assertTrue(len(files) > 0)
    
    def test_plot_risk_return_analysis(self):
        """测试风险收益分析"""
        files = self.visualizer.plot_risk_return_analysis(self.episode_data)
        self.assertTrue(len(files) > 0)
    
    def test_create_evaluation_report(self):
        """测试创建评估报告"""
        model_info = {
            'algorithm': 'PPO',
            'reward_type': 'test_reward'
        }
        
        files = self.visualizer.create_evaluation_report(
            self.episode_data, model_info, "Test Period"
        )
        self.assertTrue(len(files) > 0)


if __name__ == '__main__':
    unittest.main()