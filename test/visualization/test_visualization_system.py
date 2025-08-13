#!/usr/bin/env python
"""
测试完整的可视化系统

测试所有可视化组件的功能，包括：
1. TrainingVisualizer - 训练进度图表
2. EvaluationVisualizer - 评估结果分析
3. ForexVisualizer - 外汇专用图表
4. ComparisonVisualizer - 模型对比
5. VisualizationManager - 统一管理
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_visualization_system():
    """测试完整的可视化系统"""
    print("=" * 60)
    print("TensorTrade 可视化系统测试")
    print("=" * 60)
    
    try:
        # 导入可视化模块
        print("1. 测试模块导入...")
        from src.visualization import (
            BaseVisualizer, TrainingVisualizer, EvaluationVisualizer, 
            ForexVisualizer, ComparisonVisualizer, VisualizationManager
        )
        print("   ✅ 所有可视化模块导入成功!")
        
        # 创建测试数据
        print("\n2. 生成测试数据...")
        
        # 训练数据
        training_data = {
            'episode_rewards': np.random.normal(0.5, 0.2, 100).tolist(),
            'portfolio_values': (10000 * (1 + np.cumsum(np.random.normal(0.001, 0.05, 100)))).tolist(),
            'actions_history': np.random.uniform(-1, 1, 100).tolist(),
            'initial_balance': 10000.0
        }
        
        # 评估数据
        episode_data = []
        for i in range(20):
            episode_data.append({
                'episode': i + 1,
                'reward': np.random.normal(0.5, 0.3),
                'return': np.random.normal(2.0, 5.0),
                'steps': np.random.randint(200, 500),
                'final_value': 10000 + np.random.normal(200, 1000),
                'max_drawdown': abs(np.random.normal(0, 5)),
                'volatility': abs(np.random.normal(0.05, 0.02))
            })
        
        evaluation_data = {'episode_data': episode_data}
        
        # Forex数据
        prices = 1.1000 + np.cumsum(np.random.normal(0, 0.0001, 500))
        forex_data = {
            'ohlc_data': pd.DataFrame({
                'Open': prices[:-1],
                'High': prices[:-1] + abs(np.random.normal(0, 0.0005, 499)),
                'Low': prices[:-1] - abs(np.random.normal(0, 0.0005, 499)),
                'Close': prices[1:],
                'Volume': np.random.randint(1000, 10000, 499)
            }),
            'prices': prices.tolist(),
            'actions': np.random.uniform(-1, 1, 500).tolist(),
            'pip_size': 0.0001
        }
        
        # 模型对比数据
        models_data = {
            'PPO_Model': {
                'returns': np.random.normal(1.5, 3.0, 50).tolist(),
                'portfolio_values': (10000 * (1 + np.cumsum(np.random.normal(0.0015, 0.03, 50)))).tolist(),
                'mean_return': 1.5,
                'std_return': 3.0,
                'sharpe_ratio': 0.5,
                'max_drawdown': 8.5,
                'win_rate': 0.62
            },
            'SAC_Model': {
                'returns': np.random.normal(1.2, 2.5, 50).tolist(),
                'portfolio_values': (10000 * (1 + np.cumsum(np.random.normal(0.0012, 0.025, 50)))).tolist(),
                'mean_return': 1.2,
                'std_return': 2.5,
                'sharpe_ratio': 0.48,
                'max_drawdown': 7.2,
                'win_rate': 0.58
            }
        }
        
        print("   ✅ 测试数据生成完成!")
        
        # 测试可视化管理器
        print("\n3. 测试可视化管理器...")
        
        viz_manager = VisualizationManager(
            output_dir="test_visualizations",
            config={
                'save_formats': ['png'],
                'dpi': 150,  # 降低DPI加快测试速度
                'figure_size': [10, 6]
            }
        )
        
        print("   ✅ 可视化管理器创建成功!")
        
        # 测试训练可视化
        print("\n4. 测试训练可视化...")
        training_viz_files = viz_manager.generate_training_visualizations(
            training_data=training_data,
            experiment_name="Test_Experiment",
            detailed=True
        )
        print(f"   ✅ 生成了 {len(training_viz_files)} 类训练图表!")
        
        # 测试评估可视化
        print("\n5. 测试评估可视化...")
        evaluation_viz_files = viz_manager.generate_evaluation_visualizations(
            evaluation_data=evaluation_data,
            model_info={'algorithm': 'PPO', 'reward_type': 'test'},
            detailed=True
        )
        print(f"   ✅ 生成了 {len(evaluation_viz_files)} 类评估图表!")
        
        # 测试Forex可视化
        print("\n6. 测试Forex可视化...")
        forex_viz_files = viz_manager.generate_forex_visualizations(
            forex_data=forex_data,
            currency_pair="EURUSD",
            detailed=True
        )
        print(f"   ✅ 生成了 {len(forex_viz_files)} 类Forex图表!")
        
        # 测试模型对比可视化
        print("\n7. 测试模型对比可视化...")
        comparison_viz_files = viz_manager.generate_comparison_visualizations(
            models_data=models_data,
            comparison_type='all',
            detailed=True
        )
        print(f"   ✅ 生成了 {len(comparison_viz_files)} 类对比图表!")
        
        # 测试综合报告
        print("\n8. 测试综合报告...")
        report_data = {
            'training_data': training_data,
            'evaluation_data': evaluation_data,
            'forex_data': forex_data,
            'comparison_data': models_data,
            'experiment_name': 'Complete_Test',
            'model_info': {
                'algorithm': 'PPO',
                'reward_type': 'forex_optimized',
                'symbol': 'EURUSD'
            }
        }
        
        comprehensive_reports = viz_manager.create_comprehensive_report(
            report_data=report_data,
            report_name="Complete_Visualization_Test",
            include_html=True
        )
        print(f"   ✅ 生成了 {len(comprehensive_reports)} 个综合报告!")
        
        # 获取会话摘要
        session_summary = viz_manager.get_session_summary()
        
        print("\n" + "=" * 60)
        print("测试完成 - 可视化系统测试结果")
        print("=" * 60)
        print(f"✅ 会话ID: {session_summary['session_id']}")
        print(f"✅ 总共生成图表: {session_summary['total_charts_generated']} 个")
        print(f"✅ 输出目录: {session_summary['output_directory']}")
        print(f"✅ 训练图表: {len(training_viz_files)} 类")
        print(f"✅ 评估图表: {len(evaluation_viz_files)} 类")
        print(f"✅ Forex图表: {len(forex_viz_files)} 类") 
        print(f"✅ 对比图表: {len(comparison_viz_files)} 类")
        print(f"✅ 综合报告: {len(comprehensive_reports)} 个")
        
        print("\n🎉 所有可视化组件测试成功!")
        print("📊 可视化系统已准备就绪，可用于实际训练和评估!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 可视化系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_visualization_system()
    sys.exit(0 if success else 1)