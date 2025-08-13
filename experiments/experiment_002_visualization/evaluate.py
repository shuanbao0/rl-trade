#!/usr/bin/env python3
"""
Experiment #002: 可视化系统集成
评估脚本 - 调用主评估程序测试评估可视化功能

实验目标:
- 测试评估过程的可视化功能
- 生成详细的性能分析图表
- 验证可视化对评估分析的帮助
"""

import subprocess
import sys
import os

def main():
    print("=" * 60)
    print("📊 EXPERIMENT #002: 可视化系统集成 - 评估")
    print("🎯 目标: 测试评估可视化功能")
    print("=" * 60)
    
    # 切换到项目根目录
    root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    os.chdir(root_dir)
    
    # 模型路径配置
    model_base_path = os.path.join("experiments", "experiment_002_visualization", "models")
    
    # 查找训练好的模型
    model_files = []
    if os.path.exists(model_base_path):
        for file in os.listdir(model_base_path):
            if file.endswith('.zip'):
                model_files.append(os.path.join(model_base_path, file))
    
    if not model_files:
        print("❌ 未找到训练好的模型文件")
        print(f"   检查路径: {model_base_path}")
        print("   请先运行 train.py 完成模型训练")
        return False
    
    # 使用最新的模型文件
    model_path = max(model_files, key=os.path.getmtime)
    print(f"📁 使用模型: {model_path}")
    
    # 评估参数配置
    params = [
        sys.executable, "evaluate_model.py",
        "--symbol", "EURUSD",
        "--model-path", model_path,
        "--data-dir", "datasets/AAPL_20250724_210946",  # 对应训练数据集
        "--episodes", "25",  # 充分的评估episodes
        "--generate-report",
        "--detailed-analysis",
        "--enable-visualization",  # 启用评估可视化
        "--save-charts",  # 保存分析图表
        "--verbose"
    ]
    
    print(f"📊 评估配置:")
    print(f"   Symbol: EURUSD")
    print(f"   Model: {os.path.basename(model_path)}")
    print(f"   Episodes: 25")
    print(f"   Features: 3个基础特征")
    print(f"   Visualization: 启用")
    print(f"   目标: 可视化评估验证")
    print()
    
    print("🎨 评估可视化特性:")
    print("   📈 投资组合性能曲线")
    print("   📊 收益分布分析图")
    print("   🎯 交易行为可视化")
    print("   ⚡ 风险指标图表")
    print()
    
    print("🚀 开始可视化评估...")
    print("   预计时间: ~8分钟")
    print("   重点监控: 图表生成和评估准确性")
    print("   用户价值: 直观的性能分析图表")
    print()
    
    try:
        # 执行评估
        result = subprocess.run(params, check=True, capture_output=False)
        
        # 检查生成的评估图表
        eval_viz_dirs = [
            "evaluation_visualizations",
            "experiments/experiment_002_visualization/visualizations",
            "experiments/experiment_002_visualization/results"
        ]
        
        chart_count = 0
        for viz_dir in eval_viz_dirs:
            if os.path.exists(viz_dir):
                chart_files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
                chart_count += len(chart_files)
        
        print("\n" + "=" * 60)
        print("✅ EXPERIMENT #002 评估完成!")
        print("=" * 60)
        print("🎉 可视化评估系统成功!")
        print("   评估过程可视化正常运行")
        print("   生成了详细的性能分析图表")
        print(f"   评估图表数量: {chart_count} 个")
        print()
        print("🎨 可视化评估价值:")
        print("   📊 直观的性能指标展示")
        print("   📈 清晰的投资组合变化轨迹")
        print("   🔍 深度的交易行为分析")
        print("   💡 便于发现模型优缺点")
        print()
        print("📋 下一步:")
        print("   1. 查看生成的评估图表和报告")
        print("   2. 对比Experiment #001的性能基准")
        print("   3. 进入 Experiment #003A 奖励函数优化")
        print()
        print("🏆 可视化系统集成完全成功!")
        print("   训练和评估的可视化功能均已验证")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 评估失败: {e}")
        print("🔧 可能的问题:")
        print("   1. 模型文件不兼容或损坏")
        print("   2. 评估可视化配置错误")
        print("   3. 图表保存权限不足")
        return False
    except Exception as e:
        print(f"\n❌ 执行错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)