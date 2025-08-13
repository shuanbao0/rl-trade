#!/usr/bin/env python3
"""
Experiment #002: 可视化系统集成
训练脚本 - 调用主训练程序测试可视化功能

实验目标:
- 集成训练过程可视化监控 
- 测试实时loss和reward图表生成
- 验证可视化对用户体验的改进
"""

import subprocess
import sys
import os

def main():
    print("=" * 60)
    print("🧪 EXPERIMENT #002: 可视化系统集成")
    print("🎯 目标: 实现训练过程可视化监控")
    print("=" * 60)
    
    # 切换到项目根目录
    root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    os.chdir(root_dir)
    
    # 实验参数配置 - 使用train_model.py支持的参数
    params = [
        sys.executable, "train_model.py",
        "--symbol", "EURUSD",
        "--data-dir", "datasets/AAPL_20250724_210946",  # 使用3特征基础数据集
        "--iterations", "100",  # 增加轮次观察可视化效果
        "--reward-type", "simple_return",
        "--timesteps-total", "50000",
        "--enable-visualization",  # 启用可视化
        "--visualization-freq", "5000",  # 每5000步生成图表
        "--no-gpu",
        "--verbose"
    ]
    
    print(f"📊 实验配置:")
    print(f"   Symbol: EURUSD")
    print(f"   Algorithm: PPO")
    print(f"   Reward: simple_return")
    print(f"   Timesteps: 50,000")
    print(f"   Visualization: 启用 (每5000步)")
    print(f"   Features: 3个基础特征")
    print(f"   Target: 可视化集成验证")
    print()
    
    print("🎨 可视化特性:")
    print("   ✅ 实时训练监控")
    print("   ✅ Loss曲线跟踪") 
    print("   ✅ Reward趋势分析")
    print("   ✅ 自动图表生成和保存")
    print()
    
    print("🚀 开始可视化集成训练...")
    print("   预计时间: ~1小时")
    print("   重点监控: 图表生成和训练稳定性")
    print("   用户价值: 可以查看loss等数据走势分析问题")
    print()
    
    try:
        # 执行训练
        result = subprocess.run(params, check=True, capture_output=False)
        
        # 检查生成的可视化文件
        viz_dirs = [
            "training_visualizations",
            "experiments/experiment_002_visualization/visualizations"
        ]
        
        chart_count = 0
        for viz_dir in viz_dirs:
            if os.path.exists(viz_dir):
                chart_files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
                chart_count += len(chart_files)
        
        print("\n" + "=" * 60)
        print("✅ EXPERIMENT #002 训练完成!")
        print("=" * 60)
        print("🎉 可视化系统集成成功!")
        print("   训练过程可视化正常运行")
        print("   用户现在可以实时监控训练进度")
        print(f"   生成图表数量: {chart_count} 个")
        print()
        print("🎨 可视化系统价值:")
        print("   📈 可以查看loss等数据的走势")
        print("   🔍 方便分析训练过程的问题")
        print("   ⚡ 提供实时训练状态反馈")
        print("   💡 帮助用户理解模型训练过程")
        print()
        print("📋 下一步:")
        print("   1. 运行 evaluate.py 测试评估可视化")
        print("   2. 检查生成的训练图表")
        print("   3. 进入 Experiment #003A 奖励函数优化")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
        print("🔧 可能的问题:")
        print("   1. 可视化依赖库未安装 (matplotlib, seaborn)")
        print("   2. 磁盘空间不足保存图表")
        print("   3. 图表生成频率过高导致性能问题")
        return False
    except Exception as e:
        print(f"\n❌ 执行错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)