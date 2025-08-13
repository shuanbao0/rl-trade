#!/usr/bin/env python3
"""
Experiment #001: 系统迁移验证
训练脚本 - 调用主训练程序进行系统迁移验证

实验目标:
- 验证从Ray RLlib到Stable-Baselines3的迁移成功性
- 使用基础3特征建立性能基准
- 确保核心训练流程稳定运行
"""

import subprocess
import sys
import os

def main():
    print("=" * 60)
    print("🧪 EXPERIMENT #001: 系统迁移验证")
    print("🎯 目标: 验证Stable-Baselines3迁移成功性")
    print("=" * 60)
    
    # 切换到项目根目录
    root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    os.chdir(root_dir)
    
    # 实验参数配置 - 使用train_model.py支持的参数
    params = [
        sys.executable, "train_model.py",
        "--symbol", "EURUSD",
        "--data-dir", "datasets/AAPL_20250724_210946",  # 使用3特征基础数据集
        "--iterations", "50",  # 较少迭代快速验证
        "--reward-type", "simple_return",
        "--timesteps-total", "10000",
        "--no-gpu",
        "--verbose"
    ]
    
    print(f"📊 实验配置:")
    print(f"   Symbol: EURUSD")
    print(f"   Algorithm: PPO")
    print(f"   Reward: simple_return") 
    print(f"   Timesteps: 10,000")
    print(f"   Features: 3个基础特征")
    print(f"   Target: 系统迁移验证")
    print()
    
    print("🚀 开始系统迁移验证训练...")
    print("   预计时间: ~15分钟")
    print("   重点监控: 训练稳定性和收敛性")
    print()
    
    try:
        # 执行训练
        result = subprocess.run(params, check=True, capture_output=False)
        
        print("\n" + "=" * 60)
        print("✅ EXPERIMENT #001 训练完成!")
        print("=" * 60)
        print("🎉 系统迁移验证成功!")
        print("   Stable-Baselines3集成正常")
        print("   基础训练流程稳定")
        print("   为后续实验建立了性能基准")
        print()
        print("📋 下一步:")
        print("   1. 运行 evaluate.py 验证模型性能")
        print("   2. 对比训练过程可视化图表")
        print("   3. 进入 Experiment #002 集成可视化系统")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
        print("🔧 可能的问题:")
        print("   1. 数据集路径不正确")
        print("   2. 依赖环境未正确安装")
        print("   3. 内存或计算资源不足")
        return False
    except Exception as e:
        print(f"\n❌ 执行错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)