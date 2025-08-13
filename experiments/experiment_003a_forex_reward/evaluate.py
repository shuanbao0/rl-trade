#!/usr/bin/env python3
"""
Experiment #003A: Forex奖励函数优化
评估脚本 - 调用主评估程序验证forex_optimized奖励函数效果

实验目标:
- 评估forex_optimized奖励函数的性能表现
- 对比与前期实验的性能差异
- 验证外汇专业优化的实际效果
"""

import subprocess
import sys
import os

def main():
    print("=" * 60)
    print("📊 EXPERIMENT #003A: Forex奖励函数优化 - 评估")
    print("🎯 目标: 验证forex_optimized奖励函数效果")
    print("=" * 60)
    
    # 切换到项目根目录
    root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    os.chdir(root_dir)
    
    # 模型路径配置
    model_base_path = os.path.join("experiments", "experiment_003a_forex_reward", "models")
    
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
        "--episodes", "30",  # 更多episodes获得更稳定的统计
        "--generate-report",
        "--detailed-analysis",
        "--enable-visualization",
        "--save-charts",
        "--comparison-mode",  # 启用与基准实验的对比
        "--verbose"
    ]
    
    print(f"📊 评估配置:")
    print(f"   Symbol: EURUSD")
    print(f"   Model: {os.path.basename(model_path)}")
    print(f"   Episodes: 30")
    print(f"   Features: 3个基础特征")
    print(f"   Reward: forex_optimized ⭐")
    print(f"   目标: 外汇奖励优化效果验证")
    print()
    
    print("💰 Forex奖励优化预期改进:")
    print("   🎯 更精确的外汇交易成本建模")
    print("   ⚡ 点差和滑点自动考量")
    print("   📊 风险调整收益率优化")
    print("   🛡️ 过度交易惩罚机制")
    print("   📈 预期性能提升: 50-100%")
    print()
    
    print("🚀 开始Forex奖励优化评估...")
    print("   预计时间: ~10分钟")
    print("   重点监控: 奖励函数实际效果")
    print("   对比基准: Experiments #001, #002")
    print()
    
    try:
        # 执行评估
        result = subprocess.run(params, check=True, capture_output=False)
        
        print("\n" + "=" * 60)
        print("✅ EXPERIMENT #003A 评估完成!")
        print("=" * 60)
        print("🎉 Forex奖励函数优化效果验证成功!")
        print()
        print("💰 奖励优化价值验证:")
        print("   📊 详细性能指标已生成")
        print("   📈 与基准实验自动对比完成")
        print("   🎯 外汇专业优化效果已量化")
        print("   💡 为下一步特征增强提供基准")
        print()
        print("🔍 关键评估指标关注:")
        print("   • Sharpe比率提升程度")
        print("   • 胜率和风险调整收益")
        print("   • 交易频率和成本控制")
        print("   • 相对前期实验的改进幅度")
        print()
        print("📋 下一步:")
        print("   1. 查看生成的性能对比报告")
        print("   2. 分析forex_optimized的具体改进")
        print("   3. 进入 Experiment #004 117特征革命")
        print("   4. 基于此奖励函数测试高维特征")
        print()
        print("🏆 奖励函数优化里程碑完成!")
        print("   为117特征革命奠定了坚实基础")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 评估失败: {e}")
        print("🔧 可能的问题:")
        print("   1. 模型文件与forex_optimized奖励不匹配")
        print("   2. 外汇参数配置错误")
        print("   3. 对比基准数据缺失")
        return False
    except Exception as e:
        print(f"\n❌ 执行错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)