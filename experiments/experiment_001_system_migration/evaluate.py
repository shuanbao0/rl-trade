#!/usr/bin/env python3
"""
Experiment #001: 系统迁移验证
评估脚本 - 调用主评估程序验证模型性能

实验目标:
- 评估迁移验证模型的性能表现
- 建立系统性能基准线
- 确保评估流程正常运行
"""

import subprocess
import sys
import os

def main():
    print("=" * 60)
    print("📊 EXPERIMENT #001: 系统迁移验证 - 评估")
    print("🎯 目标: 建立系统性能基准线")
    print("=" * 60)
    
    # 切换到项目根目录
    root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    os.chdir(root_dir)
    
    # 模型路径配置
    model_base_path = os.path.join("experiments", "experiment_001_system_migration", "models")
    
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
        "--episodes", "20",  # 充分的评估episodes
        "--generate-report",
        "--detailed-analysis",
        "--verbose"
    ]
    
    print(f"📊 评估配置:")
    print(f"   Symbol: EURUSD")
    print(f"   Model: {os.path.basename(model_path)}")
    print(f"   Episodes: 20")
    print(f"   Features: 3个基础特征")
    print(f"   目标: 性能基准建立")
    print()
    
    print("🚀 开始系统迁移验证评估...")
    print("   预计时间: ~5分钟")
    print("   重点监控: 基准性能指标")
    print()
    
    try:
        # 执行评估
        result = subprocess.run(params, check=True, capture_output=False)
        
        print("\n" + "=" * 60)
        print("✅ EXPERIMENT #001 评估完成!")
        print("=" * 60)
        print("🎉 系统迁移验证评估成功!")
        print("   基准性能指标已建立")
        print("   评估流程运行正常")
        print("   为后续实验提供对比基准")
        print()
        print("📋 下一步:")
        print("   1. 查看生成的评估报告和图表")
        print("   2. 记录基准性能指标用于对比")
        print("   3. 进入 Experiment #002 可视化集成实验")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 评估失败: {e}")
        print("🔧 可能的问题:")
        print("   1. 模型文件损坏或不兼容")
        print("   2. 评估数据集路径不正确")
        print("   3. 评估参数配置有误")
        return False
    except Exception as e:
        print(f"\n❌ 执行错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)