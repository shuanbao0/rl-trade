# TensorTrade 实验目录

## 📁 目录结构

```
experiments/
├── README.md                                # 本文件：使用说明
├── experiment_001_system_migration/         # 实验001：系统迁移验证
│   ├── train.py                            # 训练脚本
│   ├── evaluate.py                         # 评估脚本
│   ├── models/                             # 训练模型存储
│   └── results/                            # 评估结果存储
├── experiment_002_visualization/            # 实验002：可视化集成
│   ├── train.py                            # 训练脚本
│   ├── evaluate.py                         # 评估脚本
│   ├── models/                             # 训练模型存储
│   ├── results/                            # 评估结果存储
│   └── visualizations/                     # 可视化图表
├── experiment_003a_forex_reward/            # 实验003A：Forex奖励优化
│   ├── train.py                            # 训练脚本
│   ├── evaluate.py                         # 评估脚本
│   ├── models/                             # 训练模型存储
│   └── results/                            # 评估结果存储
└── experiment_004_enhanced_features/        # 实验004：117特征增强
    ├── train.py                            # 训练脚本
    ├── evaluate.py                         # 评估脚本
    ├── models/                             # 训练模型存储
    └── results/                            # 评估结果存储
```

---

## 🎯 实验概览

### Experiment #001: 系统迁移验证
**目标**: 验证从Ray RLlib到Stable-Baselines3的迁移成功性  
**特点**: 3个基础特征，简单奖励函数，基础验证  
**执行**:
```bash
cd experiments/experiment_001_system_migration
python train.py    # 训练模型（约15分钟）
python evaluate.py # 评估性能
```

### Experiment #002: 可视化系统集成
**目标**: 集成训练和评估可视化功能  
**特点**: 实时监控，图表生成，用户体验提升  
**执行**:
```bash
cd experiments/experiment_002_visualization
python train.py    # 训练模型（约1小时，含可视化）
python evaluate.py # 评估并生成图表
```

### Experiment #003A: Forex优化奖励函数
**目标**: 测试针对外汇交易优化的专用奖励函数  
**特点**: forex_optimized奖励，成本意识，风险调整  
**执行**:
```bash
cd experiments/experiment_003a_forex_reward
python train.py    # 训练模型（约2.5小时）
python evaluate.py # 详细性能分析
```

### Experiment #004: 117特征增强系统
**目标**: 验证从3特征到117专业外汇特征的性能突破  
**特点**: 高维特征空间，专业指标，革命性提升  
**执行**:
```bash
cd experiments/experiment_004_enhanced_features
python train.py    # 训练模型（约4-6小时）
python evaluate.py # 全面性能评估
```

---

## 🚀 快速开始

### 1. 环境要求
```bash
# 确保使用正确的Python环境
conda activate tensortrade_modern
```

### 2. 按序执行实验
建议按照编号顺序执行实验，每个实验都为后续实验提供基准对比：

```bash
# 第一步：系统基础验证
cd experiment_001_system_migration
python train.py && python evaluate.py

# 第二步：可视化集成
cd ../experiment_002_visualization  
python train.py && python evaluate.py

# 第三步：奖励函数优化
cd ../experiment_003a_forex_reward
python train.py && python evaluate.py

# 第四步：特征系统革命
cd ../experiment_004_enhanced_features
python train.py && python evaluate.py
```

### 3. 查看结果
每个实验完成后会生成：
- **模型文件**: `models/` 目录下的 `.zip` 文件
- **评估结果**: `results/` 目录下的 JSON 报告
- **可视化图表**: 相应目录下的 PNG 图片

---

## 📊 实验配置对比

| 实验 | 特征数 | 奖励函数 | 训练轮次 | 预期时间 | 主要目标 |
|------|--------|----------|----------|----------|----------|
| #001 | 3 | simple_return | 10K | 15分钟 | 系统验证 |
| #002 | 3 | simple_return | 50K | 1小时 | 可视化集成 |
| #003A | 3 | forex_optimized | 100K | 2.5小时 | 奖励优化 |
| #004 | 117 | forex_optimized | 150K | 4-6小时 | 特征革命 |

---

## 📈 性能进化轨迹

### 预期性能提升路径：
1. **Experiment #001** (基准): 建立基础性能基准
2. **Experiment #002** (可视化): 性能监控，基本相当
3. **Experiment #003A** (Forex奖励): 预期50-100%性能提升
4. **Experiment #004** (117特征): 预期100-300%性能突破

### 关键指标追踪：
- **平均Reward**: Episode奖励的平均值
- **Sharpe比率**: 风险调整后的收益指标
- **总收益率**: 投资组合总收益百分比
- **胜率**: 盈利交易占比
- **年化波动率**: 风险衡量指标

---

## 🛠️ 脚本设计理念

### 简化设计原则
实验脚本采用**参数调用**的简化设计：
- 每个实验的 `train.py` 和 `evaluate.py` 都通过 `subprocess.run()` 调用主程序
- 只需要配置实验特定的参数，无需重复实现复杂逻辑
- 保持代码简洁，便于维护和理解
- 确保所有实验使用统一的核心训练和评估逻辑

### 脚本架构
```python
# 典型的实验脚本结构
params = [
    sys.executable, "train_model.py",  # 调用主训练程序
    "--symbol", "EURUSD",              # 实验特定参数
    "--reward-type", "forex_optimized", # 实验配置
    "--iterations", "150",              # 其他参数...
]
subprocess.run(params, check=True)
```

## 🛠️ 脚本功能说明

### 训练脚本 (`train.py`)
**功能**:
- 使用 `subprocess` 调用主训练程序 `train_model.py`
- 传递实验特定的参数配置
- 提供实验说明和预期结果
- 监控训练过程和结果报告

**输出**:
- 训练好的模型文件 (保存在 `models/` 目录)
- 训练日志和统计信息
- 可视化图表 (如启用)

### 评估脚本 (`evaluate.py`)
**功能**:
- 使用 `subprocess` 调用主评估程序 `evaluate_model.py`
- 自动查找对应的训练模型
- 传递实验特定的评估参数
- 生成实验结果总结

**输出**:
- 详细评估报告 (保存在 `results/` 目录)
- 性能分析图表 (PNG)
- 对比分析结果

---

## 🎨 可视化功能

### 训练可视化 (Experiment #002+)
- **Loss曲线**: 训练损失趋势
- **Reward趋势**: Episode奖励变化
- **学习率调度**: 自适应学习率监控
- **梯度分析**: 训练健康度监控

### 评估可视化
- **投资组合性能**: 累积收益曲线
- **风险指标分析**: Sharpe, 回撤, VaR等
- **交易行为分析**: 交易频率，持仓分析
- **收益分布**: 统计分布图表

---

## ⚠️ 注意事项

### 依赖要求
- **数据集**: Experiment #004需要117特征数据集
- **计算资源**: 高维训练需要较多内存和时间
- **环境变量**: 确保使用tensortrade_modern环境

### 常见问题
1. **内存不足**: 降低batch_size或使用较少特征
2. **训练时间长**: 可以降低total_timesteps进行快速测试
3. **模型不存在**: 确保先运行train.py再运行evaluate.py

### 最佳实践
- **按序执行**: 建议按实验编号顺序执行
- **保存结果**: 每次实验后备份重要结果
- **对比分析**: 利用evaluate.py的对比功能分析改进
- **监控资源**: 长时间训练时监控系统资源使用

---

## 📋 结果文件说明

### 模型文件 (`models/*.zip`)
- Stable-Baselines3标准格式
- 包含完整的训练状态
- 可用于继续训练或推理

### 评估报告 (`results/*.json`)
- 详细的性能指标
- 原始评估数据
- 与基准的对比分析
- 实验配置记录

### 可视化图表 (`*.png`)
- 高分辨率图表文件
- 专业的分析图表
- 适合报告和演示使用

---

## 🎯 进阶使用

### 自定义实验
1. 复制现有实验目录
2. 修改训练参数配置
3. 调整奖励函数或特征
4. 运行并分析结果

### 批量实验
```bash
# 运行所有实验的脚本示例
#!/bin/bash
experiments=("001" "002" "003a" "004")
for exp in "${experiments[@]}"; do
    echo "Running Experiment $exp..."
    cd experiment_${exp}_*/
    python train.py && python evaluate.py
    cd ..
done
```

### 结果汇总
各实验的evaluate.py会自动进行跨实验对比，生成综合分析报告。

---

## 🎉 实验成果

通过完整执行这4个实验，您将获得：

1. **完整的系统验证**: 从基础功能到高级特性
2. **性能进化轨迹**: 清晰的改进路径和效果
3. **专业级交易系统**: 117特征的高维RL交易模型
4. **全面的分析报告**: 详细的性能指标和对比分析
5. **可视化监控系统**: 训练和评估的完整可视化
6. **研究基础**: 为后续研究和优化提供坚实基础

---

*最后更新: 2025年8月11日*  
*TensorTrade团队*