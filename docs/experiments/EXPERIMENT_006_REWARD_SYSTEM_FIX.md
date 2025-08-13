# Experiment #006: 奖励函数系统修复与EURUSD优化

## 实验概述

**实验编号**: #006  
**实验名称**: 奖励函数系统性修复与EURUSD外汇交易优化  
**实验日期**: 2025年8月12日 - 设计开始  
**实验状态**: 🟡 设计阶段  
**实验优先级**: 🔴 CRITICAL - 解决系统根本问题

---

## 实验背景与动机

### 关键问题识别
基于对实验003A-005的深度分析，发现了导致所有EURUSD实验失败的**五个系统性根本原因**：

#### 1. 🚨 奖励函数设计根本性缺陷 (CRITICAL)
```python
# 所有实验的共同致命问题
实验003A: 回报 -90.00%
实验004:  奖励 +94542 vs 回报 -43.69%  (完全脱钩)
实验005-S1: 奖励 +1159 vs 回报 -65.37% (持续脱钩)
实验005-S2: 奖励 +1152 vs 回报 -63.76% (持续脱钩)

# 核心问题：奖励与实际盈亏零相关或负相关
```

#### 2. 🌍 EURUSD外汇市场适应性问题
- **数据特性不同**: 外汇24/7交易，波动性模式与股票不同
- **交易机制差异**: 点差、滑点、杠杆等外汇特有因素
- **特征工程不当**: 用股票思维设计的指标不适合外汇

#### 3. 📈 系统性过拟合问题
- **训练收敛但评估失败**: 所有实验胜率0%
- **完美一致性异常**: 所有episode结果完全相同
- **泛化能力为零**: 无法适应新的市场数据

#### 4. 🛠️ 特征工程质量问题
- **基础特征无效**: 即使3个基础特征也导致-65%亏损
- **特征质量低下**: 问题不在数量，而在质量

#### 5. ⚙️ 训练环境配置问题
- **确定性过强**: 缺乏必要的随机性
- **初始化问题**: 可能存在不当的环境设置

---

## 实验6设计理念

### 核心设计原则

#### 1. 🎯 问题优先级导向
按照问题严重程度逐一解决：
1. **首要**: 修复奖励函数（解决奖励-回报脱钩）
2. **重要**: EURUSD外汇市场专业化适配
3. **必要**: 实现严格的样本外验证
4. **优化**: 基于金融理论的特征重构

#### 2. 🔬 科学验证方法
- **对比实验**: 新旧奖励函数直接对比
- **相关性监控**: 实时监控奖励-回报相关性
- **统计显著性**: 所有改进都要求统计显著 (p<0.05)
- **可重现性**: 多次运行验证结果稳定性

#### 3. 🎛️ 渐进式改进
- **阶段1**: 修复奖励函数，建立基准
- **阶段2**: 外汇市场专业化改进
- **阶段3**: 系统优化和完善

---

## 详细实验设计

### 阶段1: 奖励函数修复验证 (关键阶段)

#### 目标
**彻底解决奖励-回报脱钩问题，建立奖励函数与实际交易表现的强相关性**

#### 新奖励函数设计: `direct_pnl_reward`
```python
class DirectPnLReward(BaseReward):
    """
    直接基于盈亏的奖励函数
    解决所有历史实验的奖励-回报脱钩问题
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.initial_balance = config.initial_balance
        self.transaction_cost_rate = 0.0002  # EURUSD点差约2点
        self.position_penalty = 0.001        # 轻微仓位惩罚
        self.prev_balance = self.initial_balance
        
    def calculate_reward(self, prev_portfolio, current_portfolio, action):
        """
        直接基于投资组合价值变化计算奖励
        确保奖励与实际盈亏100%相关
        """
        # 1. 计算实际盈亏变化
        balance_change = current_portfolio - prev_portfolio
        return_rate = balance_change / prev_portfolio if prev_portfolio > 0 else 0
        
        # 2. 计算交易成本
        position_change = abs(action - self.prev_action) if hasattr(self, 'prev_action') else 0
        transaction_cost = position_change * self.transaction_cost_rate * current_portfolio
        
        # 3. 基础奖励 = 实际收益率
        base_reward = return_rate
        
        # 4. 扣除交易成本
        cost_penalty = transaction_cost / current_portfolio if current_portfolio > 0 else 0
        
        # 5. 最终奖励 = 净收益率
        final_reward = base_reward - cost_penalty
        
        # 6. 数值范围控制（避免异常值）
        final_reward = np.clip(final_reward, -0.1, 0.1)  # ±10%范围
        
        # 7. 放大到合理数值范围
        final_reward *= 100  # 转换为百分比形式
        
        self.prev_action = action
        return final_reward
    
    def get_reward_info(self):
        return {
            "name": "DirectPnLReward",
            "description": "直接基于投资组合盈亏的奖励函数",
            "expected_correlation": "> 0.95",
            "reward_range": "[-10, +10]"
        }
```

#### 验证实验配置
```python
STAGE1_CONFIG = {
    'experiment_name': 'reward_function_fix_validation',
    'duration': '45分钟快速验证',
    'primary_objective': '验证奖励-回报相关性 > 0.8',
    
    'training_config': {
        'algorithm': 'PPO',
        'total_timesteps': 500_000,  # 快速验证
        'learning_rate': 3e-4,
        'batch_size': 64,
        'n_steps': 1024
    },
    
    'data_config': {
        'symbol': 'EURUSD=X',
        'period': '6mo',  # 减少数据量，加快验证
        'split': {'train': 0.7, 'validation': 0.15, 'test': 0.15}
    },
    
    'feature_config': {
        'features': ['Close', 'SMA_14', 'RSI_14'],  # 最基础3特征
        'preprocessing': 'standard_scaling'
    },
    
    'success_criteria': {
        'reward_return_correlation': '> 0.8',
        'training_convergence': 'loss下降且稳定',
        'reward_range_reasonable': '±10范围内',
        'no_infinite_values': '无异常数值'
    }
}
```

### 阶段2: EURUSD外汇专业化 (重要阶段)

#### 外汇市场专用特征工程
```python
class ForexFeatureEngineer:
    """
    专为EURUSD设计的特征工程器
    基于外汇交易理论和实践
    """
    
    def __init__(self):
        self.forex_sessions = {
            'asian': (0, 8),      # 亚洲时段
            'london': (8, 16),    # 伦敦时段  
            'new_york': (13, 21)  # 纽约时段
        }
    
    def create_forex_features(self, data):
        """创建外汇专用特征"""
        features = {}
        
        # 1. 价格动量特征 (基于外汇特性优化)
        features['price_momentum'] = data['Close'].pct_change(14)
        
        # 2. 波动性特征 (外汇波动性管理关键)
        features['atr_normalized'] = self._calculate_normalized_atr(data, 14)
        
        # 3. 趋势强度特征 (外汇趋势跟踪重要)
        features['trend_strength'] = self._calculate_trend_strength(data, 21)
        
        # 4. 交易时段特征 (外汇24/5特性)
        features['session_volatility'] = self._calculate_session_volatility(data)
        
        # 5. 货币强度特征 (外汇独有)
        features['currency_strength'] = self._calculate_currency_strength(data)
        
        return pd.DataFrame(features, index=data.index).fillna(method='ffill').fillna(0)
    
    def _calculate_normalized_atr(self, data, period):
        """标准化ATR，适应外汇波动性"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        
        # 标准化到[0,1]范围
        atr_normalized = (atr - atr.rolling(100).min()) / (atr.rolling(100).max() - atr.rolling(100).min())
        return atr_normalized.fillna(0)
    
    def _calculate_trend_strength(self, data, period):
        """计算趋势强度"""
        ema = data['Close'].ewm(span=period).mean()
        trend_slope = ema.diff(5) / ema.shift(5)  # 5日斜率
        return trend_slope
    
    def _calculate_session_volatility(self, data):
        """计算交易时段波动性"""
        # 简化版本：基于小时计算相对波动性
        volatility = data['High'] / data['Low'] - 1
        return volatility.rolling(24).mean()  # 24小时滚动平均
    
    def _calculate_currency_strength(self, data):
        """计算货币相对强度（简化版本）"""
        # 基于价格变化的相对强度
        price_change = data['Close'].pct_change()
        strength = price_change.rolling(20).mean() / price_change.rolling(20).std()
        return strength.fillna(0)
```

#### EURUSD专用交易环境
```python
class ForexTradingEnvironment:
    """
    专为EURUSD优化的交易环境
    """
    
    def __init__(self, data, config):
        self.data = data
        self.config = config
        
        # EURUSD特有参数
        self.pip_value = 0.0001
        self.typical_spread = 1.2  # 点
        self.min_lot_size = 0.01
        self.max_leverage = 30
        
        # 外汇交易时段
        self.active_sessions = self._identify_active_sessions()
        
    def _calculate_spread_cost(self, action, current_price):
        """计算真实的点差成本"""
        position_change = abs(action - self.current_position)
        spread_cost = position_change * self.typical_spread * self.pip_value * current_price
        return spread_cost
    
    def _apply_forex_risk_management(self, action):
        """应用外汇风险管理规则"""
        # 1. 最大仓位限制
        max_position = min(0.8, self.account_balance / 1000)  # 动态仓位限制
        action = np.clip(action, -max_position, max_position)
        
        # 2. 止损检查
        if self._check_stop_loss():
            action = 0  # 强制平仓
            
        return action
    
    def _identify_active_sessions(self):
        """识别活跃交易时段"""
        # 基于历史数据识别高波动时段
        volatility = self.data['High'] / self.data['Low'] - 1
        active_threshold = volatility.quantile(0.7)  # 70%分位数以上为活跃
        return volatility > active_threshold
```

### 阶段3: 严格验证机制 (防过拟合)

#### 时间序列样本外验证
```python
class TimeSeriesValidator:
    """
    专门用于时间序列数据的严格验证
    防止未来数据泄露和过拟合
    """
    
    def __init__(self, data, config):
        self.data = data
        self.config = config
        
    def create_time_aware_splits(self):
        """创建时间感知的数据分割"""
        total_length = len(self.data)
        
        # 严格的时间序列分割
        splits = {
            'train_start': 0,
            'train_end': int(total_length * 0.6),     # 60%训练
            'val_start': int(total_length * 0.6),
            'val_end': int(total_length * 0.8),       # 20%验证
            'test_start': int(total_length * 0.8),
            'test_end': total_length                  # 20%测试（严格保留）
        }
        
        return splits
    
    def walk_forward_validation(self, model_trainer, window_size=1000):
        """滚动窗口验证"""
        results = []
        
        for start_idx in range(0, len(self.data) - window_size, window_size // 4):
            end_idx = start_idx + window_size
            
            # 训练窗口
            train_data = self.data.iloc[start_idx:end_idx-200]
            # 测试窗口（未来数据）
            test_data = self.data.iloc[end_idx-200:end_idx]
            
            # 训练模型
            model = model_trainer.train_on_window(train_data)
            
            # 测试模型
            test_result = model_trainer.evaluate_on_window(model, test_data)
            results.append(test_result)
            
        return results
    
    def monte_carlo_validation(self, model_trainer, n_runs=10):
        """蒙特卡洛验证"""
        results = []
        
        for run in range(n_runs):
            # 设置不同随机种子
            np.random.seed(42 + run)
            
            # 随机选择训练起始点（保持时间序列顺序）
            max_start = len(self.data) - 2000
            start_idx = np.random.randint(0, max_start // 2)
            
            splits = self.create_time_aware_splits()
            # 训练和评估
            result = model_trainer.train_and_evaluate(self.data, splits, seed=42+run)
            results.append(result)
            
        return results
```

---

## 成功标准与评估指标

### 阶段1成功标准: 奖励函数修复
```python
STAGE1_SUCCESS_CRITERIA = {
    'critical': {
        'reward_return_correlation': {'target': '> 0.8', 'weight': 0.4},
        'no_infinite_rewards': {'target': 'True', 'weight': 0.3},
        'training_convergence': {'target': 'True', 'weight': 0.3}
    },
    
    'important': {
        'mean_return_improvement': {'target': '> -50%', 'baseline': '-65%'},
        'reward_range_reasonable': {'target': '[-10, +10]'},
        'std_return_nonzero': {'target': '> 0.01'}  # 打破完全一致性
    }
}
```

### 阶段2成功标准: EURUSD专业化
```python
STAGE2_SUCCESS_CRITERIA = {
    'performance': {
        'mean_return': {'target': '> -30%', 'improvement': '15%+'},
        'win_rate': {'target': '> 15%', 'baseline': '0%'},
        'sharpe_ratio': {'target': '> -2.0', 'baseline': '-6375'}
    },
    
    'stability': {
        'validation_consistency': {'target': 'train/val gap < 20%'},
        'out_of_sample_performance': {'target': '在测试集上不恶化'}
    }
}
```

### 最终成功标准
```python
FINAL_SUCCESS_CRITERIA = {
    'minimum_viable': {
        'mean_return': '> -20%',
        'reward_return_correlation': '> 0.8',
        'win_rate': '> 20%',
        'system_stability': 'True'
    },
    
    'target_performance': {
        'mean_return': '> -10%',
        'win_rate': '> 30%',
        'sharpe_ratio': '> -1.0',
        'max_drawdown': '< 40%'
    },
    
    'breakthrough_scenario': {
        'mean_return': '> 0%',  # 实际盈利
        'win_rate': '> 45%',
        'sharpe_ratio': '> 0.5',
        'consistent_across_periods': 'True'
    }
}
```

---

## 风险控制与应急方案

### 主要风险识别
1. **奖励函数修复失败**: 新奖励函数仍然无法建立正确相关性
2. **外汇专业化不足**: EURUSD特有问题未能充分解决
3. **过拟合持续存在**: 严格验证仍无法防止过拟合
4. **计算资源限制**: 实验执行时间可能超出预期

### 应急预案
```python
CONTINGENCY_PLANS = {
    'reward_fix_failure': {
        'fallback': '使用最简单的percentage_change奖励',
        'validation': '人工验证奖励计算逻辑',
        'timeline': '额外2天调试'
    },
    
    'forex_adaptation_failure': {
        'fallback': '回到基础特征，专注奖励修复',
        'alternative': '考虑其他外汇对比较实验',
        'timeline': '重新评估实验范围'
    },
    
    'persistent_overfitting': {
        'fallback': '极度简化模型和特征',
        'diagnostic': '深度诊断训练环境',
        'timeline': '可能需要架构级调整'
    }
}
```

---

## 预期影响与价值

### 技术价值
1. **解决根本问题**: 修复困扰所有实验的奖励函数问题
2. **建立外汇基准**: 为EURUSD交易建立可靠的性能基准
3. **方法论创新**: 创建适用于外汇RL的完整方法论

### 科学价值
1. **问题诊断**: 系统性识别和解决RL交易中的常见陷阱
2. **验证方法**: 建立时间序列RL的严格验证标准
3. **经验积累**: 为类似系统提供宝贵经验教训

### 商业价值
1. **系统可用性**: 建立真正可用的外汇交易RL系统
2. **风险管理**: 实现可控的交易风险管理
3. **扩展潜力**: 为其他外汇对和金融工具奠定基础

---

*实验设计完成时间: 2025年8月12日*  
*设计负责人: TensorTrade系统优化团队*  
*实验优先级: CRITICAL - 系统根本问题修复*  
*预期完成时间: 3-5天*