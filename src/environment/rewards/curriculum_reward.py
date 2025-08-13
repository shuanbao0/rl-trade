"""
课程学习奖励函数模块

基于2024-2025年最新课程学习研究，实现渐进式复杂度递增的奖励函数系统。
核心技术包括：
- 多阶段奖励课程设计
- 自适应难度调整
- 性能驱动的阶段转换
- 层次化课程结构
- 交易策略复杂度递进

参考文献：
- Curriculum Learning for Reinforcement Learning (2024)
- Self-Supervised Success Discriminator for Curriculum Learning (2024) 
- Multi-Agent Curriculum Learning with Collaborative Optimization (2024)
- Progressive Reward Complexity in Financial Trading (2025)
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque

from .base_reward import BaseRewardScheme

# 配置日志
logger = logging.getLogger(__name__)

class CurriculumStage(Enum):
    """课程学习阶段枚举"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class MarketComplexity(Enum):
    """市场复杂度等级"""
    LOW_VOLATILITY = "low_volatility"
    MODERATE_VOLATILITY = "moderate_volatility"
    HIGH_VOLATILITY = "high_volatility"
    EXTREME_CONDITIONS = "extreme_conditions"

@dataclass
class CurriculumStageConfig:
    """课程阶段配置"""
    stage: CurriculumStage
    market_complexity: MarketComplexity
    reward_weights: Dict[str, float]
    success_threshold: float
    min_episodes: int
    max_episodes: int
    transition_criteria: Dict[str, float]

@dataclass
class PerformanceMetrics:
    """性能指标"""
    success_rate: float
    average_reward: float
    reward_variance: float
    critic_actor_fit: float
    policy_stability: float
    consistency_score: float

class SuccessDiscriminator:
    """成功判别器 - 基于2024年最新研究"""
    
    def __init__(self, lookback_window: int = 50):
        self.lookback_window = lookback_window
        self.success_history = deque(maxlen=lookback_window)
        self.performance_threshold = 0.6
        
    def estimate_success_probability(self, recent_rewards: List[float]) -> float:
        """估计成功概率"""
        if len(recent_rewards) < 10:
            return 0.5  # 默认概率
        
        # 计算成功率（正奖励的比例）
        positive_rewards = sum(1 for r in recent_rewards if r > 0)
        success_rate = positive_rewards / len(recent_rewards)
        
        # 考虑奖励方差 - 稳定性指标
        reward_std = np.std(recent_rewards)
        stability_bonus = max(0, 1 - reward_std) * 0.2
        
        return min(1.0, success_rate + stability_bonus)
    
    def should_advance(self, recent_performance: PerformanceMetrics) -> bool:
        """判断是否应该进入下一阶段"""
        criteria = [
            recent_performance.success_rate > self.performance_threshold,
            recent_performance.policy_stability < 0.3,  # 策略稳定
            recent_performance.critic_actor_fit > 0.7   # critic-actor拟合度
        ]
        
        return sum(criteria) >= 2  # 至少满足2个条件

class CurriculumManager:
    """课程管理器"""
    
    def __init__(self):
        self.stage_configs = self._initialize_stage_configs()
        self.current_stage = CurriculumStage.BEGINNER
        self.episodes_in_stage = 0
        self.success_discriminator = SuccessDiscriminator()
        self.stage_history = []
        
    def _initialize_stage_configs(self) -> Dict[CurriculumStage, CurriculumStageConfig]:
        """初始化阶段配置"""
        return {
            CurriculumStage.BEGINNER: CurriculumStageConfig(
                stage=CurriculumStage.BEGINNER,
                market_complexity=MarketComplexity.LOW_VOLATILITY,
                reward_weights={
                    'profit': 1.0,
                    'risk': 0.0,
                    'consistency': 0.0,
                    'efficiency': 0.0
                },
                success_threshold=0.55,
                min_episodes=50,
                max_episodes=200,
                transition_criteria={
                    'success_rate': 0.55,
                    'stability': 0.3,
                    'fit_score': 0.6
                }
            ),
            CurriculumStage.INTERMEDIATE: CurriculumStageConfig(
                stage=CurriculumStage.INTERMEDIATE,
                market_complexity=MarketComplexity.MODERATE_VOLATILITY,
                reward_weights={
                    'profit': 0.7,
                    'risk': 0.3,
                    'consistency': 0.0,
                    'efficiency': 0.0
                },
                success_threshold=0.6,
                min_episodes=100,
                max_episodes=300,
                transition_criteria={
                    'success_rate': 0.6,
                    'stability': 0.25,
                    'fit_score': 0.7
                }
            ),
            CurriculumStage.ADVANCED: CurriculumStageConfig(
                stage=CurriculumStage.ADVANCED,
                market_complexity=MarketComplexity.HIGH_VOLATILITY,
                reward_weights={
                    'profit': 0.5,
                    'risk': 0.3,
                    'consistency': 0.15,
                    'efficiency': 0.05
                },
                success_threshold=0.65,
                min_episodes=150,
                max_episodes=400,
                transition_criteria={
                    'success_rate': 0.65,
                    'stability': 0.2,
                    'fit_score': 0.75
                }
            ),
            CurriculumStage.EXPERT: CurriculumStageConfig(
                stage=CurriculumStage.EXPERT,
                market_complexity=MarketComplexity.EXTREME_CONDITIONS,
                reward_weights={
                    'profit': 0.4,
                    'risk': 0.35,
                    'consistency': 0.2,
                    'efficiency': 0.05
                },
                success_threshold=0.7,
                min_episodes=200,
                max_episodes=float('inf'),
                transition_criteria={
                    'success_rate': 0.8,  # 高要求，很难达到
                    'stability': 0.15,
                    'fit_score': 0.85
                }
            )
        }
    
    def get_current_config(self) -> CurriculumStageConfig:
        """获取当前阶段配置"""
        return self.stage_configs[self.current_stage]
    
    def should_advance_stage(self, performance: PerformanceMetrics) -> bool:
        """判断是否应该进入下一阶段"""
        config = self.get_current_config()
        
        # 检查最小episode要求
        if self.episodes_in_stage < config.min_episodes:
            return False
        
        # 检查最大episode限制
        if self.episodes_in_stage >= config.max_episodes:
            return True
        
        # 使用成功判别器
        should_advance = self.success_discriminator.should_advance(performance)
        
        # 额外检查阶段特定条件
        criteria_met = (
            performance.success_rate >= config.transition_criteria['success_rate'] and
            performance.policy_stability <= config.transition_criteria['stability'] and
            performance.critic_actor_fit >= config.transition_criteria['fit_score']
        )
        
        return should_advance and criteria_met
    
    def advance_to_next_stage(self) -> bool:
        """进入下一阶段"""
        current_stages = list(CurriculumStage)
        current_index = current_stages.index(self.current_stage)
        
        if current_index < len(current_stages) - 1:
            self.stage_history.append({
                'from_stage': self.current_stage.value,
                'episodes_completed': self.episodes_in_stage,
                'timestamp': self.episodes_in_stage  # 简化的时间戳
            })
            
            self.current_stage = current_stages[current_index + 1]
            self.episodes_in_stage = 0
            
            logger.info(f"Advanced to {self.current_stage.value} stage")
            return True
        
        return False  # 已经是最高阶段

class DifficultyAdapter:
    """难度自适应器"""
    
    def __init__(self):
        self.volatility_multipliers = {
            MarketComplexity.LOW_VOLATILITY: 0.5,
            MarketComplexity.MODERATE_VOLATILITY: 1.0,
            MarketComplexity.HIGH_VOLATILITY: 1.5,
            MarketComplexity.EXTREME_CONDITIONS: 2.0
        }
        
    def adapt_market_complexity(self, 
                               base_data: np.ndarray, 
                               target_complexity: MarketComplexity) -> np.ndarray:
        """根据目标复杂度调整市场数据"""
        if len(base_data) < 2:
            return base_data
            
        multiplier = self.volatility_multipliers[target_complexity]
        
        # 计算收益率
        returns = np.diff(base_data) / base_data[:-1]
        
        # 调整波动率
        adjusted_returns = returns * multiplier
        
        # 重构价格序列
        adjusted_data = np.zeros_like(base_data)
        adjusted_data[0] = base_data[0]
        
        for i in range(1, len(adjusted_data)):
            adjusted_data[i] = adjusted_data[i-1] * (1 + adjusted_returns[i-1])
        
        return adjusted_data
    
    def get_complexity_penalty(self, complexity: MarketComplexity) -> float:
        """获取复杂度惩罚"""
        penalties = {
            MarketComplexity.LOW_VOLATILITY: 0.0,
            MarketComplexity.MODERATE_VOLATILITY: 0.1,
            MarketComplexity.HIGH_VOLATILITY: 0.2,
            MarketComplexity.EXTREME_CONDITIONS: 0.3
        }
        return penalties.get(complexity, 0.0)

class CurriculumReward(BaseRewardScheme):
    """
    课程学习奖励函数
    
    基于2024-2025年最新课程学习研究的多阶段奖励函数，
    支持渐进式复杂度递增和自适应阶段转换。
    """
    
    def __init__(self,
                 enable_auto_progression: bool = True,
                 manual_stage: Optional[str] = None,
                 progression_sensitivity: float = 1.0,
                 performance_window: int = 50,
                 initial_balance: float = 10000.0):
        """
        初始化课程学习奖励函数
        
        Args:
            enable_auto_progression: 是否启用自动阶段转换
            manual_stage: 手动设置阶段 (beginner/intermediate/advanced/expert)
            progression_sensitivity: 阶段转换敏感度
            performance_window: 性能评估窗口大小
            initial_balance: 初始余额
        """
        super().__init__(initial_balance=initial_balance)
        
        self.enable_auto_progression = enable_auto_progression
        self.progression_sensitivity = progression_sensitivity
        self.performance_window = performance_window
        
        # 初始化组件
        self.curriculum_manager = CurriculumManager()
        self.difficulty_adapter = DifficultyAdapter()
        
        # 手动设置阶段
        if manual_stage:
            try:
                stage = CurriculumStage(manual_stage.lower())
                self.curriculum_manager.current_stage = stage
                logger.info(f"Manually set stage to {stage.value}")
            except ValueError:
                logger.warning(f"Invalid manual stage: {manual_stage}, using default")
        
        # 性能追踪
        self.reward_history = deque(maxlen=performance_window)
        self.episode_count = 0
        self.stage_transition_history = []
        
        # 组件奖励计算
        self.component_calculators = {
            'profit': self._calculate_profit_reward,
            'risk': self._calculate_risk_reward,
            'consistency': self._calculate_consistency_reward,
            'efficiency': self._calculate_efficiency_reward
        }
        
        logger.info(f"CurriculumReward initialized, starting stage: {self.curriculum_manager.current_stage.value}")
    
    def calculate_reward(self, portfolio_value: float, action: float, price: float, 
                        portfolio_info: Dict, trade_info: Dict, step: int, **kwargs) -> float:
        """
        奖励计算接口 - 课程学习奖励
        
        Args:
            portfolio_value: 当前投资组合价值
            action: 执行的动作
            price: 当前价格
            portfolio_info: 投资组合详细信息
            trade_info: 交易执行信息
            step: 当前步数
            **kwargs: 其他参数
            
        Returns:
            float: 计算得到的奖励值
        """
        try:
            # 获取当前阶段配置
            config = self.curriculum_manager.get_current_config()
            
            # 构建portfolio数据用于现有的组件计算器
            portfolio_data = {
                'portfolio_value': portfolio_value,
                'action': action,
                'price': price,
                'portfolio_info': portfolio_info,
                'trade_info': trade_info,
                'step': step,
                **kwargs
            }
            
            # 计算各组件奖励
            component_rewards = {}
            for component, weight in config.reward_weights.items():
                if component in self.component_calculators:
                    component_rewards[component] = self.component_calculators[component](portfolio_data) * weight
                else:
                    component_rewards[component] = 0.0
            
            # 综合奖励
            total_reward = sum(component_rewards.values())
            
            # 复杂度惩罚
            complexity_penalty = self.difficulty_adapter.get_complexity_penalty(config.market_complexity)
            adjusted_reward = total_reward * (1 - complexity_penalty)
            
            # 记录奖励历史
            self.reward_history.append(adjusted_reward)
            self.step_count += 1
            
            # 检查阶段转换
            if self.enable_auto_progression and len(self.reward_history) >= self.performance_window:
                self._check_stage_progression()
            
            return float(adjusted_reward)
            
        except Exception as e:
            logger.error(f"CurriculumReward计算异常: {e}")
            return 0.0
    
    def reward(self, portfolio) -> float:
        """计算奖励值 (BaseRewardScheme接口)"""
        return self.get_reward(portfolio)
    
    def get_reward(self, portfolio) -> float:
        """计算课程学习奖励"""
        try:
            # 获取当前阶段配置
            config = self.curriculum_manager.get_current_config()
            
            # 计算各组件奖励
            component_rewards = {}
            for component, weight in config.reward_weights.items():
                if weight > 0:
                    component_rewards[component] = self.component_calculators[component](portfolio)
            
            # 计算加权奖励
            total_reward = sum(
                component_rewards.get(comp, 0) * weight 
                for comp, weight in config.reward_weights.items()
            )
            
            # 应用市场复杂度调整
            complexity_penalty = self.difficulty_adapter.get_complexity_penalty(
                config.market_complexity
            )
            adjusted_reward = total_reward * (1 - complexity_penalty)
            
            # 记录奖励历史
            self.reward_history.append(adjusted_reward)
            self.episode_count += 1
            self.curriculum_manager.episodes_in_stage += 1
            
            # 检查阶段转换（如果启用）
            if self.enable_auto_progression and len(self.reward_history) >= 20:
                self._check_stage_transition()
            
            return float(adjusted_reward)
            
        except Exception as e:
            logger.error(f"Curriculum reward calculation failed: {e}")
            return 0.001
    
    def _calculate_profit_reward(self, portfolio) -> float:
        """计算利润奖励"""
        current_value = portfolio.net_worth
        
        if hasattr(portfolio, '_curriculum_previous_value'):
            return_rate = (current_value - portfolio._curriculum_previous_value) / portfolio._curriculum_previous_value
            portfolio._curriculum_previous_value = current_value
            return return_rate
        else:
            portfolio._curriculum_previous_value = current_value
            return 0.0
    
    def _calculate_risk_reward(self, portfolio) -> float:
        """计算风险调整奖励"""
        if not hasattr(portfolio, '_curriculum_value_history'):
            portfolio._curriculum_value_history = deque(maxlen=50)
        
        portfolio._curriculum_value_history.append(portfolio.net_worth)
        
        if len(portfolio._curriculum_value_history) < 10:
            return 0.0
        
        values = np.array(portfolio._curriculum_value_history)
        returns = np.diff(values) / values[:-1]
        
        if len(returns) < 2:
            return 0.0
        
        # 简单的风险调整（负的波动率）
        volatility = np.std(returns)
        return -volatility
    
    def _calculate_consistency_reward(self, portfolio) -> float:
        """计算一致性奖励"""
        if len(self.reward_history) < 10:
            return 0.0
        
        recent_rewards = list(self.reward_history)[-10:]
        
        # 奖励一致性（低方差）
        consistency = 1.0 / (1.0 + np.var(recent_rewards))
        return consistency * 0.1  # 缩放到合适范围
    
    def _calculate_efficiency_reward(self, portfolio) -> float:
        """计算效率奖励"""
        # 简单的效率指标：奖励与交易次数的比率
        if not hasattr(portfolio, '_curriculum_trade_count'):
            portfolio._curriculum_trade_count = 0
        
        # 假设的交易计数逻辑（实际应根据具体实现）
        portfolio._curriculum_trade_count += 1
        
        if portfolio._curriculum_trade_count > 0 and len(self.reward_history) > 0:
            avg_reward = np.mean(list(self.reward_history))
            efficiency = avg_reward / max(portfolio._curriculum_trade_count, 1)
            return efficiency * 0.01  # 缩放
        
        return 0.0
    
    def _check_stage_transition(self):
        """检查并执行阶段转换"""
        try:
            # 计算当前性能指标
            performance = self._calculate_performance_metrics()
            
            # 判断是否应该转换
            if self.curriculum_manager.should_advance_stage(performance):
                old_stage = self.curriculum_manager.current_stage.value
                success = self.curriculum_manager.advance_to_next_stage()
                
                if success:
                    new_stage = self.curriculum_manager.current_stage.value
                    self.stage_transition_history.append({
                        'from': old_stage,
                        'to': new_stage,
                        'episode': self.episode_count,
                        'performance': performance
                    })
                    logger.info(f"Curriculum stage advanced: {old_stage} -> {new_stage}")
                    
        except Exception as e:
            logger.error(f"Stage transition check failed: {e}")
    
    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """计算性能指标"""
        if len(self.reward_history) < 10:
            return PerformanceMetrics(0.5, 0.0, 1.0, 0.5, 1.0, 0.5)
        
        recent_rewards = list(self.reward_history)
        
        # 成功率（正奖励比例）
        success_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
        
        # 平均奖励
        average_reward = np.mean(recent_rewards)
        
        # 奖励方差
        reward_variance = np.var(recent_rewards)
        
        # 模拟的critic-actor拟合度（基于奖励稳定性）
        critic_actor_fit = max(0, 1 - reward_variance)
        
        # 策略稳定性（基于奖励方差）
        policy_stability = reward_variance
        
        # 一致性得分
        consistency_score = 1.0 / (1.0 + reward_variance)
        
        return PerformanceMetrics(
            success_rate=success_rate,
            average_reward=average_reward,
            reward_variance=reward_variance,
            critic_actor_fit=critic_actor_fit,
            policy_stability=policy_stability,
            consistency_score=consistency_score
        )
    
    def get_curriculum_info(self) -> Dict[str, Any]:
        """获取课程学习信息"""
        config = self.curriculum_manager.get_current_config()
        performance = self._calculate_performance_metrics()
        
        return {
            'current_stage': config.stage.value,
            'market_complexity': config.market_complexity.value,
            'reward_weights': config.reward_weights,
            'episodes_in_stage': self.curriculum_manager.episodes_in_stage,
            'total_episodes': self.episode_count,
            'success_threshold': config.success_threshold,
            'stage_transition_history': self.stage_transition_history,
            'current_performance': {
                'success_rate': performance.success_rate,
                'average_reward': performance.average_reward,
                'policy_stability': performance.policy_stability,
                'consistency_score': performance.consistency_score
            },
            'progression_enabled': self.enable_auto_progression,
            'can_advance': self.curriculum_manager.should_advance_stage(performance) if len(self.reward_history) >= 20 else False
        }
    
    def manual_advance_stage(self) -> bool:
        """手动进入下一阶段"""
        success = self.curriculum_manager.advance_to_next_stage()
        if success:
            logger.info(f"Manually advanced to {self.curriculum_manager.current_stage.value}")
        return success
    
    def set_stage(self, stage_name: str) -> bool:
        """设置当前阶段"""
        try:
            stage = CurriculumStage(stage_name.lower())
            old_stage = self.curriculum_manager.current_stage.value
            self.curriculum_manager.current_stage = stage
            self.curriculum_manager.episodes_in_stage = 0
            logger.info(f"Stage set: {old_stage} -> {stage.value}")
            return True
        except ValueError:
            logger.error(f"Invalid stage name: {stage_name}")
            return False
    
    def reset(self):
        """重置奖励函数状态"""
        # 保留课程进度，只重置当前episode的状态
        self.reward_history.clear()
        logger.info("CurriculumReward episode reset completed")
    
    def reset_curriculum(self):
        """完全重置课程学习进度"""
        self.curriculum_manager = CurriculumManager()
        self.reward_history.clear()
        self.episode_count = 0
        self.stage_transition_history.clear()
        logger.info("CurriculumReward curriculum reset completed")
    
    @staticmethod
    def get_reward_info() -> Dict[str, Any]:
        """获取奖励函数信息"""
        return {
            'name': 'Curriculum Learning Reward Function',
            'description': 'Multi-stage progressive reward function with automatic difficulty adaptation '
                          'based on 2024-2025 curriculum learning research',
            'category': 'Advanced Curriculum Learning',
            'complexity': 'High',
            'parameters': {
                'enable_auto_progression': 'Enable automatic stage progression based on performance',
                'manual_stage': 'Manually set curriculum stage (beginner/intermediate/advanced/expert)',
                'progression_sensitivity': 'Sensitivity for stage transition decisions',
                'performance_window': 'Number of episodes for performance evaluation',
                'initial_balance': 'Initial portfolio balance'
            },
            'features': [
                'Multi-stage curriculum progression',
                'Adaptive difficulty adjustment',
                'Performance-based stage transitions',
                'Success discriminator for progression',
                'Market complexity adaptation',
                'Progressive reward complexity',
                'Automated curriculum management',
                'Manual stage control',
                'Comprehensive progress tracking'
            ],
            'curriculum_stages': [
                'Beginner: Simple profit/loss optimization',
                'Intermediate: Risk-adjusted returns (Sharpe ratio)',
                'Advanced: Multi-objective optimization',
                'Expert: Complex market conditions with all factors'
            ],
            'use_cases': [
                'Progressive agent training',
                'Curriculum-based strategy development',
                'Adaptive trading complexity',
                'Educational trading simulations',
                'Research in progressive learning'
            ],
            'computational_complexity': 'Medium (stage management and performance tracking)',
            'memory_usage': 'Low to Medium (performance history)',
            'recommended_for': 'Training environments requiring progressive complexity increase'
        }

# 为向后兼容性提供别名
CurriculumLearningReward = CurriculumReward