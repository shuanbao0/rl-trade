"""
奖励函数工厂模块

提供统一的奖励函数创建接口，支持通过配置参数动态创建不同类型的奖励函数。
"""

from typing import Dict, Any, Type, Optional, List
import logging
from .base_reward import BaseRewardScheme
from .risk_adjusted import RiskAdjustedReward
from .simple_return import SimpleReturnReward
from .profit_loss import ProfitLossReward
from .diversified import DiversifiedReward
from .log_sharpe import LogSharpeReward
from .return_drawdown import ReturnDrawdownReward
from .dynamic_sortino import DynamicSortinoReward
from .regime_aware import RegimeAwareReward
from .expert_committee import ExpertCommitteeReward
from .uncertainty_aware import UncertaintyAwareReward
from .curiosity_driven import CuriosityDrivenReward
from .self_rewarding import SelfRewardingReward
from .causal_reward import CausalReward
from .llm_guided import LLMGuidedReward
from .curriculum_reward import CurriculumReward
from .federated_reward import FederatedReward
from .meta_learning_reward import MetaLearningReward
from .rlhf_reward import RLHFReward
from .forex_optimized import ForexOptimizedReward
from .forex_simple import ForexSimpleReward
from .optimized_forex_reward import OptimizedForexReward
from .direct_pnl_reward import DirectPnLReward


class RewardFactory:
    """
    奖励函数工厂类
    
    负责根据配置参数创建对应的奖励函数实例。
    支持动态注册新的奖励函数类型。
    """
    
    # 内置奖励函数注册表
    _reward_registry: Dict[str, Type[BaseRewardScheme]] = {
        'risk_adjusted': RiskAdjustedReward,
        'simple_return': SimpleReturnReward,
        'profit_loss': ProfitLossReward,
        'diversified': DiversifiedReward,
        'log_sharpe': LogSharpeReward,
        'return_drawdown': ReturnDrawdownReward,
        'dynamic_sortino': DynamicSortinoReward,
        'regime_aware': RegimeAwareReward,
        'expert_committee': ExpertCommitteeReward,
        'uncertainty_aware': UncertaintyAwareReward,
        'curiosity_driven': CuriosityDrivenReward,
        'self_rewarding': SelfRewardingReward,
        'causal_reward': CausalReward,
        'llm_guided': LLMGuidedReward,
        'curriculum_reward': CurriculumReward,
        'federated_reward': FederatedReward,
        'meta_learning_reward': MetaLearningReward,
        'rlhf_reward': RLHFReward,
        'forex_optimized': ForexOptimizedReward,
        'forex_simple': ForexSimpleReward,
        'optimized_forex_reward': OptimizedForexReward,
        'experiment_005_reward': OptimizedForexReward,
        'direct_pnl': DirectPnLReward,
        'direct_pnl_reward': DirectPnLReward,
        'experiment_006_reward': DirectPnLReward,
        
        # 别名支持
        'default': RiskAdjustedReward,
        'sharpe': RiskAdjustedReward,
        'basic': SimpleReturnReward,
        'simple': SimpleReturnReward,
        'pnl': ProfitLossReward,
        'comprehensive': DiversifiedReward,
        'multi': DiversifiedReward,
        'differential_sharpe': LogSharpeReward,
        'dsr': LogSharpeReward,
        'log_dsr': LogSharpeReward,
        'calmar': ReturnDrawdownReward,
        'return_dd': ReturnDrawdownReward,
        'rdd': ReturnDrawdownReward,
        'drawdown': ReturnDrawdownReward,
        'dts': DynamicSortinoReward,
        'adaptive_sortino': DynamicSortinoReward,
        'time_varying_sortino': DynamicSortinoReward,
        'sortino': DynamicSortinoReward,
        'adaptive_expert': RegimeAwareReward,
        'market_state': RegimeAwareReward,
        'regime': RegimeAwareReward,
        'state_aware': RegimeAwareReward,
        'committee': ExpertCommitteeReward,
        'multi_objective': ExpertCommitteeReward,
        'morl': ExpertCommitteeReward,
        'experts': ExpertCommitteeReward,
        'pareto': ExpertCommitteeReward,
        'uncertainty': UncertaintyAwareReward,
        'bayesian': UncertaintyAwareReward,
        'confidence': UncertaintyAwareReward,
        'risk_sensitive': UncertaintyAwareReward,
        'cvar': UncertaintyAwareReward,
        'epistemic': UncertaintyAwareReward,
        'aleatoric': UncertaintyAwareReward,
        'curiosity': CuriosityDrivenReward,
        'intrinsic': CuriosityDrivenReward,
        'exploration': CuriosityDrivenReward,
        'intrinsic_motivation': CuriosityDrivenReward,
        'forward_model': CuriosityDrivenReward,
        'learning_progress': CuriosityDrivenReward,
        'hierarchical_rl': CuriosityDrivenReward,
        'dinat': CuriosityDrivenReward,
        'self_improving': SelfRewardingReward,
        'meta_ai': SelfRewardingReward,
        'llm_judge': SelfRewardingReward,
        'self_evaluation': SelfRewardingReward,
        'dpo': SelfRewardingReward,
        'iterative_improvement': SelfRewardingReward,
        'bias_detection': SelfRewardingReward,
        'meta_reward': SelfRewardingReward,
        'causal': CausalReward,
        'causal_inference': CausalReward,
        'confounding': CausalReward,
        'backdoor': CausalReward,
        'frontdoor': CausalReward,
        'dovi': CausalReward,
        'do_calculus': CausalReward,
        'causal_graph': CausalReward,
        'llm': LLMGuidedReward,
        'language_guided': LLMGuidedReward,
        'natural_language': LLMGuidedReward,
        'eureka': LLMGuidedReward,
        'constitutional': LLMGuidedReward,
        'ai_guided': LLMGuidedReward,
        'code_generation': LLMGuidedReward,
        'auto_reward': LLMGuidedReward,
        'smart_reward': LLMGuidedReward,
        'curriculum': CurriculumReward,
        'curriculum_learning': CurriculumReward,
        'progressive': CurriculumReward,
        'adaptive_difficulty': CurriculumReward,
        'staged_learning': CurriculumReward,
        'multi_stage': CurriculumReward,
        'beginner_to_expert': CurriculumReward,
        'difficulty_progression': CurriculumReward,
        'federated': FederatedReward,
        'distributed': FederatedReward,
        'collaborative': FederatedReward,
        'multi_client': FederatedReward,
        'privacy_preserving': FederatedReward,
        'differential_privacy': FederatedReward,
        'secure_aggregation': FederatedReward,
        'blockchain': FederatedReward,
        'smart_contracts': FederatedReward,
        'reputation_based': FederatedReward,
        'consensus': FederatedReward,
        'decentralized': FederatedReward,
        'meta_learning': MetaLearningReward,
        'maml': MetaLearningReward,
        'adaptive': MetaLearningReward,
        'meta_gradient': MetaLearningReward,
        'self_adapting': MetaLearningReward,
        'task_adaptive': MetaLearningReward,
        'few_shot': MetaLearningReward,
        'meta_optimization': MetaLearningReward,
        'gradient_based_meta': MetaLearningReward,
        'memory_augmented': MetaLearningReward,
        'agnostic_meta': MetaLearningReward,
        
        # RLHF (Reinforcement Learning from Human Feedback) 别名
        'rlhf': RLHFReward,
        'human_feedback': RLHFReward,
        'constitutional_ai': RLHFReward,
        'human_preference': RLHFReward,
        'expert_feedback': RLHFReward,
        'ppo_alignment': RLHFReward,
        'human_aligned': RLHFReward,
        'expert_guided': RLHFReward,
        'preference_learning': RLHFReward,
        'constitutional': RLHFReward,
        'ai_safety': RLHFReward,
        'human_centered': RLHFReward,
        
        # Forex专用别名
        'forex': ForexOptimizedReward,
        'currency': ForexOptimizedReward,
        'fx': ForexOptimizedReward,
        'pip_based': ForexOptimizedReward,
        'trend_following': ForexOptimizedReward,
        'forex_quality': ForexOptimizedReward,
        'eurusd': ForexOptimizedReward,
        'major_pairs': ForexOptimizedReward,
        'forex_basic': ForexSimpleReward,
        'fx_simple': ForexSimpleReward,
        'simple_forex': ForexSimpleReward,
        
        # Experiment #005 专用别名
        'optimized_forex': OptimizedForexReward,
        'experiment_005': OptimizedForexReward,
        'enhanced_forex': OptimizedForexReward,
        'reward_return_consistent': OptimizedForexReward,
        'stable_forex': OptimizedForexReward,
        'correlation_fixed': OptimizedForexReward,
    }
    
    @classmethod
    def create_reward_function(cls, 
                              reward_type: str = 'risk_adjusted',
                              **kwargs) -> BaseRewardScheme:
        """
        创建奖励函数实例
        
        Args:
            reward_type: 奖励函数类型
            **kwargs: 传递给奖励函数构造函数的参数
            
        Returns:
            BaseRewardScheme: 奖励函数实例
            
        Raises:
            ValueError: 当指定的奖励函数类型不存在时
        """
        reward_type = reward_type.lower().strip()
        
        if reward_type not in cls._reward_registry:
            available_types = list(cls._reward_registry.keys())
            raise ValueError(f"未知的奖励函数类型: '{reward_type}'. "
                           f"可用类型: {available_types}")
        
        reward_class = cls._reward_registry[reward_type]
        
        try:
            # 过滤出该奖励函数支持的参数
            filtered_kwargs = cls._filter_kwargs(reward_class, kwargs)
            reward_instance = reward_class(**filtered_kwargs)
            
            logging.info(f"成功创建奖励函数: {reward_class.__name__} "
                        f"参数: {filtered_kwargs}")
            
            return reward_instance
            
        except Exception as e:
            logging.error(f"创建奖励函数失败: {reward_class.__name__}, 错误: {e}")
            raise
    
    @classmethod
    def _filter_kwargs(cls, reward_class: Type[BaseRewardScheme], 
                      kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        过滤参数，只保留目标类构造函数支持的参数
        
        Args:
            reward_class: 目标奖励函数类
            kwargs: 原始参数字典
            
        Returns:
            Dict[str, Any]: 过滤后的参数字典
        """
        import inspect
        
        try:
            # 获取构造函数签名
            sig = inspect.signature(reward_class.__init__)
            valid_params = set(sig.parameters.keys()) - {'self'}
            
            # 过滤参数
            filtered = {k: v for k, v in kwargs.items() if k in valid_params}
            
            # 记录被过滤掉的参数
            ignored = set(kwargs.keys()) - valid_params
            if ignored:
                logging.debug(f"忽略了不支持的参数: {ignored}")
            
            return filtered
            
        except Exception as e:
            logging.warning(f"参数过滤失败，使用原始参数: {e}")
            return kwargs
    
    @classmethod
    def register_reward_function(cls, 
                                name: str, 
                                reward_class: Type[BaseRewardScheme]) -> None:
        """
        注册新的奖励函数类型
        
        Args:
            name: 奖励函数名称
            reward_class: 奖励函数类
            
        Raises:
            ValueError: 当奖励函数类不继承自BaseRewardScheme时
        """
        if not issubclass(reward_class, BaseRewardScheme):
            raise ValueError(f"奖励函数类必须继承自BaseRewardScheme: {reward_class}")
        
        name = name.lower().strip()
        if name in cls._reward_registry:
            logging.warning(f"覆盖已存在的奖励函数类型: {name}")
        
        cls._reward_registry[name] = reward_class
        logging.info(f"注册奖励函数成功: {name} -> {reward_class.__name__}")
    
    @classmethod
    def get_available_types(cls) -> Dict[str, Type[BaseRewardScheme]]:
        """
        获取所有可用的奖励函数类型
        
        Returns:
            Dict[str, Type[BaseRewardScheme]]: 奖励函数类型映射
        """
        return cls._reward_registry.copy()
    
    @classmethod
    def get_reward_info(cls, reward_type: str = None) -> Dict[str, Any]:
        """
        获取奖励函数信息
        
        Args:
            reward_type: 奖励函数类型，None表示获取所有类型的信息
            
        Returns:
            Dict[str, Any]: 奖励函数信息
        """
        if reward_type is None:
            # 返回所有奖励函数的信息
            info = {}
            processed_classes = set()
            
            for name, reward_class in cls._reward_registry.items():
                if reward_class not in processed_classes:
                    # 创建临时实例以获取信息
                    temp_instance = reward_class()
                    info[name] = temp_instance.get_reward_info()
                    processed_classes.add(reward_class)
            
            return info
        
        else:
            # 返回指定奖励函数的信息
            reward_type = reward_type.lower().strip()
            if reward_type not in cls._reward_registry:
                raise ValueError(f"未知的奖励函数类型: '{reward_type}'")
            
            reward_class = cls._reward_registry[reward_type]
            return reward_class.get_reward_info()
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> BaseRewardScheme:
        """
        从配置字典创建奖励函数
        
        Args:
            config: 配置字典，应包含 'type' 和其他参数
            
        Returns:
            BaseRewardScheme: 奖励函数实例
            
        Example:
            config = {
                'type': 'risk_adjusted',
                'risk_free_rate': 0.02,
                'window_size': 50,
                'initial_balance': 10000.0
            }
            reward_fn = RewardFactory.create_from_config(config)
        """
        if not isinstance(config, dict):
            raise ValueError("配置必须是字典类型")
        
        if 'type' not in config:
            raise ValueError("配置字典必须包含 'type' 字段")
        
        reward_type = config.pop('type')  # 移除type字段
        
        return cls.create_reward_function(reward_type, **config)


def create_reward_function(reward_type: str = 'risk_adjusted', 
                          **kwargs) -> BaseRewardScheme:
    """
    便捷函数：创建奖励函数实例
    
    Args:
        reward_type: 奖励函数类型
        **kwargs: 传递给奖励函数构造函数的参数
        
    Returns:
        BaseRewardScheme: 奖励函数实例
        
    Example:
        # 创建风险调整奖励函数
        reward_fn = create_reward_function('risk_adjusted', 
                                         risk_free_rate=0.02,
                                         window_size=50)
        
        # 创建简单收益奖励函数
        reward_fn = create_reward_function('simple_return',
                                         step_weight=1.0,
                                         total_weight=0.1)
        
        # 创建盈亏比奖励函数
        reward_fn = create_reward_function('profit_loss',
                                         profit_bonus=2.0,
                                         loss_penalty=1.5)
        
        # 创建多指标综合奖励函数
        weights = {
            'return': 0.4,
            'risk': 0.2,
            'stability': 0.15,
            'efficiency': 0.15,
            'drawdown': 0.1
        }
        reward_fn = create_reward_function('diversified', weights=weights)
    """
    return RewardFactory.create_reward_function(reward_type, **kwargs)


def get_reward_function_info(reward_type: str = None) -> Dict[str, Any]:
    """
    便捷函数：获取奖励函数信息
    
    Args:
        reward_type: 奖励函数类型，None表示获取所有类型的信息
        
    Returns:
        Dict[str, Any]: 奖励函数信息
        
    Example:
        # 获取所有奖励函数信息
        all_info = get_reward_function_info()
        
        # 获取特定奖励函数信息
        info = get_reward_function_info('risk_adjusted')
        print(info['description'])
        print(info['parameters'])
    """
    return RewardFactory.get_reward_info(reward_type)


def list_available_reward_types() -> List[str]:
    """
    便捷函数：列出所有可用的奖励函数类型
    
    Returns:
        List[str]: 可用的奖励函数类型列表
        
    Example:
        types = list_available_reward_types()
        print("可用的奖励函数类型:")
        for t in types:
            print(f"  - {t}")
    """
    return list(RewardFactory.get_available_types().keys())


# 为了向后兼容，提供原始配置映射
REWARD_TYPE_MAPPING = {
    'risk_adjusted': 'risk_adjusted',
    'simple': 'simple_return', 
    'pnl': 'profit_loss',
    'comprehensive': 'diversified',
    
    # 保持原有的别名
    'default': 'risk_adjusted',
    'sharpe': 'risk_adjusted',
    'basic': 'simple_return',
    'multi': 'diversified',
}


if __name__ == "__main__":
    # 测试代码
    print("奖励函数工厂测试")
    print("=" * 50)
    
    # 列出所有可用类型
    print("可用的奖励函数类型:")
    for reward_type in list_available_reward_types():
        print(f"  - {reward_type}")
    
    print("\n" + "=" * 50)
    
    # 测试创建不同类型的奖励函数
    test_types = ['risk_adjusted', 'simple_return', 'profit_loss', 'diversified', 'optimized_forex_reward']
    
    for reward_type in test_types:
        try:
            reward_fn = create_reward_function(reward_type, initial_balance=10000.0)
            info = reward_fn.get_reward_info()
            print(f"\n✓ {reward_type}: {info['name']}")
            print(f"  描述: {info['description']}")
            print(f"  类别: {info['category']}")
            
        except Exception as e:
            print(f"\n✗ {reward_type}: 创建失败 - {e}")
    
    print("\n" + "=" * 50)
    print("奖励函数工厂测试完成")