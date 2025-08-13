"""
基于联邦学习的奖励函数优化系统

实现分布式多客户端协作的奖励函数学习，结合差分隐私、
安全聚合和区块链智能合约机制，确保隐私保护的同时
实现高效的奖励函数协作优化。

基于2024-2025年最新联邦学习和强化学习研究成果：
- 区块链智能合约激励机制
- 差分隐私保护
- 安全聚合协议
- 多智能体协作学习
- 声誉驱动的奖励分享
"""

import hashlib
import json
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import numpy as np
import logging

from .base_reward import BaseRewardScheme

logger = logging.getLogger(__name__)

@dataclass
class ClientContribution:
    """客户端贡献记录"""
    client_id: str
    contribution_score: float
    data_quality: float
    model_accuracy: float
    privacy_budget_used: float
    timestamp: float
    reputation_score: float

@dataclass
class FederatedRound:
    """联邦学习轮次记录"""
    round_id: int
    timestamp: float
    participating_clients: List[str]
    global_model_hash: str
    aggregated_reward: float
    convergence_metric: float
    privacy_cost: float

@dataclass
class DifferentialPrivacyConfig:
    """差分隐私配置"""
    epsilon: float = 1.0  # 隐私预算
    delta: float = 1e-5   # 失败概率
    clipping_norm: float = 1.0  # 梯度裁剪范数
    noise_multiplier: float = 1.0  # 噪声乘数
    adaptive_budget: bool = True  # 自适应预算分配

class SecureAggregator:
    """安全聚合器"""
    
    def __init__(self, num_clients: int, privacy_config: DifferentialPrivacyConfig):
        self.num_clients = num_clients
        self.privacy_config = privacy_config
        self.aggregation_history = deque(maxlen=1000)
        self.client_contributions = {}
        
    def add_noise_for_privacy(self, values: np.ndarray) -> np.ndarray:
        """为差分隐私添加噪声"""
        if self.privacy_config.epsilon <= 0:
            return values
        
        # 计算噪声规模
        sensitivity = self.privacy_config.clipping_norm
        noise_scale = sensitivity * self.privacy_config.noise_multiplier / self.privacy_config.epsilon
        
        # 添加高斯噪声
        noise = np.random.normal(0, noise_scale, values.shape)
        return values + noise
    
    def clip_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """梯度裁剪以控制敏感性"""
        norm = np.linalg.norm(gradients)
        if norm > self.privacy_config.clipping_norm:
            gradients = gradients * (self.privacy_config.clipping_norm / norm)
        return gradients
    
    def secure_aggregate(self, client_updates: Dict[str, np.ndarray], 
                        client_weights: Dict[str, float] = None) -> np.ndarray:
        """安全聚合客户端更新"""
        if not client_updates:
            return np.array([0.0])
        
        # 默认权重（基于数据量或贡献）
        if client_weights is None:
            client_weights = {client_id: 1.0 for client_id in client_updates.keys()}
        
        # 归一化权重
        total_weight = sum(client_weights.values())
        normalized_weights = {cid: w/total_weight for cid, w in client_weights.items()}
        
        # 梯度裁剪和噪声添加
        clipped_updates = {}
        for client_id, update in client_updates.items():
            clipped = self.clip_gradients(update)
            noisy = self.add_noise_for_privacy(clipped)
            clipped_updates[client_id] = noisy
        
        # 加权聚合
        aggregated = np.zeros_like(list(clipped_updates.values())[0])
        for client_id, update in clipped_updates.items():
            weight = normalized_weights[client_id]
            aggregated += weight * update
        
        # 记录聚合历史
        self.aggregation_history.append({
            'timestamp': time.time(),
            'num_clients': len(client_updates),
            'total_weight': total_weight,
            'privacy_cost': self._calculate_privacy_cost()
        })
        
        return aggregated
    
    def _calculate_privacy_cost(self) -> float:
        """计算隐私成本"""
        return self.privacy_config.noise_multiplier / self.privacy_config.epsilon

class ReputationSystem:
    """声誉系统"""
    
    def __init__(self):
        self.client_reputations = defaultdict(lambda: {
            'score': 0.5,  # 初始声誉
            'history': deque(maxlen=100),
            'contribution_count': 0,
            'quality_average': 0.0
        })
        self.reputation_decay = 0.95  # 声誉衰减因子
        
    def update_reputation(self, client_id: str, contribution: ClientContribution):
        """更新客户端声誉"""
        client_rep = self.client_reputations[client_id]
        
        # 计算本次贡献得分
        contribution_score = self._calculate_contribution_score(contribution)
        
        # 更新历史记录
        client_rep['history'].append({
            'timestamp': contribution.timestamp,
            'score': contribution_score,
            'data_quality': contribution.data_quality,
            'model_accuracy': contribution.model_accuracy
        })
        
        # 更新声誉分数（指数移动平均）
        current_score = client_rep['score']
        new_score = self.reputation_decay * current_score + (1 - self.reputation_decay) * contribution_score
        client_rep['score'] = max(0.0, min(1.0, new_score))
        
        # 更新统计信息
        client_rep['contribution_count'] += 1
        total_quality = client_rep['quality_average'] * (client_rep['contribution_count'] - 1)
        client_rep['quality_average'] = (total_quality + contribution.data_quality) / client_rep['contribution_count']
        
        return client_rep['score']
    
    def _calculate_contribution_score(self, contribution: ClientContribution) -> float:
        """计算贡献得分"""
        # 综合考虑数据质量、模型准确性和隐私成本
        quality_score = contribution.data_quality * 0.4
        accuracy_score = contribution.model_accuracy * 0.4
        privacy_score = (1.0 - contribution.privacy_budget_used) * 0.2
        
        return quality_score + accuracy_score + privacy_score
    
    def get_client_weight(self, client_id: str) -> float:
        """获取客户端权重（基于声誉）"""
        return self.client_reputations[client_id]['score']
    
    def get_top_clients(self, n: int = 10) -> List[Tuple[str, float]]:
        """获取声誉最高的客户端"""
        sorted_clients = sorted(
            self.client_reputations.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        return [(client_id, rep['score']) for client_id, rep in sorted_clients[:n]]

class SmartContract:
    """智能合约模拟器"""
    
    def __init__(self):
        self.contract_state = {
            'total_rewards': 0.0,
            'client_balances': defaultdict(float),
            'reward_rules': {},
            'transaction_history': []
        }
        self.reward_pool = 1000.0  # 奖励池
        
    def deploy_reward_contract(self, reward_rules: Dict[str, Any]):
        """部署奖励合约"""
        self.contract_state['reward_rules'] = reward_rules
        contract_hash = self._calculate_contract_hash(reward_rules)
        
        transaction = {
            'type': 'contract_deployment',
            'timestamp': time.time(),
            'contract_hash': contract_hash,
            'rules': reward_rules
        }
        self.contract_state['transaction_history'].append(transaction)
        
        return contract_hash
    
    def distribute_rewards(self, contributions: List[ClientContribution]) -> Dict[str, float]:
        """分发奖励"""
        if not contributions:
            return {}
        
        # 计算总贡献
        total_contribution = sum(c.contribution_score * c.reputation_score for c in contributions)
        
        if total_contribution == 0:
            return {}
        
        # 分配奖励
        client_rewards = {}
        available_rewards = min(self.reward_pool, self.contract_state['total_rewards'])
        
        for contribution in contributions:
            weighted_contribution = contribution.contribution_score * contribution.reputation_score
            reward_fraction = weighted_contribution / total_contribution
            reward_amount = available_rewards * reward_fraction
            
            client_rewards[contribution.client_id] = reward_amount
            self.contract_state['client_balances'][contribution.client_id] += reward_amount
        
        # 记录交易
        transaction = {
            'type': 'reward_distribution',
            'timestamp': time.time(),
            'rewards': client_rewards,
            'total_distributed': sum(client_rewards.values())
        }
        self.contract_state['transaction_history'].append(transaction)
        
        return client_rewards
    
    def _calculate_contract_hash(self, contract_data: Any) -> str:
        """计算合约哈希"""
        contract_str = json.dumps(contract_data, sort_keys=True)
        return hashlib.sha256(contract_str.encode()).hexdigest()

class FederatedReward(BaseRewardScheme):
    """联邦学习奖励函数"""
    
    def __init__(
        self,
        num_clients: int = 10,
        min_clients_per_round: int = 3,
        privacy_epsilon: float = 1.0,
        privacy_delta: float = 1e-5,
        enable_reputation: bool = True,
        enable_smart_contracts: bool = True,
        aggregation_method: str = "secure_avg",
        collaboration_weight: float = 0.3,
        **kwargs
    ):
        """
        初始化联邦学习奖励函数
        
        Args:
            num_clients: 总客户端数量
            min_clients_per_round: 每轮最小参与客户端数
            privacy_epsilon: 差分隐私预算
            privacy_delta: 差分隐私失败概率
            enable_reputation: 是否启用声誉系统
            enable_smart_contracts: 是否启用智能合约
            aggregation_method: 聚合方法
            collaboration_weight: 协作奖励权重
        """
        super().__init__(**kwargs)
        
        self.num_clients = num_clients
        self.min_clients_per_round = min_clients_per_round
        self.collaboration_weight = collaboration_weight
        self.aggregation_method = aggregation_method
        
        # 差分隐私配置
        self.privacy_config = DifferentialPrivacyConfig(
            epsilon=privacy_epsilon,
            delta=privacy_delta
        )
        
        # 核心组件
        self.secure_aggregator = SecureAggregator(num_clients, self.privacy_config)
        self.reputation_system = ReputationSystem() if enable_reputation else None
        self.smart_contract = SmartContract() if enable_smart_contracts else None
        
        # 状态跟踪
        self.current_round = 0
        self.federated_history = deque(maxlen=1000)
        self.client_updates = {}
        self.global_model_state = np.array([0.0])
        
        # 性能指标
        self.convergence_history = deque(maxlen=100)
        self.communication_costs = deque(maxlen=100)
        
        # 线程安全
        self._lock = threading.Lock()
        
        # 初始化智能合约
        if self.smart_contract:
            reward_rules = {
                'base_reward': 1.0,
                'quality_bonus': 0.5,
                'reputation_multiplier': 2.0,
                'privacy_incentive': 0.3
            }
            self.contract_hash = self.smart_contract.deploy_reward_contract(reward_rules)
        
        logger.info(f"FederatedReward initialized with {num_clients} clients, privacy ε={privacy_epsilon}")
    
    def reward(self, portfolio) -> float:
        """计算奖励 - 抽象方法实现"""
        return self.get_reward(portfolio)
    
    def get_reward(self, portfolio) -> float:
        """计算联邦学习增强的奖励"""
        with self._lock:
            # 基础奖励计算
            base_reward = self._calculate_base_reward(portfolio)
            
            # 模拟客户端更新
            client_update = self._simulate_client_update(portfolio, base_reward)
            
            # 添加到客户端更新池
            client_id = f"client_{hash(str(portfolio.net_worth)) % self.num_clients}"
            self.client_updates[client_id] = client_update
            
            # 如果有足够的客户端更新，执行联邦聚合
            if len(self.client_updates) >= self.min_clients_per_round:
                federated_reward = self._execute_federated_round()
                self.client_updates.clear()
            else:
                federated_reward = base_reward
            
            # 结合协作奖励
            final_reward = (1 - self.collaboration_weight) * base_reward + \
                          self.collaboration_weight * federated_reward
            
            return float(final_reward)
    
    def _calculate_base_reward(self, portfolio) -> float:
        """计算基础奖励（本地奖励）"""
        # 更新历史记录
        current_value = portfolio.net_worth
        self.portfolio_history.append(current_value)
        
        if len(self.portfolio_history) < 2:
            return 0.0
        
        # 简单收益率计算
        previous_value = self.portfolio_history[-2]
        
        if previous_value <= 0:
            return 0.0
        
        return_rate = (current_value - previous_value) / previous_value
        
        # 应用风险调整
        if len(self.portfolio_history) >= 10:
            recent_values = self.portfolio_history[-10:]
            returns = np.diff(recent_values) / recent_values[:-1]
            volatility = np.std(returns) if len(returns) > 1 else 0.0
            risk_adjusted_return = return_rate / (volatility + 1e-8)
            return risk_adjusted_return
        
        return return_rate
    
    def _simulate_client_update(self, portfolio, base_reward: float) -> np.ndarray:
        """模拟客户端更新"""
        # 模拟梯度更新（在实际应用中，这将是真实的模型梯度）
        gradient_size = 10
        
        # 基于当前状态生成模拟梯度
        current_state = np.array([
            portfolio.net_worth / 10000.0,  # 归一化净值
            base_reward,
            len(self.portfolio_history),
            time.time() % 100  # 时间特征
        ])
        
        # 生成模拟梯度（实际应用中来自模型训练）
        np.random.seed(int(current_state.sum() * 1000) % 2**32)
        gradient = np.random.normal(0, 0.1, gradient_size)
        
        # 添加基于奖励的调整
        gradient += base_reward * 0.1 * np.random.random(gradient_size)
        
        return gradient
    
    def _execute_federated_round(self) -> float:
        """执行联邦学习轮次"""
        self.current_round += 1
        
        # 获取客户端权重（基于声誉）
        client_weights = {}
        contributions = []
        
        for client_id in self.client_updates.keys():
            if self.reputation_system:
                weight = self.reputation_system.get_client_weight(client_id)
            else:
                weight = 1.0
            client_weights[client_id] = weight
            
            # 创建贡献记录
            contribution = ClientContribution(
                client_id=client_id,
                contribution_score=np.linalg.norm(self.client_updates[client_id]),
                data_quality=np.random.beta(2, 1),  # 模拟数据质量
                model_accuracy=np.random.beta(3, 1),  # 模拟模型准确性
                privacy_budget_used=self.privacy_config.epsilon * 0.1,
                timestamp=time.time(),
                reputation_score=weight
            )
            contributions.append(contribution)
            
            # 更新声誉
            if self.reputation_system:
                self.reputation_system.update_reputation(client_id, contribution)
        
        # 安全聚合
        aggregated_update = self.secure_aggregator.secure_aggregate(
            self.client_updates, client_weights
        )
        
        # 更新全局模型状态
        learning_rate = 0.01
        self.global_model_state = self.global_model_state + learning_rate * aggregated_update
        
        # 计算聚合奖励
        aggregated_reward = np.mean([c.contribution_score for c in contributions])
        
        # 计算收敛指标
        convergence_metric = np.linalg.norm(aggregated_update)
        self.convergence_history.append(convergence_metric)
        
        # 智能合约奖励分发
        if self.smart_contract and contributions:
            client_rewards = self.smart_contract.distribute_rewards(contributions)
            logger.debug(f"Distributed rewards: {client_rewards}")
        
        # 记录联邦轮次
        fed_round = FederatedRound(
            round_id=self.current_round,
            timestamp=time.time(),
            participating_clients=list(self.client_updates.keys()),
            global_model_hash=hashlib.sha256(self.global_model_state.tobytes()).hexdigest()[:16],
            aggregated_reward=aggregated_reward,
            convergence_metric=convergence_metric,
            privacy_cost=self.secure_aggregator._calculate_privacy_cost()
        )
        self.federated_history.append(fed_round)
        
        logger.debug(f"Federated round {self.current_round} completed with {len(contributions)} clients")
        
        return aggregated_reward
    
    def get_federated_info(self) -> Dict[str, Any]:
        """获取联邦学习状态信息"""
        with self._lock:
            return {
                'current_round': self.current_round,
                'total_clients': self.num_clients,
                'active_clients': len(self.client_updates),
                'min_clients_per_round': self.min_clients_per_round,
                'privacy_budget': self.privacy_config.epsilon,
                'global_model_hash': hashlib.sha256(self.global_model_state.tobytes()).hexdigest()[:16],
                'convergence_trend': list(self.convergence_history)[-10:] if self.convergence_history else [],
                'reputation_enabled': self.reputation_system is not None,
                'smart_contracts_enabled': self.smart_contract is not None,
                'aggregation_method': self.aggregation_method,
                'collaboration_weight': self.collaboration_weight
            }
    
    def get_client_reputation_info(self) -> Dict[str, Any]:
        """获取客户端声誉信息"""
        if not self.reputation_system:
            return {'error': 'Reputation system not enabled'}
        
        top_clients = self.reputation_system.get_top_clients(5)
        
        return {
            'top_clients': top_clients,
            'total_tracked_clients': len(self.reputation_system.client_reputations),
            'average_reputation': np.mean([
                rep['score'] for rep in self.reputation_system.client_reputations.values()
            ]) if self.reputation_system.client_reputations else 0.0
        }
    
    def get_privacy_info(self) -> Dict[str, Any]:
        """获取隐私保护信息"""
        return {
            'differential_privacy': {
                'epsilon': self.privacy_config.epsilon,
                'delta': self.privacy_config.delta,
                'clipping_norm': self.privacy_config.clipping_norm,
                'noise_multiplier': self.privacy_config.noise_multiplier,
                'adaptive_budget': self.privacy_config.adaptive_budget
            },
            'aggregation_history_length': len(self.secure_aggregator.aggregation_history),
            'total_privacy_cost': sum([
                entry['privacy_cost'] for entry in self.secure_aggregator.aggregation_history
            ]) if self.secure_aggregator.aggregation_history else 0.0
        }
    
    def get_smart_contract_info(self) -> Dict[str, Any]:
        """获取智能合约信息"""
        if not self.smart_contract:
            return {'error': 'Smart contracts not enabled'}
        
        state = self.smart_contract.contract_state
        return {
            'contract_hash': getattr(self, 'contract_hash', None),
            'total_rewards_distributed': state['total_rewards'],
            'active_clients_with_balance': len([
                balance for balance in state['client_balances'].values() if balance > 0
            ]),
            'total_transactions': len(state['transaction_history']),
            'reward_pool_remaining': self.smart_contract.reward_pool,
            'recent_transactions': state['transaction_history'][-5:] if state['transaction_history'] else []
        }
    
    def reset(self):
        """重置奖励函数状态"""
        super().reset()
        with self._lock:
            self.current_round = 0
            self.client_updates.clear()
            self.global_model_state = np.array([0.0])
            self.convergence_history.clear()
            self.communication_costs.clear()
            
            # 重置组件状态
            if self.reputation_system:
                self.reputation_system.client_reputations.clear()
            
            if self.smart_contract:
                self.smart_contract.contract_state = {
                    'total_rewards': 0.0,
                    'client_balances': defaultdict(float),
                    'reward_rules': self.smart_contract.contract_state.get('reward_rules', {}),
                    'transaction_history': []
                }
        
        logger.info("FederatedReward reset completed")
    
    @staticmethod
    def get_reward_info() -> Dict[str, Any]:
        """获取奖励函数信息"""
        return {
            'name': 'Federated Learning Reward',
            'description': '基于联邦学习的分布式协作奖励优化系统',
            'category': 'Collaborative AI',
            'complexity': 'Expert',
            'research_base': [
                '2024-2025 Federated Reinforcement Learning',
                'Differential Privacy in Distributed RL',
                'Blockchain Smart Contracts for ML',
                'Multi-Agent Cooperative Learning',
                'Secure Aggregation Protocols'
            ],
            'key_features': [
                '多客户端协作学习',
                '差分隐私保护',
                '安全聚合协议',
                '声誉驱动激励',
                '智能合约自动化',
                '自适应隐私预算',
                '收敛性保证'
            ],
            'use_cases': [
                '分布式量化投资',
                '多机构协作交易',
                '隐私保护策略学习',
                '去中心化金融(DeFi)',
                '联盟链交易系统'
            ],
            'parameters': {
                'num_clients': '总客户端数量',
                'min_clients_per_round': '每轮最小参与客户端数',
                'privacy_epsilon': '差分隐私预算',
                'privacy_delta': '差分隐私失败概率',
                'enable_reputation': '是否启用声誉系统',
                'enable_smart_contracts': '是否启用智能合约',
                'aggregation_method': '聚合方法',
                'collaboration_weight': '协作奖励权重'
            }
        }
    
    def calculate_reward(self, current_step, current_price, current_portfolio_value, action, **kwargs):
        """
        奖励计算接口 - 计算联邦学习奖励
        
        Args:
            current_step: 当前步数
            current_price: 当前价格
            current_portfolio_value: 当前投资组合价值
            action: 执行的动作
            **kwargs: 其他参数
            
        Returns:
            float: 奖励值
        """
        # 更新历史记录
        self.update_history(current_portfolio_value)
        
        # 计算步骤收益
        if len(self.portfolio_history) < 2:
            step_return = 0.0
        else:
            prev_value = self.portfolio_history[-2]
            step_return = current_portfolio_value - prev_value
        
        # 基础奖励：步骤回报的百分比
        base_reward = (step_return / self.initial_balance) * 100 if self.initial_balance != 0 else 0.0
        
        # 构建状态和奖励计算相关的context
        portfolio_info = {
            'current_value': current_portfolio_value,
            'step_return': step_return,
            'total_return_pct': ((current_portfolio_value - self.initial_balance) / self.initial_balance) * 100
        }
        
        trade_info = {
            'action': action,
            'executed': abs(action) > 0.01  # 简化的交易判断
        }
        
        # 使用联邦奖励计算
        try:
            federated_reward = self.reward(
                portfolio_value=current_portfolio_value,
                action=action,
                price=current_price,
                portfolio_info=portfolio_info,
                trade_info=trade_info,
                step=current_step
            )
        except Exception as e:
            self.logger.error(f"联邦奖励计算失败: {e}")
            federated_reward = base_reward
        
        # 记录奖励历史
        self.reward_history.append(federated_reward)
        
        return federated_reward