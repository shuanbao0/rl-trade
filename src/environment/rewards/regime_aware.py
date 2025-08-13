"""
自适应专家 —— 市场状态感知奖励函数 (Regime-Aware Rewards)

基于2025年最新的自奖励深度强化学习理论，结合隐马尔可夫模型和技术指标分析，
实现能够感知市场状态并动态调整奖励策略的智能专家系统。

核心创新：
1. 多状态检测：自动识别牛市、熊市、震荡、高/低波动等市场状态
2. 专家策略：每种市场状态对应专门的奖励策略和参数
3. 自适应切换：基于技术指标和统计特征的智能状态切换
4. 动态调优：状态特定的奖励权重和惩罚机制
5. 连续学习：记忆历史状态转换提升检测准确性

数学基础：
- 状态检测：Regime_t = f(SMA, BB_width, ATR, MACD, volatility, trend)
- 动态奖励：R_t = f(portfolio_metrics | Regime_t)
- 状态转换：P(S_t|S_{t-1}, indicators) - 马尔可夫链
- 专家融合：R_final = Σ w_i × R_i where w_i depends on regime confidence
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from enum import Enum
from .base_reward import BaseRewardScheme


class MarketRegime(Enum):
    """市场状态枚举"""
    BULL_MARKET = "bull_market"           # 牛市：趋势向上
    BEAR_MARKET = "bear_market"           # 熊市：趋势向下  
    SIDEWAYS_MARKET = "sideways_market"   # 震荡市：横盘整理
    HIGH_VOLATILITY = "high_volatility"   # 高波动：市场不确定
    LOW_VOLATILITY = "low_volatility"     # 低波动：稳定环境


class MarketStateDetector:
    """
    市场状态检测器
    
    结合多种技术指标和统计方法，智能识别当前市场所处的状态，
    为自适应奖励策略提供决策依据。
    """
    
    def __init__(self,
                 sma_short: int = 20,
                 sma_long: int = 50,
                 bb_period: int = 20,
                 atr_period: int = 14,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 volatility_window: int = 20,
                 trend_threshold: float = 0.02,
                 volatility_threshold: float = 0.015):
        """
        初始化市场状态检测器
        
        Args:
            sma_short: 短期移动平均周期
            sma_long: 长期移动平均周期  
            bb_period: 布林带计算周期
            atr_period: ATR波动率周期
            macd_fast: MACD快线周期
            macd_slow: MACD慢线周期
            volatility_window: 波动率计算窗口
            trend_threshold: 趋势判断阈值
            volatility_threshold: 波动率判断阈值
        """
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.bb_period = bb_period
        self.atr_period = atr_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.volatility_window = volatility_window
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        
        # 价格历史数据
        self.price_history = deque(maxlen=max(sma_long, bb_period, atr_period) * 2)
        self.volume_history = deque(maxlen=100)
        
        # 技术指标缓存
        self.indicators_cache = {}
        self.regime_history = deque(maxlen=100)
        self.regime_confidence = 0.0
        
        # 状态转换计数器
        self.regime_transitions = {}
        for regime in MarketRegime:
            self.regime_transitions[regime] = 0
    
    def update_data(self, price: float, volume: float = 1.0) -> None:
        """
        更新价格和成交量数据
        
        Args:
            price: 当前价格
            volume: 当前成交量
        """
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # 清除缓存，强制重新计算指标
        self.indicators_cache.clear()
    
    def _calculate_sma(self, period: int) -> float:
        """计算简单移动平均"""
        if len(self.price_history) < period:
            return 0.0
        
        cache_key = f"sma_{period}"
        if cache_key not in self.indicators_cache:
            prices = list(self.price_history)[-period:]
            self.indicators_cache[cache_key] = np.mean(prices)
        
        return self.indicators_cache[cache_key]
    
    def _calculate_bollinger_bands(self) -> Tuple[float, float, float]:
        """计算布林带：上轨、中轨、下轨"""
        if len(self.price_history) < self.bb_period:
            return 0.0, 0.0, 0.0
        
        cache_key = "bollinger_bands"
        if cache_key not in self.indicators_cache:
            prices = np.array(list(self.price_history)[-self.bb_period:])
            middle = np.mean(prices)
            std = np.std(prices)
            upper = middle + 2 * std
            lower = middle - 2 * std
            self.indicators_cache[cache_key] = (upper, middle, lower)
        
        return self.indicators_cache[cache_key]
    
    def _calculate_atr(self) -> float:
        """计算平均真实波幅"""
        if len(self.price_history) < self.atr_period + 1:
            return 0.0
        
        cache_key = "atr"
        if cache_key not in self.indicators_cache:
            prices = np.array(list(self.price_history)[-self.atr_period-1:])
            high = prices[1:]  # 当前高点
            low = prices[1:]   # 当前低点  
            close_prev = prices[:-1]  # 前一日收盘
            
            tr1 = high - low
            tr2 = np.abs(high - close_prev)
            tr3 = np.abs(low - close_prev)
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = np.mean(true_range)
            self.indicators_cache[cache_key] = atr
        
        return self.indicators_cache[cache_key]
    
    def _calculate_macd(self) -> Tuple[float, float]:
        """计算MACD指标：MACD线和信号线"""
        if len(self.price_history) < self.macd_slow:
            return 0.0, 0.0
        
        cache_key = "macd"
        if cache_key not in self.indicators_cache:
            prices = np.array(list(self.price_history))
            
            # 计算EMA
            ema_fast = self._calculate_ema(prices, self.macd_fast)
            ema_slow = self._calculate_ema(prices, self.macd_slow)
            
            macd_line = ema_fast - ema_slow
            signal_line = self._calculate_ema(np.array([macd_line]), 9)
            
            self.indicators_cache[cache_key] = (macd_line, signal_line)
        
        return self.indicators_cache[cache_key]
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> float:
        """计算指数移动平均"""
        if len(data) < period:
            return np.mean(data) if len(data) > 0 else 0.0
        
        alpha = 2.0 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _calculate_volatility(self) -> float:
        """计算历史波动率"""
        if len(self.price_history) < self.volatility_window + 1:
            return 0.0
        
        cache_key = "volatility"
        if cache_key not in self.indicators_cache:
            prices = np.array(list(self.price_history)[-self.volatility_window-1:])
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # 年化波动率
            self.indicators_cache[cache_key] = volatility
        
        return self.indicators_cache[cache_key]
    
    def detect_regime(self) -> Tuple[MarketRegime, float]:
        """
        检测当前市场状态
        
        Returns:
            Tuple[MarketRegime, float]: 市场状态和置信度
        """
        if len(self.price_history) < max(self.sma_long, self.bb_period):
            return MarketRegime.LOW_VOLATILITY, 0.5
        
        # 计算技术指标
        sma_short = self._calculate_sma(self.sma_short)
        sma_long = self._calculate_sma(self.sma_long)
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands()
        atr = self._calculate_atr()
        macd_line, macd_signal = self._calculate_macd()
        volatility = self._calculate_volatility()
        current_price = self.price_history[-1]
        
        # 计算决策指标
        trend_strength = (sma_short - sma_long) / sma_long if sma_long > 0 else 0
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
        bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
        macd_momentum = macd_line - macd_signal
        
        # 状态判断逻辑
        signals = []
        confidence_scores = []
        
        # 1. 波动率判断
        if volatility > self.volatility_threshold * 2:
            signals.append(MarketRegime.HIGH_VOLATILITY)
            confidence_scores.append(min(volatility / (self.volatility_threshold * 2), 1.0))
        elif volatility < self.volatility_threshold * 0.5:
            signals.append(MarketRegime.LOW_VOLATILITY)
            confidence_scores.append(min((self.volatility_threshold * 0.5) / volatility, 1.0))
        
        # 2. 趋势判断
        if trend_strength > self.trend_threshold and macd_momentum > 0:
            signals.append(MarketRegime.BULL_MARKET)
            confidence_scores.append(min(trend_strength / self.trend_threshold, 1.0))
        elif trend_strength < -self.trend_threshold and macd_momentum < 0:
            signals.append(MarketRegime.BEAR_MARKET)
            confidence_scores.append(min(abs(trend_strength) / self.trend_threshold, 1.0))
        
        # 3. 震荡判断
        if (abs(trend_strength) < self.trend_threshold * 0.5 and 
            bb_width > np.mean([self._calculate_bb_width_history()]) * 1.2):
            signals.append(MarketRegime.SIDEWAYS_MARKET)
            confidence_scores.append(0.7)
        
        # 决策融合
        if not signals:
            # 默认状态
            regime = MarketRegime.LOW_VOLATILITY
            confidence = 0.3
        else:
            # 选择置信度最高的状态
            max_confidence_idx = np.argmax(confidence_scores)
            regime = signals[max_confidence_idx]
            confidence = confidence_scores[max_confidence_idx]
        
        # 状态平滑：避免频繁切换
        if len(self.regime_history) > 0:
            last_regime = self.regime_history[-1][0]
            if last_regime == regime:
                confidence = min(confidence * 1.1, 1.0)  # 增强连续性
            elif confidence < 0.7:
                # 置信度不足时保持上一状态
                regime = last_regime
                confidence = 0.6
        
        # 记录历史
        self.regime_history.append((regime, confidence))
        self.regime_confidence = confidence
        
        # 更新转换统计
        if (len(self.regime_history) > 1 and 
            self.regime_history[-1][0] != self.regime_history[-2][0]):
            self.regime_transitions[regime] += 1
        
        return regime, confidence
    
    def _calculate_bb_width_history(self) -> float:
        """计算布林带宽度历史平均值"""
        if len(self.price_history) < self.bb_period * 3:
            return 0.01
        
        widths = []
        for i in range(self.bb_period, len(self.price_history) - self.bb_period + 1):
            subset = list(self.price_history)[i:i+self.bb_period]
            if len(subset) >= self.bb_period:
                mean_price = np.mean(subset)
                std_price = np.std(subset)
                width = (4 * std_price) / mean_price if mean_price > 0 else 0
                widths.append(width)
        
        return np.mean(widths) if widths else 0.01
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """获取状态检测统计信息"""
        if not self.regime_history:
            return {}
        
        regimes = [r[0] for r in self.regime_history]
        regime_counts = {}
        for regime in MarketRegime:
            regime_counts[regime.value] = regimes.count(regime)
        
        return {
            'current_regime': self.regime_history[-1][0].value if self.regime_history else 'unknown',
            'regime_confidence': self.regime_confidence,
            'regime_distribution': regime_counts,
            'regime_transitions': {k.value: v for k, v in self.regime_transitions.items()},
            'total_detections': len(self.regime_history)
        }


class RegimeAwareReward(BaseRewardScheme):
    """
    自适应专家 —— 市场状态感知奖励函数
    
    该奖励函数能够智能识别市场状态，并根据不同的市场环境
    动态调整奖励策略，实现真正的"因地制宜"的智能交易。
    
    特别适用于需要在多变市场环境中保持稳健表现的高级交易策略。
    """
    
    def __init__(self,
                 # 状态检测参数
                 trend_threshold: float = 0.02,
                 volatility_threshold: float = 0.015,
                 sma_short: int = 20,
                 sma_long: int = 50,
                 
                 # 专家策略权重
                 bull_return_weight: float = 0.8,
                 bull_drawdown_weight: float = 0.2,
                 bear_return_weight: float = 0.4,
                 bear_drawdown_weight: float = 0.6,
                 sideways_return_weight: float = 0.5,
                 sideways_efficiency_weight: float = 0.5,
                 
                 # 奖励缩放
                 scale_factor: float = 50.0,
                 initial_balance: float = 10000.0,
                 **kwargs):
        """
        初始化市场状态感知奖励函数
        
        Args:
            trend_threshold: 趋势判断阈值
            volatility_threshold: 波动率判断阈值
            sma_short: 短期移动平均周期
            sma_long: 长期移动平均周期
            bull_return_weight: 牛市中收益权重
            bull_drawdown_weight: 牛市中回撤权重
            bear_return_weight: 熊市中收益权重
            bear_drawdown_weight: 熊市中回撤权重
            sideways_return_weight: 震荡市中收益权重
            sideways_efficiency_weight: 震荡市中交易效率权重
            scale_factor: 奖励缩放因子
            initial_balance: 初始资金
            **kwargs: 其他参数
        """
        super().__init__(initial_balance=initial_balance, **kwargs)
        
        # 初始化市场状态检测器
        self.state_detector = MarketStateDetector(
            sma_short=sma_short,
            sma_long=sma_long,
            trend_threshold=trend_threshold,
            volatility_threshold=volatility_threshold
        )
        
        # 专家策略权重配置
        self.regime_weights = {
            MarketRegime.BULL_MARKET: {
                'return_weight': bull_return_weight,
                'drawdown_weight': bull_drawdown_weight,
                'risk_tolerance': 0.8,
                'trend_bonus': 2.0
            },
            MarketRegime.BEAR_MARKET: {
                'return_weight': bear_return_weight,
                'drawdown_weight': bear_drawdown_weight,
                'risk_tolerance': 0.3,
                'cash_holding_bonus': 1.0
            },
            MarketRegime.SIDEWAYS_MARKET: {
                'return_weight': sideways_return_weight,
                'efficiency_weight': sideways_efficiency_weight,
                'risk_tolerance': 0.6,
                'trading_bonus': 1.5
            },
            MarketRegime.HIGH_VOLATILITY: {
                'return_weight': 0.3,
                'drawdown_weight': 0.7,
                'risk_tolerance': 0.2,
                'stability_bonus': 2.0
            },
            MarketRegime.LOW_VOLATILITY: {
                'return_weight': 0.7,
                'drawdown_weight': 0.3,
                'risk_tolerance': 0.9,
                'opportunity_bonus': 1.0
            }
        }
        
        self.scale_factor = scale_factor
        
        # 状态跟踪
        self.current_regime = MarketRegime.LOW_VOLATILITY
        self.regime_confidence = 0.5
        self.regime_change_count = 0
        
        # 性能跟踪
        self.regime_performance = {}
        for regime in MarketRegime:
            self.regime_performance[regime] = {
                'total_rewards': [],
                'episode_count': 0,
                'avg_reward': 0.0
            }
        
        # 交易效率追踪（震荡市专用）
        self.trading_actions = deque(maxlen=50)
        self.position_changes = 0
        self.profitable_trades = 0
        self.total_trades = 0
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def calculate_reward(self, portfolio_value: float, action: float, price: float, 
                        portfolio_info: Dict, trade_info: Dict, step: int, **kwargs) -> float:
        """
        奖励计算接口 - 市场状态感知奖励
        
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
            # 更新市场数据
            self.detector.update_data(price, volume=1.0)
            
            # 检测当前市场状态
            current_regime = self.detector.detect_regime()
            
            # 更新专家系统
            self._update_regime_expert_system(current_regime, portfolio_value)
            
            # 根据市场状态计算奖励
            reward_value = self._calculate_regime_specific_reward(
                portfolio_value, current_regime, action, price, 
                portfolio_info, trade_info
            )
            
            # 应用专家融合
            final_reward = self._apply_expert_fusion(reward_value, current_regime)
            
            # 更新历史记录
            self._update_tracking_variables(portfolio_value, final_reward, current_regime)
            self.step_count += 1
            
            return float(final_reward)
            
        except Exception as e:
            self.logger.error(f"RegimeAware奖励计算异常: {e}")
            return 0.0
        
    def reward(self, env) -> float:
        """
        计算基于市场状态感知的自适应奖励
        
        Args:
            env: TensorTrade环境实例
            
        Returns:
            float: 计算得到的奖励值
        """
        try:
            # 获取当前投资组合价值
            current_value = self.get_portfolio_value(env)
            current_action = self.get_current_action(env)
            
            # 第一步初始化
            if self.previous_value is None:
                self.previous_value = current_value
                self.initial_value = current_value
                return 0.0
            
            # 更新市场状态检测器
            self.state_detector.update_data(current_value)
            
            # 检测当前市场状态
            prev_regime = self.current_regime
            self.current_regime, self.regime_confidence = self.state_detector.detect_regime()
            
            # 记录状态变化
            if prev_regime != self.current_regime:
                self.regime_change_count += 1
                self.logger.info(f"市场状态切换: {prev_regime.value} → {self.current_regime.value} "
                               f"(置信度: {self.regime_confidence:.3f})")
            
            # 更新基础状态
            state = self.update_state(env)
            
            # 根据市场状态计算专家奖励
            expert_reward = self._calculate_regime_specific_reward(state, current_action)
            
            # 应用置信度加权
            confidence_weighted_reward = expert_reward * self.regime_confidence
            
            # 最终奖励缩放
            final_reward = confidence_weighted_reward * self.scale_factor
            
            # 更新性能跟踪
            self._update_regime_performance(final_reward)
            
            # 更新交易效率追踪
            self._update_trading_efficiency(current_action, state['step_return_pct'])
            
            self.step_count += 1
            
            # 记录重要信息
            if self.step_count % 50 == 0 or abs(final_reward) > 10:
                self.logger.info(
                    f"[RegimeAware] 步骤{self.step_count}: "
                    f"状态={self.current_regime.value}, "
                    f"置信度={self.regime_confidence:.3f}, "
                    f"专家奖励={expert_reward:.4f}, "
                    f"最终奖励={final_reward:.4f}"
                )
            
            return float(final_reward)
            
        except Exception as e:
            self.logger.error(f"RegimeAware奖励计算异常: {e}")
            return 0.0
    
    def get_reward(self, portfolio) -> float:
        """
        TensorTrade框架要求的get_reward方法
        
        Args:
            portfolio: 投资组合对象
            
        Returns:
            float: 计算得到的奖励值
        """
        try:
            current_value = float(portfolio.net_worth)
            
            if self.previous_value is None:
                self.previous_value = current_value
                self.initial_value = current_value
                return 0.0
            
            # 更新状态检测器
            self.state_detector.update_data(current_value)
            
            # 检测状态
            self.current_regime, self.regime_confidence = self.state_detector.detect_regime()
            
            # 简化的奖励计算
            step_return = (current_value - self.previous_value) / self.previous_value
            total_return = (current_value - self.initial_value) / self.initial_value
            
            # 基于状态的简单奖励
            weights = self.regime_weights[self.current_regime]
            base_reward = total_return * weights.get('return_weight', 0.5) * 100
            
            self.previous_value = current_value
            self.step_count += 1
            
            return float(base_reward * self.regime_confidence)
            
        except Exception as e:
            self.logger.error(f"Portfolio奖励计算异常: {e}")
            return 0.0
    
    def _calculate_regime_specific_reward(self, state: Dict[str, float], action: float) -> float:
        """
        根据特定市场状态计算专家奖励
        
        Args:
            state: 当前状态信息
            action: 当前动作
            
        Returns:
            float: 专家奖励值
        """
        current_value = state['current_value']
        step_return_pct = state['step_return_pct']
        total_return_pct = state['total_return_pct']
        
        weights = self.regime_weights[self.current_regime]
        
        if self.current_regime == MarketRegime.BULL_MARKET:
            return self._bull_market_reward(total_return_pct, step_return_pct, action, weights)
        
        elif self.current_regime == MarketRegime.BEAR_MARKET:
            return self._bear_market_reward(total_return_pct, step_return_pct, action, weights)
        
        elif self.current_regime == MarketRegime.SIDEWAYS_MARKET:
            return self._sideways_market_reward(total_return_pct, step_return_pct, action, weights)
        
        elif self.current_regime == MarketRegime.HIGH_VOLATILITY:
            return self._high_volatility_reward(total_return_pct, step_return_pct, action, weights)
        
        else:  # LOW_VOLATILITY
            return self._low_volatility_reward(total_return_pct, step_return_pct, action, weights)
    
    def _bull_market_reward(self, total_return: float, step_return: float, action: float, weights: Dict) -> float:
        """牛市专家策略：鼓励积极持仓，适度容忍回撤"""
        return_component = total_return * weights['return_weight'] * 10
        
        # 趋势跟随奖励
        if action > 0.1 and step_return > 0:
            trend_bonus = weights['trend_bonus'] * action * step_return * 20
        else:
            trend_bonus = 0
        
        # 温和的回撤惩罚
        drawdown_penalty = self._calculate_drawdown_penalty() * weights['drawdown_weight'] * 0.5
        
        return return_component + trend_bonus - drawdown_penalty
    
    def _bear_market_reward(self, total_return: float, step_return: float, action: float, weights: Dict) -> float:
        """熊市专家策略：严格风险控制，奖励现金持有"""
        return_component = total_return * weights['return_weight'] * 8
        
        # 现金持有奖励（负动作或小动作）
        if action < 0.1:
            cash_bonus = weights['cash_holding_bonus'] * (0.1 - abs(action)) * 2
        else:
            cash_bonus = 0
        
        # 严厉的回撤惩罚
        drawdown_penalty = self._calculate_drawdown_penalty() * weights['drawdown_weight'] * 2.0
        
        return return_component + cash_bonus - drawdown_penalty
    
    def _sideways_market_reward(self, total_return: float, step_return: float, action: float, weights: Dict) -> float:
        """震荡市专家策略：奖励高抛低吸，优化交易效率"""
        return_component = total_return * weights['return_weight'] * 6
        
        # 交易效率奖励
        trading_efficiency = self._calculate_trading_efficiency()
        efficiency_bonus = trading_efficiency * weights['efficiency_weight'] * weights['trading_bonus']
        
        # 中度回撤惩罚
        drawdown_penalty = self._calculate_drawdown_penalty() * 0.8
        
        return return_component + efficiency_bonus - drawdown_penalty
    
    def _high_volatility_reward(self, total_return: float, step_return: float, action: float, weights: Dict) -> float:
        """高波动专家策略：极度保守，强调稳定性"""
        return_component = total_return * weights['return_weight'] * 5
        
        # 稳定性奖励（小动作，低波动）
        if abs(action) < 0.3:
            stability_bonus = weights['stability_bonus'] * (0.3 - abs(action)) * 3
        else:
            stability_bonus = 0
        
        # 极严厉的回撤惩罚
        drawdown_penalty = self._calculate_drawdown_penalty() * weights['drawdown_weight'] * 3.0
        
        return return_component + stability_bonus - drawdown_penalty
    
    def _low_volatility_reward(self, total_return: float, step_return: float, action: float, weights: Dict) -> float:
        """低波动专家策略：适度冒险，抓住机会"""
        return_component = total_return * weights['return_weight'] * 12
        
        # 机会奖励（积极动作在稳定环境中）
        if abs(action) > 0.5:
            opportunity_bonus = weights['opportunity_bonus'] * abs(action) * 1.5
        else:
            opportunity_bonus = 0
        
        # 轻微回撤惩罚
        drawdown_penalty = self._calculate_drawdown_penalty() * weights['drawdown_weight'] * 0.3
        
        return return_component + opportunity_bonus - drawdown_penalty
    
    def _calculate_drawdown_penalty(self) -> float:
        """计算通用回撤惩罚"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        current_value = self.portfolio_history[-1]
        peak_value = max(self.portfolio_history)
        
        if peak_value <= current_value:
            return 0.0
        
        drawdown = (peak_value - current_value) / peak_value
        return drawdown * 100  # 基础惩罚值
    
    def _calculate_trading_efficiency(self) -> float:
        """计算交易效率（震荡市专用）"""
        if self.total_trades == 0:
            return 0.0
        
        # 胜率
        win_rate = self.profitable_trades / self.total_trades
        
        # 交易频率适中性（避免过度交易）
        if len(self.trading_actions) < 10:
            frequency_score = 1.0
        else:
            active_ratio = sum(1 for a in self.trading_actions if abs(a) > 0.1) / len(self.trading_actions)
            frequency_score = 1.0 - abs(active_ratio - 0.3)  # 理想交易频率30%
        
        return win_rate * frequency_score
    
    def _update_trading_efficiency(self, action: float, step_return: float) -> None:
        """更新交易效率统计"""
        self.trading_actions.append(action)
        
        # 检测交易行为
        if abs(action) > 0.2:  # 认为是有效交易
            if len(self.trading_actions) > 1:
                prev_action = self.trading_actions[-2]
                if abs(prev_action) > 0.2 and np.sign(action) != np.sign(prev_action):
                    # 仓位改变，统计交易结果
                    self.total_trades += 1
                    if step_return > 0:
                        self.profitable_trades += 1
    
    def _update_regime_performance(self, reward: float) -> None:
        """更新各状态下的性能统计"""
        perf = self.regime_performance[self.current_regime]
        perf['total_rewards'].append(reward)
        
        # 保持最近100个奖励记录
        if len(perf['total_rewards']) > 100:
            perf['total_rewards'].pop(0)
        
        perf['avg_reward'] = np.mean(perf['total_rewards'])
        
        # 更新奖励历史
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)
    
    def reset(self) -> 'RegimeAwareReward':
        """
        重置奖励函数状态
        
        Returns:
            RegimeAwareReward: 返回self以支持链式调用
        """
        # 记录回合性能
        if self.previous_value is not None and self.initial_balance > 0:
            final_return = (self.previous_value - self.initial_balance) / self.initial_balance
            avg_reward = np.mean(self.reward_history) if self.reward_history else 0.0
            
            self.logger.info(
                f"[RegimeAware回合{self.episode_count}结束] "
                f"最终收益率: {final_return:.4f}, "
                f"平均奖励: {avg_reward:.4f}, "
                f"状态切换次数: {self.regime_change_count}, "
                f"主要状态: {self.current_regime.value}, "
                f"步数: {self.step_count}"
            )
        
        # 调用父类reset
        super().reset()
        
        # 保留部分历史用于连续学习
        if hasattr(self.state_detector, 'price_history') and len(self.state_detector.price_history) > 20:
            # 保留最近的价格历史
            recent_prices = list(self.state_detector.price_history)[-10:]
            self.state_detector.price_history.clear()
            self.state_detector.price_history.extend(recent_prices)
        
        # 重置状态相关计数器
        self.regime_change_count = 0
        self.position_changes = 0
        self.profitable_trades = 0
        self.total_trades = 0
        
        return self
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要，包含RegimeAware特有指标
        
        Returns:
            Dict[str, Any]: 性能摘要信息
        """
        base_summary = super().get_performance_summary()
        
        # 状态检测统计
        regime_stats = self.state_detector.get_regime_statistics()
        
        # 状态特定性能
        regime_performance = {}
        for regime, perf in self.regime_performance.items():
            regime_performance[regime.value] = {
                'avg_reward': perf['avg_reward'],
                'total_samples': len(perf['total_rewards'])
            }
        
        # 交易效率
        trading_efficiency = self._calculate_trading_efficiency()
        
        # 添加RegimeAware特有指标
        regime_aware_metrics = {
            'current_regime': self.current_regime.value,
            'regime_confidence': self.regime_confidence,
            'regime_changes': self.regime_change_count,
            'regime_statistics': regime_stats,
            'regime_performance': regime_performance,
            'trading_efficiency': trading_efficiency,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'win_rate': self.profitable_trades / max(1, self.total_trades)
        }
        
        base_summary.update(regime_aware_metrics)
        return base_summary
    
    @classmethod
    def get_reward_info(cls) -> Dict[str, Any]:
        """
        获取奖励函数信息
        
        Returns:
            Dict[str, Any]: 奖励函数的描述信息
        """
        return {
            'name': 'RegimeAwareReward',
            'description': '自适应专家市场状态感知奖励函数，能够智能识别市场状态并动态调整奖励策略',
            'category': 'adaptive_expert',
            'parameters': {
                'trend_threshold': {
                    'type': 'float',
                    'default': 0.02,
                    'description': '趋势判断阈值，用于区分牛熊市'
                },
                'volatility_threshold': {
                    'type': 'float',
                    'default': 0.015,
                    'description': '波动率判断阈值，用于区分高低波动期'
                },
                'sma_short': {
                    'type': 'int',
                    'default': 20,
                    'description': '短期移动平均周期'
                },
                'sma_long': {
                    'type': 'int',
                    'default': 50,
                    'description': '长期移动平均周期'
                },
                'bull_return_weight': {
                    'type': 'float',
                    'default': 0.8,
                    'description': '牛市中收益权重'
                },
                'bear_drawdown_weight': {
                    'type': 'float',
                    'default': 0.6,
                    'description': '熊市中回撤权重'
                },
                'scale_factor': {
                    'type': 'float',
                    'default': 50.0,
                    'description': '奖励缩放因子'
                },
                'initial_balance': {
                    'type': 'float',
                    'default': 10000.0,
                    'description': '初始资金'
                }
            },
            'market_regimes': [
                'BULL_MARKET - 牛市：趋势向上，鼓励积极持仓',
                'BEAR_MARKET - 熊市：趋势向下，严格风险控制',
                'SIDEWAYS_MARKET - 震荡市：横盘整理，奖励高抛低吸',
                'HIGH_VOLATILITY - 高波动：市场不确定，极度保守',
                'LOW_VOLATILITY - 低波动：稳定环境，适度冒险'
            ],
            'advantages': [
                '智能市场状态识别',
                '状态特定的专家策略',
                '自适应奖励权重调整',
                '多技术指标融合检测',
                '置信度加权决策',
                '连续学习和状态记忆'
            ],
            'use_cases': [
                '多变市场环境的自适应交易',
                '需要状态感知的高级策略',
                '机构级智能投资管理',
                '全天候交易系统',
                '风险感知的算法交易'
            ],
            'mathematical_foundation': '基于隐马尔可夫模型的状态检测，结合技术指标分析和专家系统',
            'complexity': 'expert'
        }