"""
Real-time Reward-Return Correlation Monitor
奖励-回报相关性实时监控系统

Purpose: 实时监控强化学习训练过程中奖励函数与实际回报的相关性
解决历史实验中奖励-回报脱钩问题的检测与预警

Key Features:
- 实时计算奖励与回报的相关性
- 相关性异常检测与预警
- 训练过程中的动态调整建议
- 详细的相关性分析报告
"""

import numpy as np
import pandas as pd
import logging
from collections import deque
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
import json
import matplotlib.pyplot as plt
from threading import Lock
import warnings

from ..utils.logger import setup_logger


class CorrelationMonitor:
    """
    奖励-回报相关性实时监控器
    
    专门用于监控强化学习训练过程中奖励与实际回报的相关性
    提供实时检测、预警和调整建议
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = setup_logger("CorrelationMonitor")
        
        # 监控配置
        self.correlation_threshold = self.config.get('correlation_threshold', 0.8)
        self.warning_threshold = self.config.get('warning_threshold', 0.6)
        self.critical_threshold = self.config.get('critical_threshold', 0.4)
        self.min_samples_for_correlation = self.config.get('min_samples', 10)
        
        # 滑动窗口配置
        self.window_size = self.config.get('window_size', 100)
        self.update_frequency = self.config.get('update_frequency', 10)  # 每10步更新一次
        
        # 数据存储
        self._lock = Lock()
        self.episode_rewards = deque(maxlen=self.window_size * 2)
        self.episode_returns = deque(maxlen=self.window_size * 2) 
        self.step_rewards = deque(maxlen=self.window_size * 10)
        self.step_returns = deque(maxlen=self.window_size * 10)
        
        # 相关性历史
        self.correlation_history = []
        self.correlation_timestamps = []
        
        # 统计计数
        self.total_episodes = 0
        self.total_steps = 0
        self.update_counter = 0
        
        # 预警系统
        self.alerts = []
        self.alert_callbacks = []
        
        # 分析结果缓存
        self.latest_analysis = {}
        
        self.logger.info("CorrelationMonitor初始化完成")
        self.logger.info(f"相关性阈值: 目标={self.correlation_threshold}, 警告={self.warning_threshold}, 严重={self.critical_threshold}")

    def record_episode(self, episode_reward: float, episode_return: float, 
                      episode_info: Dict = None):
        """
        记录一个完整episode的奖励和回报
        
        Args:
            episode_reward: episode总奖励
            episode_return: episode总回报率(%)
            episode_info: 额外的episode信息
        """
        
        with self._lock:
            self.episode_rewards.append(episode_reward)
            self.episode_returns.append(episode_return)
            self.total_episodes += 1
            
            # 每隔一定频率进行相关性分析
            if self.total_episodes % self.update_frequency == 0:
                self._update_correlation_analysis()
            
            # 记录详细信息用于调试
            if episode_info:
                self.logger.debug(f"Episode {self.total_episodes}: "
                                f"奖励={episode_reward:.2f}, 回报={episode_return:.2f}%, "
                                f"额外信息={episode_info}")

    def record_step(self, step_reward: float, step_return: float):
        """
        记录单步的奖励和回报
        
        Args:
            step_reward: 单步奖励
            step_return: 单步回报率(%)
        """
        
        with self._lock:
            self.step_rewards.append(step_reward)
            self.step_returns.append(step_return)
            self.total_steps += 1
            
            self.update_counter += 1
            
            # 高频更新步级相关性（可选）
            if self.update_counter % (self.update_frequency * 5) == 0:
                self._update_step_correlation_analysis()

    def get_current_correlation(self, analysis_type: str = "episode") -> Dict[str, Any]:
        """
        获取当前相关性分析结果
        
        Args:
            analysis_type: 分析类型 ("episode", "step", "both")
            
        Returns:
            current_correlation_analysis: 当前相关性分析
        """
        
        with self._lock:
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "analysis_type": analysis_type,
                "data_points": {
                    "episode_count": len(self.episode_rewards),
                    "step_count": len(self.step_rewards)
                },
                "correlation_results": {},
                "status": "UNKNOWN",
                "alerts": []
            }
            
            if analysis_type in ["episode", "both"] and len(self.episode_rewards) >= self.min_samples_for_correlation:
                episode_correlation = self._calculate_correlation(
                    list(self.episode_rewards), list(self.episode_returns)
                )
                
                analysis["correlation_results"]["episode_correlation"] = {
                    "correlation": episode_correlation,
                    "status": self._get_correlation_status(episode_correlation),
                    "sample_size": len(self.episode_rewards)
                }
            
            if analysis_type in ["step", "both"] and len(self.step_rewards) >= self.min_samples_for_correlation:
                step_correlation = self._calculate_correlation(
                    list(self.step_rewards), list(self.step_returns)
                )
                
                analysis["correlation_results"]["step_correlation"] = {
                    "correlation": step_correlation,
                    "status": self._get_correlation_status(step_correlation),
                    "sample_size": len(self.step_rewards)
                }
            
            # 确定整体状态
            correlations = []
            if "episode_correlation" in analysis["correlation_results"]:
                correlations.append(analysis["correlation_results"]["episode_correlation"]["correlation"])
            if "step_correlation" in analysis["correlation_results"]:
                correlations.append(analysis["correlation_results"]["step_correlation"]["correlation"])
            
            if correlations:
                best_correlation = max(correlations)
                analysis["status"] = self._get_correlation_status(best_correlation)
                analysis["best_correlation"] = best_correlation
                
                # 检查是否需要预警
                self._check_correlation_alerts(best_correlation, analysis)
            
            self.latest_analysis = analysis
            return analysis

    def get_correlation_trends(self, lookback_periods: int = 50) -> Dict[str, Any]:
        """
        获取相关性变化趋势
        
        Args:
            lookback_periods: 回顾周期数
            
        Returns:
            correlation_trends: 相关性趋势分析
        """
        
        with self._lock:
            if len(self.correlation_history) < 3:
                return {
                    "status": "INSUFFICIENT_DATA",
                    "message": "相关性历史数据不足，无法分析趋势"
                }
            
            # 提取最近的相关性数据
            recent_correlations = self.correlation_history[-lookback_periods:]
            recent_timestamps = self.correlation_timestamps[-lookback_periods:]
            
            trends_analysis = {
                "analysis_period": f"最近{len(recent_correlations)}个监控点",
                "correlation_statistics": {
                    "mean": np.mean(recent_correlations),
                    "std": np.std(recent_correlations),
                    "min": np.min(recent_correlations),
                    "max": np.max(recent_correlations),
                    "latest": recent_correlations[-1] if recent_correlations else 0
                },
                "trend_direction": self._analyze_trend_direction(recent_correlations),
                "volatility_analysis": self._analyze_correlation_volatility(recent_correlations),
                "improvement_suggestions": []
            }
            
            # 生成改进建议
            trends_analysis["improvement_suggestions"] = self._generate_improvement_suggestions(trends_analysis)
            
            return trends_analysis

    def generate_correlation_report(self, save_to_file: bool = True, 
                                  file_path: str = None) -> Dict[str, Any]:
        """
        生成详细的相关性分析报告
        
        Args:
            save_to_file: 是否保存到文件
            file_path: 保存文件路径
            
        Returns:
            comprehensive_report: 综合相关性报告
        """
        
        current_time = datetime.now()
        
        report = {
            "report_metadata": {
                "report_type": "Correlation_Monitoring_Report",
                "generation_time": current_time.isoformat(),
                "monitor_duration": str(current_time - datetime.fromisoformat(self.correlation_timestamps[0])) if self.correlation_timestamps else "N/A",
                "total_episodes_monitored": self.total_episodes,
                "total_steps_monitored": self.total_steps
            },
            
            "current_status": self.get_current_correlation("both"),
            "trend_analysis": self.get_correlation_trends(),
            "alert_summary": self._generate_alert_summary(),
            "performance_assessment": self._assess_correlation_performance(),
            "recommendations": self._generate_comprehensive_recommendations()
        }
        
        # 添加统计摘要
        if self.correlation_history:
            report["historical_summary"] = {
                "correlation_count": len(self.correlation_history),
                "best_correlation": max(self.correlation_history),
                "worst_correlation": min(self.correlation_history),
                "average_correlation": np.mean(self.correlation_history),
                "correlation_stability": 1 - (np.std(self.correlation_history) / (np.mean(self.correlation_history) + 1e-8))
            }
        
        # 保存到文件
        if save_to_file:
            if not file_path:
                file_path = f"correlation_report_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
            
            try:
                with open(file_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                self.logger.info(f"相关性报告已保存: {file_path}")
                report["report_metadata"]["saved_to"] = file_path
                
            except Exception as e:
                self.logger.error(f"保存相关性报告失败: {e}")
                report["report_metadata"]["save_error"] = str(e)
        
        return report

    def register_alert_callback(self, callback: Callable[[Dict], None]):
        """
        注册预警回调函数
        
        Args:
            callback: 当发生相关性预警时调用的回调函数
        """
        self.alert_callbacks.append(callback)
        self.logger.info(f"注册预警回调函数: {callback.__name__}")

    def plot_correlation_history(self, save_path: str = None, show_plot: bool = False):
        """
        绘制相关性历史图表
        
        Args:
            save_path: 图表保存路径
            show_plot: 是否显示图表
        """
        
        if len(self.correlation_history) < 3:
            self.logger.warning("相关性历史数据不足，无法绘制图表")
            return
        
        try:
            plt.figure(figsize=(12, 8))
            
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 相关性趋势图
            ax1.plot(self.correlation_history, 'b-', linewidth=2, label='奖励-回报相关性')
            ax1.axhline(y=self.correlation_threshold, color='g', linestyle='--', 
                       label=f'目标阈值 ({self.correlation_threshold})')
            ax1.axhline(y=self.warning_threshold, color='orange', linestyle='--', 
                       label=f'警告阈值 ({self.warning_threshold})')
            ax1.axhline(y=self.critical_threshold, color='r', linestyle='--', 
                       label=f'严重阈值 ({self.critical_threshold})')
            
            ax1.set_title('奖励-回报相关性监控历史')
            ax1.set_xlabel('监控点')
            ax1.set_ylabel('相关性系数')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(-1.1, 1.1)
            
            # 相关性分布直方图
            ax2.hist(self.correlation_history, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(x=self.correlation_threshold, color='g', linestyle='--', label='目标阈值')
            ax2.axvline(x=np.mean(self.correlation_history), color='red', linestyle='-', 
                       label=f'平均值 ({np.mean(self.correlation_history):.3f})')
            
            ax2.set_title('相关性分布直方图')
            ax2.set_xlabel('相关性系数')
            ax2.set_ylabel('频次')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"相关性图表已保存: {save_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            self.logger.error(f"绘制相关性图表失败: {e}")

    def reset(self):
        """重置监控器状态"""
        
        with self._lock:
            self.episode_rewards.clear()
            self.episode_returns.clear()
            self.step_rewards.clear()
            self.step_returns.clear()
            self.correlation_history.clear()
            self.correlation_timestamps.clear()
            self.alerts.clear()
            
            self.total_episodes = 0
            self.total_steps = 0
            self.update_counter = 0
            self.latest_analysis = {}
            
            self.logger.info("CorrelationMonitor已重置")

    def _calculate_correlation(self, rewards: List[float], returns: List[float]) -> float:
        """计算奖励与回报的相关性"""
        
        if len(rewards) != len(returns) or len(rewards) < 2:
            return 0.0
        
        try:
            # 过滤无效值
            valid_pairs = [(r, ret) for r, ret in zip(rewards, returns) 
                          if np.isfinite(r) and np.isfinite(ret)]
            
            if len(valid_pairs) < 2:
                return 0.0
            
            rewards_clean, returns_clean = zip(*valid_pairs)
            
            correlation = np.corrcoef(rewards_clean, returns_clean)[0, 1]
            
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            self.logger.warning(f"计算相关性失败: {e}")
            return 0.0

    def _get_correlation_status(self, correlation: float) -> str:
        """根据相关性值确定状态"""
        
        if abs(correlation) >= self.correlation_threshold:
            return "EXCELLENT"
        elif abs(correlation) >= self.warning_threshold:
            return "GOOD"
        elif abs(correlation) >= self.critical_threshold:
            return "WARNING"
        else:
            return "CRITICAL"

    def _update_correlation_analysis(self):
        """更新相关性分析"""
        
        if len(self.episode_rewards) >= self.min_samples_for_correlation:
            current_correlation = self._calculate_correlation(
                list(self.episode_rewards), list(self.episode_returns)
            )
            
            self.correlation_history.append(current_correlation)
            self.correlation_timestamps.append(datetime.now().isoformat())
            
            # 检查预警条件
            self._check_correlation_alerts(current_correlation)
            
            self.logger.debug(f"相关性更新: {current_correlation:.3f} "
                            f"({self._get_correlation_status(current_correlation)})")

    def _update_step_correlation_analysis(self):
        """更新步级相关性分析"""
        
        if len(self.step_rewards) >= self.min_samples_for_correlation:
            step_correlation = self._calculate_correlation(
                list(self.step_rewards), list(self.step_returns)
            )
            
            self.logger.debug(f"步级相关性: {step_correlation:.3f}")

    def _check_correlation_alerts(self, correlation: float, analysis: Dict = None):
        """检查相关性预警条件"""
        
        alert = None
        
        if abs(correlation) < self.critical_threshold:
            alert = {
                "level": "CRITICAL",
                "message": f"严重警告：奖励-回报相关性极低 ({correlation:.3f})",
                "correlation": correlation,
                "timestamp": datetime.now().isoformat(),
                "action_required": "立即检查奖励函数设计"
            }
        elif abs(correlation) < self.warning_threshold:
            alert = {
                "level": "WARNING", 
                "message": f"警告：奖励-回报相关性偏低 ({correlation:.3f})",
                "correlation": correlation,
                "timestamp": datetime.now().isoformat(),
                "action_required": "考虑调整奖励函数参数"
            }
        
        if alert:
            self.alerts.append(alert)
            
            # 调用注册的回调函数
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"执行预警回调失败: {e}")
            
            self.logger.warning(alert["message"])
            
            # 添加到分析结果中
            if analysis:
                analysis["alerts"].append(alert)

    def _analyze_trend_direction(self, correlations: List[float]) -> Dict[str, Any]:
        """分析相关性趋势方向"""
        
        if len(correlations) < 3:
            return {"direction": "UNKNOWN", "confidence": 0}
        
        # 简单线性趋势分析
        x = np.arange(len(correlations))
        coeffs = np.polyfit(x, correlations, 1)
        slope = coeffs[0]
        
        if slope > 0.01:
            direction = "IMPROVING"
            confidence = min(abs(slope) * 10, 1.0)
        elif slope < -0.01:
            direction = "DECLINING"
            confidence = min(abs(slope) * 10, 1.0)
        else:
            direction = "STABLE"
            confidence = 1.0 - abs(slope) * 10
        
        return {
            "direction": direction,
            "slope": slope,
            "confidence": confidence,
            "interpretation": self._interpret_trend(direction, slope)
        }

    def _analyze_correlation_volatility(self, correlations: List[float]) -> Dict[str, Any]:
        """分析相关性波动情况"""
        
        if len(correlations) < 3:
            return {"volatility": "UNKNOWN", "stability": 0}
        
        volatility = np.std(correlations)
        mean_correlation = np.mean(correlations)
        
        if volatility < 0.05:
            volatility_level = "LOW"
            stability = 1.0
        elif volatility < 0.15:
            volatility_level = "MODERATE"
            stability = 0.7
        else:
            volatility_level = "HIGH"
            stability = 0.3
        
        return {
            "volatility": volatility_level,
            "volatility_value": volatility,
            "stability": stability,
            "coefficient_of_variation": volatility / (abs(mean_correlation) + 1e-8)
        }

    def _generate_improvement_suggestions(self, trends_analysis: Dict) -> List[str]:
        """根据趋势分析生成改进建议"""
        
        suggestions = []
        
        correlation_stats = trends_analysis.get("correlation_statistics", {})
        trend_direction = trends_analysis.get("trend_direction", {})
        volatility_analysis = trends_analysis.get("volatility_analysis", {})
        
        mean_correlation = correlation_stats.get("mean", 0)
        trend_dir = trend_direction.get("direction", "UNKNOWN")
        volatility_level = volatility_analysis.get("volatility", "UNKNOWN")
        
        if mean_correlation < self.critical_threshold:
            suggestions.append("紧急：奖励函数设计存在根本问题，建议重新设计")
        elif mean_correlation < self.warning_threshold:
            suggestions.append("建议：优化奖励函数参数，提高与实际回报的相关性")
        
        if trend_dir == "DECLINING":
            suggestions.append("警告：相关性呈下降趋势，需要调整训练策略")
        elif trend_dir == "IMPROVING":
            suggestions.append("良好：相关性正在改善，继续当前策略")
        
        if volatility_level == "HIGH":
            suggestions.append("建议：相关性波动较大，考虑增加训练稳定性")
        
        return suggestions

    def _generate_alert_summary(self) -> Dict[str, Any]:
        """生成预警摘要"""
        
        if not self.alerts:
            return {
                "total_alerts": 0,
                "alert_levels": {},
                "recent_alerts": [],
                "status": "NO_ALERTS"
            }
        
        alert_levels = {}
        for alert in self.alerts:
            level = alert["level"]
            alert_levels[level] = alert_levels.get(level, 0) + 1
        
        recent_alerts = sorted(self.alerts, key=lambda x: x["timestamp"])[-5:]
        
        return {
            "total_alerts": len(self.alerts),
            "alert_levels": alert_levels,
            "recent_alerts": recent_alerts,
            "status": "ACTIVE_ALERTS" if "CRITICAL" in alert_levels else "NORMAL_ALERTS"
        }

    def _assess_correlation_performance(self) -> Dict[str, Any]:
        """评估相关性表现"""
        
        if not self.correlation_history:
            return {"status": "NO_DATA"}
        
        recent_correlations = self.correlation_history[-20:]  # 最近20个监控点
        
        assessment = {
            "overall_performance": "UNKNOWN",
            "consistency": "UNKNOWN",
            "improvement_rate": 0,
            "target_achievement": False
        }
        
        mean_correlation = np.mean(recent_correlations)
        std_correlation = np.std(recent_correlations)
        
        # 整体表现评估
        if mean_correlation >= self.correlation_threshold:
            assessment["overall_performance"] = "EXCELLENT"
        elif mean_correlation >= self.warning_threshold:
            assessment["overall_performance"] = "GOOD"
        elif mean_correlation >= self.critical_threshold:
            assessment["overall_performance"] = "NEEDS_IMPROVEMENT"
        else:
            assessment["overall_performance"] = "POOR"
        
        # 一致性评估
        if std_correlation < 0.05:
            assessment["consistency"] = "HIGH"
        elif std_correlation < 0.15:
            assessment["consistency"] = "MODERATE"
        else:
            assessment["consistency"] = "LOW"
        
        # 改进率计算
        if len(recent_correlations) > 10:
            early_mean = np.mean(recent_correlations[:10])
            late_mean = np.mean(recent_correlations[-10:])
            assessment["improvement_rate"] = late_mean - early_mean
        
        # 目标达成评估
        assessment["target_achievement"] = mean_correlation >= self.correlation_threshold
        
        return assessment

    def _generate_comprehensive_recommendations(self) -> List[str]:
        """生成综合改进建议"""
        
        recommendations = []
        
        if self.latest_analysis:
            status = self.latest_analysis.get("status", "UNKNOWN")
            best_correlation = self.latest_analysis.get("best_correlation", 0)
            
            if status == "CRITICAL":
                recommendations.extend([
                    "立即停止训练，检查奖励函数实现",
                    "验证环境返回的奖励与实际PnL计算",
                    "检查数据预处理和特征工程流程",
                    "考虑使用更简单的直接PnL奖励函数"
                ])
            elif status == "WARNING":
                recommendations.extend([
                    "调整奖励函数参数，提高相关性",
                    "增加奖励函数的数值稳定性控制",
                    "检查交易成本和滑点的计算方式",
                    "考虑引入奖励函数正则化"
                ])
            elif status in ["GOOD", "EXCELLENT"]:
                recommendations.extend([
                    "维持当前奖励函数配置",
                    "继续监控相关性稳定性",
                    "可以尝试优化其他超参数",
                    "准备进行生产环境测试"
                ])
        
        # 基于历史数据的通用建议
        if len(self.correlation_history) > 10:
            recent_std = np.std(self.correlation_history[-10:])
            if recent_std > 0.2:
                recommendations.append("相关性波动较大，考虑增加训练batch size或调整学习率")
        
        return recommendations

    def _interpret_trend(self, direction: str, slope: float) -> str:
        """解释趋势含义"""
        
        interpretations = {
            "IMPROVING": f"相关性正在改善，改善速度为每监控点{slope:.4f}",
            "DECLINING": f"相关性正在恶化，恶化速度为每监控点{slope:.4f}",
            "STABLE": f"相关性相对稳定，变化幅度为每监控点{slope:.4f}"
        }
        
        return interpretations.get(direction, "无法解释趋势")


# 使用示例和测试
if __name__ == "__main__":
    # 创建监控器
    monitor = CorrelationMonitor({
        'correlation_threshold': 0.8,
        'warning_threshold': 0.6,
        'window_size': 50,
        'update_frequency': 5
    })
    
    # 注册预警回调
    def alert_callback(alert):
        print(f"🚨 预警: {alert['message']}")
    
    monitor.register_alert_callback(alert_callback)
    
    # 模拟训练数据
    print("模拟奖励-回报相关性监控...")
    
    for episode in range(50):
        # 模拟不同情况的奖励和回报
        if episode < 20:
            # 初期：相关性较低
            base_return = np.random.normal(-10, 5)
            episode_reward = base_return * 0.3 + np.random.normal(0, 2)  # 低相关性
        elif episode < 35:
            # 中期：相关性改善
            base_return = np.random.normal(-5, 3)
            episode_reward = base_return * 0.7 + np.random.normal(0, 1)  # 中等相关性
        else:
            # 后期：高相关性
            base_return = np.random.normal(2, 2)
            episode_reward = base_return * 0.9 + np.random.normal(0, 0.5)  # 高相关性
        
        monitor.record_episode(episode_reward, base_return)
        
        # 每10个episode显示一次状态
        if (episode + 1) % 10 == 0:
            current_status = monitor.get_current_correlation()
            correlation = current_status.get("best_correlation", 0)
            status = current_status.get("status", "UNKNOWN")
            print(f"Episode {episode + 1}: 相关性={correlation:.3f}, 状态={status}")
    
    # 生成最终报告
    print("\n生成相关性监控报告...")
    final_report = monitor.generate_correlation_report(save_to_file=False)
    
    print("\n=== 监控总结 ===")
    print(f"监控episodes: {monitor.total_episodes}")
    print(f"最佳相关性: {max(monitor.correlation_history):.3f}")
    print(f"平均相关性: {np.mean(monitor.correlation_history):.3f}")
    print(f"预警次数: {len(monitor.alerts)}")
    
    # 绘制相关性历史图
    monitor.plot_correlation_history(save_path="correlation_monitor_test.png", show_plot=False)
    print("相关性历史图已保存")
    
    print("\nCorrelationMonitor测试完成！")