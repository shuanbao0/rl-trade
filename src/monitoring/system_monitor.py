"""
系统监控服务
监控系统性能、健康状态、资源使用和业务指标

主要功能:
1. 系统资源监控
2. 业务指标追踪
3. 健康状态检查
4. 性能基准测试
5. 历史数据存储
6. 实时监控面板
"""

import os
import time
import threading
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import pandas as pd
import numpy as np
from queue import Queue, Empty

from ..utils.logger import setup_logger, get_default_log_file
from ..utils.config import Config


@dataclass
class SystemMetrics:
    """系统指标数据"""
    timestamp: datetime
    
    # 系统资源
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    
    # 应用指标
    process_count: int
    thread_count: int
    open_files: int
    
    # 业务指标
    total_trades: int = 0
    active_orders: int = 0
    portfolio_value: float = 0.0
    daily_pnl: float = 0.0
    success_rate: float = 0.0
    
    # 性能指标
    data_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    order_latency_ms: float = 0.0
    
    # 错误统计
    error_count: int = 0
    warning_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'disk_percent': self.disk_percent,
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_recv': self.network_bytes_recv,
            'process_count': self.process_count,
            'thread_count': self.thread_count,
            'open_files': self.open_files,
            'total_trades': self.total_trades,
            'active_orders': self.active_orders,
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'success_rate': self.success_rate,
            'data_latency_ms': self.data_latency_ms,
            'inference_latency_ms': self.inference_latency_ms,
            'order_latency_ms': self.order_latency_ms,
            'error_count': self.error_count,
            'warning_count': self.warning_count
        }


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    component: str
    status: str  # "healthy", "warning", "critical"
    message: str
    timestamp: datetime
    response_time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBenchmark:
    """性能基准"""
    metric_name: str
    target_value: float
    warning_threshold: float
    critical_threshold: float
    unit: str
    description: str


class SystemMonitor:
    """
    系统监控器
    
    提供全面的系统监控功能:
    - 系统资源监控
    - 应用性能监控  
    - 业务指标追踪
    - 健康状态检查
    - 告警触发
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        collection_interval: int = 30,
        retention_hours: int = 72
    ):
        """
        初始化系统监控器
        
        Args:
            config: 配置对象
            collection_interval: 采集间隔(秒)
            retention_hours: 数据保留时间(小时)
        """
        self.config = config or Config()
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        
        # 初始化日志
        self.logger = setup_logger(
            name="SystemMonitor",
            level="INFO",
            log_file=get_default_log_file("system_monitor")
        )
        
        # 监控状态
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.health_check_thread: Optional[threading.Thread] = None
        
        # 数据存储
        self.metrics_history: deque = deque(maxlen=int(retention_hours * 3600 / collection_interval))
        self.health_checks: Dict[str, HealthCheckResult] = {}
        
        # 组件引用
        self.trading_system = None
        self.data_manager = None
        self.model_service = None
        self.broker_api = None
        
        # 健康检查器
        self.health_checkers: Dict[str, Callable[[], HealthCheckResult]] = {}
        
        # 性能基准
        self.benchmarks: Dict[str, PerformanceBenchmark] = {
            'data_latency_ms': PerformanceBenchmark(
                metric_name='data_latency_ms',
                target_value=50.0,
                warning_threshold=100.0,
                critical_threshold=200.0,
                unit='ms',
                description='数据延迟'
            ),
            'inference_latency_ms': PerformanceBenchmark(
                metric_name='inference_latency_ms', 
                target_value=25.0,
                warning_threshold=50.0,
                critical_threshold=100.0,
                unit='ms',
                description='推理延迟'
            ),
            'order_latency_ms': PerformanceBenchmark(
                metric_name='order_latency_ms',
                target_value=100.0,
                warning_threshold=200.0,
                critical_threshold=500.0,
                unit='ms',
                description='订单延迟'
            ),
            'cpu_percent': PerformanceBenchmark(
                metric_name='cpu_percent',
                target_value=50.0,
                warning_threshold=80.0,
                critical_threshold=95.0,
                unit='%',
                description='CPU使用率'
            ),
            'memory_percent': PerformanceBenchmark(
                metric_name='memory_percent',
                target_value=60.0,
                warning_threshold=85.0,
                critical_threshold=95.0,
                unit='%',
                description='内存使用率'
            )
        }
        
        # 告警回调
        self.alert_callbacks: List[Callable[[str, str, Dict[str, Any]], None]] = []
        
        # 初始网络统计
        self._last_network_stats = psutil.net_io_counters()
        self._network_start_time = time.time()
        
        self.logger.info("SystemMonitor 初始化完成")
    
    def register_components(
        self,
        trading_system=None,
        data_manager=None,
        model_service=None,
        broker_api=None
    ) -> None:
        """
        注册监控组件
        
        Args:
            trading_system: 交易系统
            data_manager: 数据管理器
            model_service: 模型服务
            broker_api: 经纪商API
        """
        self.trading_system = trading_system
        self.data_manager = data_manager
        self.model_service = model_service
        self.broker_api = broker_api
        
        # 注册健康检查器
        if trading_system:
            self.register_health_checker("trading_system", self._check_trading_system_health)
        if data_manager:
            self.register_health_checker("data_manager", self._check_data_manager_health)
        if model_service:
            self.register_health_checker("model_service", self._check_model_service_health)
        if broker_api:
            self.register_health_checker("broker_api", self._check_broker_api_health)
        
        self.logger.info("组件注册完成")
    
    def register_health_checker(self, name: str, checker: Callable[[], HealthCheckResult]) -> None:
        """注册健康检查器"""
        self.health_checkers[name] = checker
        self.logger.debug(f"注册健康检查器: {name}")
    
    def add_alert_callback(self, callback: Callable[[str, str, Dict[str, Any]], None]) -> None:
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    def start(self) -> None:
        """启动监控"""
        if self.running:
            self.logger.warning("监控器已在运行")
            return
        
        self.running = True
        
        # 启动指标收集线程
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        # 启动健康检查线程
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
        
        self.logger.info("系统监控已启动")
    
    def stop(self) -> None:
        """停止监控"""
        if not self.running:
            return
        
        self.running = False
        
        # 等待线程结束
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=5)
        
        self.logger.info("系统监控已停止")
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self.running:
            try:
                # 收集指标
                metrics = self._collect_metrics()
                
                # 存储指标
                self.metrics_history.append(metrics)
                
                # 检查告警
                self._check_alerts(metrics)
                
                # 清理过期数据
                self._cleanup_old_data()
                
            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
            
            time.sleep(self.collection_interval)
    
    def _health_check_loop(self) -> None:
        """健康检查循环"""
        while self.running:
            try:
                # 执行所有健康检查
                for name, checker in self.health_checkers.items():
                    try:
                        start_time = time.time()
                        result = checker()
                        result.response_time_ms = (time.time() - start_time) * 1000
                        self.health_checks[name] = result
                        
                        # 记录健康状态变化
                        if result.status != "healthy":
                            self.logger.warning(f"健康检查警告 {name}: {result.message}")
                            
                    except Exception as e:
                        self.health_checks[name] = HealthCheckResult(
                            component=name,
                            status="critical",
                            message=f"健康检查失败: {e}",
                            timestamp=datetime.now(),
                            response_time_ms=0.0
                        )
                        self.logger.error(f"健康检查异常 {name}: {e}")
                
            except Exception as e:
                self.logger.error(f"健康检查循环异常: {e}")
            
            time.sleep(60)  # 每分钟检查一次
    
    def _collect_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        try:
            # 系统资源
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 网络统计
            current_network = psutil.net_io_counters()
            
            # 进程信息
            current_process = psutil.Process()
            
            # 业务指标
            business_metrics = self._collect_business_metrics()
            
            # 性能指标
            performance_metrics = self._collect_performance_metrics()
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                network_bytes_sent=current_network.bytes_sent,
                network_bytes_recv=current_network.bytes_recv,
                process_count=len(psutil.pids()),
                thread_count=current_process.num_threads(),
                open_files=len(current_process.open_files()),
                **business_metrics,
                **performance_metrics
            )
            
        except Exception as e:
            self.logger.error(f"收集指标失败: {e}")
            # 返回默认指标
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                process_count=0,
                thread_count=0,
                open_files=0
            )
    
    def _collect_business_metrics(self) -> Dict[str, Any]:
        """收集业务指标"""
        metrics = {
            'total_trades': 0,
            'active_orders': 0,
            'portfolio_value': 0.0,
            'daily_pnl': 0.0,
            'success_rate': 0.0,
            'error_count': 0,
            'warning_count': 0
        }
        
        try:
            if self.trading_system:
                status = self.trading_system.get_system_status()
                
                metrics.update({
                    'total_trades': status['system_state'].get('total_trades_today', 0),
                    'active_orders': status['trading_status'].get('pending_orders_count', 0),
                    'daily_pnl': status['system_state'].get('daily_pnl', 0.0),
                    'success_rate': status['performance_metrics'].get('success_rate', 0.0)
                })
                
                # 获取投资组合价值
                if hasattr(self.trading_system, '_calculate_portfolio_value'):
                    metrics['portfolio_value'] = self.trading_system._calculate_portfolio_value()
        
        except Exception as e:
            self.logger.error(f"收集业务指标失败: {e}")
        
        return metrics
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """收集性能指标"""
        metrics = {
            'data_latency_ms': 0.0,
            'inference_latency_ms': 0.0,
            'order_latency_ms': 0.0
        }
        
        try:
            if self.trading_system:
                perf_metrics = self.trading_system.performance_metrics
                metrics.update({
                    'data_latency_ms': perf_metrics.data_latency_ms,
                    'inference_latency_ms': perf_metrics.inference_latency_ms,
                    'order_latency_ms': perf_metrics.order_execution_latency_ms
                })
        
        except Exception as e:
            self.logger.error(f"收集性能指标失败: {e}")
        
        return metrics
    
    def _check_alerts(self, metrics: SystemMetrics) -> None:
        """检查告警条件"""
        try:
            for metric_name, benchmark in self.benchmarks.items():
                value = getattr(metrics, metric_name, 0.0)
                
                if value >= benchmark.critical_threshold:
                    self._trigger_alert(
                        level="critical",
                        message=f"{benchmark.description}达到严重水平: {value}{benchmark.unit}",
                        details={
                            'metric': metric_name,
                            'value': value,
                            'threshold': benchmark.critical_threshold,
                            'target': benchmark.target_value
                        }
                    )
                elif value >= benchmark.warning_threshold:
                    self._trigger_alert(
                        level="warning",
                        message=f"{benchmark.description}达到警告水平: {value}{benchmark.unit}",
                        details={
                            'metric': metric_name,
                            'value': value,
                            'threshold': benchmark.warning_threshold,
                            'target': benchmark.target_value
                        }
                    )
        
        except Exception as e:
            self.logger.error(f"检查告警失败: {e}")
    
    def _trigger_alert(self, level: str, message: str, details: Dict[str, Any]) -> None:
        """触发告警"""
        self.logger.warning(f"告警触发 [{level.upper()}]: {message}")
        
        # 调用告警回调
        for callback in self.alert_callbacks:
            try:
                callback(level, message, details)
            except Exception as e:
                self.logger.error(f"告警回调异常: {e}")
    
    def _cleanup_old_data(self) -> None:
        """清理过期数据"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
            
            # 清理健康检查结果
            expired_checks = [
                name for name, result in self.health_checks.items()
                if result.timestamp < cutoff_time
            ]
            
            for name in expired_checks:
                del self.health_checks[name]
        
        except Exception as e:
            self.logger.error(f"清理过期数据失败: {e}")
    
    def _check_trading_system_health(self) -> HealthCheckResult:
        """检查交易系统健康状态"""
        try:
            if not self.trading_system:
                return HealthCheckResult(
                    component="trading_system",
                    status="critical",
                    message="交易系统未初始化",
                    timestamp=datetime.now()
                )
            
            status = self.trading_system.get_system_status()
            system_state = status['system_state']
            
            issues = []
            
            if not system_state.get('is_running', False):
                issues.append("系统未运行")
            
            if not system_state.get('data_connection_ok', False):
                issues.append("数据连接异常")
            
            if not system_state.get('model_service_ok', False):
                issues.append("模型服务异常")
            
            if not system_state.get('broker_connection_ok', False):
                issues.append("经纪商连接异常")
            
            if issues:
                return HealthCheckResult(
                    component="trading_system",
                    status="warning" if len(issues) == 1 else "critical",
                    message="; ".join(issues),
                    timestamp=datetime.now(),
                    details=system_state
                )
            else:
                return HealthCheckResult(
                    component="trading_system",
                    status="healthy",
                    message="交易系统运行正常",
                    timestamp=datetime.now(),
                    details=system_state
                )
        
        except Exception as e:
            return HealthCheckResult(
                component="trading_system",
                status="critical",
                message=f"健康检查异常: {e}",
                timestamp=datetime.now()
            )
    
    def _check_data_manager_health(self) -> HealthCheckResult:
        """检查数据管理器健康状态"""
        try:
            if not self.data_manager:
                return HealthCheckResult(
                    component="data_manager",
                    status="critical",
                    message="数据管理器未初始化",
                    timestamp=datetime.now()
                )
            
            status = self.data_manager.get_connection_status()
            
            if not status.get('running', False):
                return HealthCheckResult(
                    component="data_manager",
                    status="critical",
                    message="数据管理器未运行",
                    timestamp=datetime.now(),
                    details=status
                )
            
            # 检查数据源连接
            data_feeds = status.get('data_feeds', {})
            disconnected_feeds = [
                name for name, feed_status in data_feeds.items()
                if not feed_status.get('connected', False)
            ]
            
            if disconnected_feeds:
                return HealthCheckResult(
                    component="data_manager",
                    status="warning",
                    message=f"数据源连接异常: {', '.join(disconnected_feeds)}",
                    timestamp=datetime.now(),
                    details=status
                )
            else:
                return HealthCheckResult(
                    component="data_manager",
                    status="healthy",
                    message="数据管理器运行正常",
                    timestamp=datetime.now(),
                    details=status
                )
        
        except Exception as e:
            return HealthCheckResult(
                component="data_manager",
                status="critical",
                message=f"健康检查异常: {e}",
                timestamp=datetime.now()
            )
    
    def _check_model_service_health(self) -> HealthCheckResult:
        """检查模型服务健康状态"""
        try:
            if not self.model_service:
                return HealthCheckResult(
                    component="model_service",
                    status="critical",
                    message="模型服务未初始化",
                    timestamp=datetime.now()
                )
            
            metrics = self.model_service.get_metrics()
            
            if not metrics.get('is_running', False):
                return HealthCheckResult(
                    component="model_service",
                    status="critical",
                    message="模型服务未运行",
                    timestamp=datetime.now(),
                    details=metrics
                )
            
            # 检查成功率
            success_rate = metrics.get('success_rate', 0.0)
            if success_rate < 0.9:  # 90%成功率阈值
                return HealthCheckResult(
                    component="model_service",
                    status="warning",
                    message=f"模型服务成功率较低: {success_rate:.1%}",
                    timestamp=datetime.now(),
                    details=metrics
                )
            else:
                return HealthCheckResult(
                    component="model_service",
                    status="healthy",
                    message="模型服务运行正常",
                    timestamp=datetime.now(),
                    details=metrics
                )
        
        except Exception as e:
            return HealthCheckResult(
                component="model_service",
                status="critical",
                message=f"健康检查异常: {e}",
                timestamp=datetime.now()
            )
    
    def _check_broker_api_health(self) -> HealthCheckResult:
        """检查经纪商API健康状态"""
        try:
            if not self.broker_api:
                return HealthCheckResult(
                    component="broker_api",
                    status="critical",
                    message="经纪商API未初始化",
                    timestamp=datetime.now()
                )
            
            if not self.broker_api.is_connected():
                return HealthCheckResult(
                    component="broker_api",
                    status="critical",
                    message="经纪商API连接断开",
                    timestamp=datetime.now()
                )
            else:
                return HealthCheckResult(
                    component="broker_api",
                    status="healthy",
                    message="经纪商API连接正常",
                    timestamp=datetime.now()
                )
        
        except Exception as e:
            return HealthCheckResult(
                component="broker_api",
                status="critical",
                message=f"健康检查异常: {e}",
                timestamp=datetime.now()
            )
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """获取当前指标"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, hours: int = 24) -> List[SystemMetrics]:
        """获取指标历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            metrics for metrics in self.metrics_history
            if metrics.timestamp >= cutoff_time
        ]
    
    def get_health_status(self) -> Dict[str, HealthCheckResult]:
        """获取健康状态"""
        return self.health_checks.copy()
    
    def get_overall_health(self) -> str:
        """获取整体健康状态"""
        if not self.health_checks:
            return "unknown"
        
        statuses = [result.status for result in self.health_checks.values()]
        
        if "critical" in statuses:
            return "critical"
        elif "warning" in statuses:
            return "warning"
        else:
            return "healthy"
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.get_metrics_history(1)  # 最近1小时
        
        if not recent_metrics:
            return {}
        
        summary = {}
        
        for metric_name, benchmark in self.benchmarks.items():
            values = [getattr(m, metric_name, 0.0) for m in recent_metrics]
            
            if values:
                summary[metric_name] = {
                    'current': values[-1],
                    'average': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'target': benchmark.target_value,
                    'warning_threshold': benchmark.warning_threshold,
                    'critical_threshold': benchmark.critical_threshold,
                    'status': self._get_metric_status(values[-1], benchmark)
                }
        
        return summary
    
    def _get_metric_status(self, value: float, benchmark: PerformanceBenchmark) -> str:
        """获取指标状态"""
        if value >= benchmark.critical_threshold:
            return "critical"
        elif value >= benchmark.warning_threshold:
            return "warning"
        else:
            return "healthy"
    
    def export_metrics_to_csv(self, filename: str, hours: int = 24) -> None:
        """导出指标到CSV"""
        try:
            metrics_data = self.get_metrics_history(hours)
            
            if not metrics_data:
                self.logger.warning("没有指标数据可导出")
                return
            
            # 转换为DataFrame
            data = [metrics.to_dict() for metrics in metrics_data]
            df = pd.DataFrame(data)
            
            # 导出CSV
            df.to_csv(filename, index=False, encoding='utf-8')
            
            self.logger.info(f"指标已导出到: {filename}, 共 {len(metrics_data)} 条记录")
            
        except Exception as e:
            self.logger.error(f"导出指标失败: {e}")
            raise
    
    def generate_health_report(self) -> Dict[str, Any]:
        """生成健康报告"""
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_health': self.get_overall_health(),
            'health_checks': {
                name: {
                    'status': result.status,
                    'message': result.message,
                    'response_time_ms': result.response_time_ms,
                    'timestamp': result.timestamp.isoformat()
                }
                for name, result in self.health_checks.items()
            },
            'performance_summary': self.get_performance_summary(),
            'current_metrics': self.get_current_metrics().to_dict() if self.get_current_metrics() else {},
            'system_info': {
                'collection_interval': self.collection_interval,
                'retention_hours': self.retention_hours,
                'metrics_count': len(self.metrics_history),
                'running': self.running
            }
        }