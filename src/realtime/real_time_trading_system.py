"""
实时交易系统核心
整合数据管理、模型推理、订单管理和风险控制

主要功能:
1. 系统初始化和组件连接
2. 实时市场数据处理
3. 模型推理和信号生成
4. 风险控制和订单管理
5. 交易执行和状态监控
6. 性能监控和错误处理
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from queue import Queue, Empty

from ..utils.logger import setup_logger, get_default_log_file
from ..utils.config import Config
from ..risk.risk_manager import RiskManager
from .real_time_data_manager import RealTimeDataManager, MarketData
from .model_inference_service import ModelInferenceService
from .order_manager import OrderManager, Order
from .broker_api import BrokerAPI


class TradingMode(Enum):
    """交易模式"""
    LIVE = "live"          # 实盘交易
    SIMULATION = "simulation"  # 模拟交易
    PAPER = "paper"        # 纸上交易


@dataclass
class TradingSignal:
    """交易信号"""
    symbol: str
    action: float  # -1 到 1 之间
    confidence: float  # 0 到 1 之间
    target_position: float
    current_position: float
    timestamp: datetime
    features_used: Optional[Dict[str, Any]] = None
    model_version: str = "unknown"


@dataclass
class SystemState:
    """系统状态"""
    is_running: bool = False
    is_trading_active: bool = False
    last_heartbeat: Optional[datetime] = None
    data_connection_ok: bool = False
    model_service_ok: bool = False
    broker_connection_ok: bool = False
    total_trades_today: int = 0
    current_pnl: float = 0.0
    daily_pnl: float = 0.0
    system_start_time: Optional[datetime] = None
    last_error: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """性能指标"""
    data_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    order_execution_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    signals_processed: int = 0
    orders_executed: int = 0
    orders_failed: int = 0
    last_update: Optional[datetime] = None


class RealTimeTradingSystem:
    """
    实时交易系统核心
    
    整合所有组件，提供统一的实时交易接口:
    - 数据接收和处理
    - 模型推理和信号生成
    - 风险控制和订单管理
    - 交易执行和状态监控
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        trading_mode: TradingMode = TradingMode.SIMULATION,
        initial_capital: float = 100000.0
    ):
        """
        初始化实时交易系统
        
        Args:
            config: 配置对象
            trading_mode: 交易模式
            initial_capital: 初始资金
        """
        self.config = config or Config()
        self.trading_mode = trading_mode
        self.initial_capital = initial_capital
        
        # 初始化日志
        self.logger = setup_logger(
            name="RealTimeTradingSystem",
            level="INFO",
            log_file=get_default_log_file("real_time_trading_system")
        )
        
        # 系统组件
        self.data_manager: Optional[RealTimeDataManager] = None
        self.model_service: Optional[ModelInferenceService] = None
        self.risk_manager: Optional[RiskManager] = None
        self.order_manager: Optional[OrderManager] = None
        self.broker_api: Optional[BrokerAPI] = None
        
        # 系统状态
        self.system_state = SystemState()
        self.performance_metrics = PerformanceMetrics()
        
        # 交易状态
        self.current_positions: Dict[str, float] = {}  # symbol -> position ratio
        self.pending_orders: Dict[str, Order] = {}     # order_id -> Order
        self.trade_history: List[Dict[str, Any]] = []
        
        # 事件处理
        self.signal_queue = Queue(maxsize=1000)
        self.signal_processor_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        
        # 回调函数
        self.signal_callbacks: List[Callable[[TradingSignal], None]] = []
        self.trade_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.error_callbacks: List[Callable[[str, Exception], None]] = []
        
        # 配置参数
        self.min_signal_strength = 0.1    # 最小信号强度
        self.max_position_size = 0.2       # 最大单个资产仓位
        self.signal_timeout_seconds = 30   # 信号超时时间
        
        self.logger.info(f"RealTimeTradingSystem 初始化完成 - 模式: {trading_mode.value}")
    
    def initialize_components(
        self,
        symbols: List[str],
        model_path: str,
        broker_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        初始化系统组件
        
        Args:
            symbols: 交易标的列表
            model_path: 模型文件路径
            broker_config: 经纪商配置
        """
        try:
            self.logger.info("开始初始化系统组件...")
            
            # 1. 初始化数据管理器
            self.data_manager = RealTimeDataManager(
                buffer_size=self.config.data.get('buffer_size', 10000),
                max_latency_ms=self.config.data.get('max_latency_ms', 100.0)
            )
            
            # 添加数据回调
            self.data_manager.add_data_callback(self._on_market_data_received)
            self.data_manager.add_error_callback(self._on_data_error)
            
            # 2. 初始化模型推理服务
            self.model_service = ModelInferenceService(
                model_path=model_path,
                config=self.config,
                use_ray_serve=True,
                batch_size=1,  # 实时交易使用单个预测
                max_batch_wait_ms=10
            )
            
            # 3. 初始化风险管理器
            self.risk_manager = RiskManager(config=self.config)
            
            # 4. 初始化订单管理器
            self.order_manager = OrderManager(
                config=self.config,
                trading_mode=self.trading_mode
            )
            
            # 5. 初始化经纪商接口
            if self.trading_mode == TradingMode.LIVE:
                # 实盘交易需要真实的经纪商API
                if broker_config is None:
                    raise ValueError("实盘交易模式需要提供经纪商配置")
                # TODO: 根据broker_config创建真实的经纪商API
                from .broker_api import SimulatedBroker
                self.broker_api = SimulatedBroker(initial_capital=self.initial_capital)
            else:
                # 模拟交易使用模拟经纪商
                from .broker_api import SimulatedBroker
                self.broker_api = SimulatedBroker(initial_capital=self.initial_capital)
            
            # 6. 初始化持仓
            for symbol in symbols:
                self.current_positions[symbol] = 0.0
            
            self.logger.info("系统组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise
    
    async def start_system(self, symbols: List[str], data_feed_config: Dict[str, Any]) -> None:
        """
        启动系统
        
        Args:
            symbols: 交易标的列表
            data_feed_config: 数据源配置
        """
        try:
            self.logger.info("启动实时交易系统...")
            
            self.system_state.system_start_time = datetime.now()
            self.system_state.is_running = True
            
            # 1. 启动模型推理服务
            self.logger.info("启动模型推理服务...")
            if self.model_service:
                self.model_service.start()
                self.system_state.model_service_ok = True
            
            # 2. 启动数据管理器
            self.logger.info("启动数据连接...")
            if self.data_manager:
                # 添加数据源
                from .real_time_data_manager import DataFeed
                for name, config in data_feed_config.items():
                    data_feed = DataFeed(
                        name=name,
                        url=config['url'],
                        symbols=symbols,
                        reconnect_interval=config.get('reconnect_interval', 5),
                        max_reconnect_attempts=config.get('max_reconnect_attempts', 10),
                        heartbeat_interval=config.get('heartbeat_interval', 30)
                    )
                    self.data_manager.add_data_feed(data_feed)
                
                self.data_manager.start()
                # 等待连接建立
                await asyncio.sleep(2)
                self.system_state.data_connection_ok = True
            
            # 3. 连接经纪商
            self.logger.info("连接经纪商...")
            if self.broker_api:
                await self.broker_api.connect()
                self.system_state.broker_connection_ok = True
            
            # 4. 启动信号处理线程
            self.logger.info("启动信号处理...")
            self._start_signal_processor()
            
            # 5. 启动心跳监控
            self._start_heartbeat_monitor()
            
            # 6. 验证系统状态
            if self._verify_system_ready():
                self.logger.info("系统启动成功，准备开始交易")
            else:
                raise RuntimeError("系统未能正确启动")
                
        except Exception as e:
            self.logger.error(f"系统启动失败: {e}")
            await self.stop_system()
            raise
    
    def _verify_system_ready(self) -> bool:
        """验证系统就绪状态"""
        checks = {
            "数据连接": self.system_state.data_connection_ok,
            "模型服务": self.system_state.model_service_ok,
            "经纪商连接": self.system_state.broker_connection_ok,
            "风险管理": self.risk_manager is not None,
            "订单管理": self.order_manager is not None
        }
        
        all_ready = True
        for check_name, status in checks.items():
            if not status:
                self.logger.error(f"系统检查失败: {check_name}")
                all_ready = False
            else:
                self.logger.info(f"✓ {check_name}")
        
        return all_ready
    
    async def _on_market_data_received(self, market_data: MarketData) -> None:
        """
        市场数据接收回调
        
        Args:
            market_data: 市场数据
        """
        if not self.system_state.is_trading_active:
            return
        
        start_time = time.time()
        
        try:
            # 更新数据延迟统计
            data_latency = (datetime.now() - market_data.timestamp).total_seconds() * 1000
            self.performance_metrics.data_latency_ms = data_latency
            
            # 检查数据有效性
            if not self._is_valid_market_data(market_data):
                return
            
            # 获取历史数据窗口
            window_data = self._get_feature_window(market_data.symbol)
            if window_data is None:
                return
            
            # 模型推理
            inference_start = time.time()
            prediction = await self._get_model_prediction(window_data, market_data.symbol)
            if prediction is None:
                return
            
            inference_time = (time.time() - inference_start) * 1000
            self.performance_metrics.inference_latency_ms = inference_time
            
            # 生成交易信号
            signal = self._create_trading_signal(prediction, market_data)
            if signal is None:
                return
            
            # 添加到信号队列
            try:
                self.signal_queue.put_nowait(signal)
                self.performance_metrics.signals_processed += 1
            except:
                self.logger.warning("信号队列已满，丢弃信号")
            
            # 更新性能指标
            total_time = (time.time() - start_time) * 1000
            self.performance_metrics.total_latency_ms = total_time
            self.performance_metrics.last_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"处理市场数据失败: {e}")
            self._handle_error("market_data_processing", e)
    
    def _is_valid_market_data(self, market_data: MarketData) -> bool:
        """验证市场数据有效性"""
        if market_data.price <= 0:
            return False
        if market_data.volume < 0:
            return False
        
        # 检查价格变化是否异常
        if market_data.symbol in self.data_manager.latest_data:
            last_price = self.data_manager.latest_data[market_data.symbol].price
            price_change = abs(market_data.price - last_price) / last_price
            if price_change > 0.1:  # 10%的价格变化阈值
                self.logger.warning(f"异常价格变化 {market_data.symbol}: {price_change:.2%}")
                return False
        
        return True
    
    def _get_feature_window(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取特征计算窗口数据"""
        try:
            # 从数据管理器获取历史数据
            window_size = self.config.trading.get('window_size', 50)
            historical_data = self.data_manager.get_data_frame(symbol, window_size)
            
            if len(historical_data) < window_size:
                return None
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"获取特征窗口失败: {e}")
            return None
    
    async def _get_model_prediction(self, window_data: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
        """获取模型预测"""
        try:
            if self.model_service is None:
                return None
            
            # 预处理数据为特征向量
            # 这里简化处理，实际需要与模型训练时的特征工程保持一致
            features = window_data[['Close', 'Volume']].values.flatten()[-20:]  # 简化特征
            
            # 异步模型推理
            result = await self.model_service.predict_async(features, symbol)
            
            return {
                'action': result.action,
                'confidence': result.confidence,
                'model_version': result.model_version,
                'inference_time_ms': result.inference_time_ms
            }
            
        except Exception as e:
            self.logger.error(f"模型预测失败: {e}")
            return None
    
    def _create_trading_signal(self, prediction: Dict[str, Any], market_data: MarketData) -> Optional[TradingSignal]:
        """创建交易信号"""
        try:
            action = prediction['action']
            confidence = prediction['confidence']
            
            # 过滤弱信号
            signal_strength = abs(action) * confidence
            if signal_strength < self.min_signal_strength:
                return None
            
            # 获取当前仓位
            current_position = self.current_positions.get(market_data.symbol, 0.0)
            
            # 计算目标仓位
            target_position = action * confidence
            target_position = np.clip(target_position, -self.max_position_size, self.max_position_size)
            
            signal = TradingSignal(
                symbol=market_data.symbol,
                action=action,
                confidence=confidence,
                target_position=target_position,
                current_position=current_position,
                timestamp=datetime.now(),
                model_version=prediction.get('model_version', 'unknown')
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"创建交易信号失败: {e}")
            return None
    
    def _start_signal_processor(self) -> None:
        """启动信号处理线程"""
        self.signal_processor_thread = threading.Thread(
            target=self._signal_processing_loop,
            daemon=True
        )
        self.signal_processor_thread.start()
        self.logger.info("信号处理线程已启动")
    
    def _signal_processing_loop(self) -> None:
        """信号处理循环"""
        while self.system_state.is_running:
            try:
                # 获取信号
                signal = self.signal_queue.get(timeout=1.0)
                
                # 处理信号
                asyncio.run(self._process_trading_signal(signal))
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"信号处理错误: {e}")
    
    async def _process_trading_signal(self, signal: TradingSignal) -> None:
        """
        处理交易信号
        
        Args:
            signal: 交易信号
        """
        try:
            self.logger.debug(f"处理信号: {signal.symbol} {signal.action:.3f} conf={signal.confidence:.3f}")
            
            # 触发信号回调
            for callback in self.signal_callbacks:
                try:
                    callback(signal)
                except Exception as e:
                    self.logger.error(f"信号回调错误: {e}")
            
            # 风险控制
            adjusted_signal = self._apply_risk_controls(signal)
            if adjusted_signal is None:
                return
            
            # 生成订单
            order = self._generate_order(adjusted_signal)
            if order is None:
                return
            
            # 执行订单
            await self._execute_order(order)
            
        except Exception as e:
            self.logger.error(f"处理交易信号失败: {e}")
            self._handle_error("signal_processing", e)
    
    def _apply_risk_controls(self, signal: TradingSignal) -> Optional[TradingSignal]:
        """应用风险控制"""
        try:
            if self.risk_manager is None:
                return signal
            
            # 更新投资组合状态
            total_value = self._calculate_portfolio_value()
            positions = {k: v * total_value for k, v in self.current_positions.items()}
            cash = total_value - sum(positions.values())
            
            self.risk_manager.update_portfolio_state(
                total_value=total_value,
                cash=cash,
                positions=positions,
                timestamp=datetime.now()
            )
            
            # 应用风险控制
            proposed_action = {
                'symbol': signal.symbol,
                'target_position': signal.target_position,
                'current_position': signal.current_position
            }
            
            adjusted_action, warnings = self.risk_manager.apply_risk_controls(proposed_action)
            
            # 记录风险警告
            if warnings:
                for warning in warnings:
                    self.logger.warning(f"风险控制: {warning}")
            
            # 创建调整后的信号
            if 'target_position' in adjusted_action:
                signal.target_position = adjusted_action['target_position']
            
            return signal
            
        except Exception as e:
            self.logger.error(f"风险控制失败: {e}")
            return None
    
    def _generate_order(self, signal: TradingSignal) -> Optional[Order]:
        """生成订单"""
        try:
            if self.order_manager is None:
                return None
            
            # 计算仓位变化
            position_change = signal.target_position - signal.current_position
            
            if abs(position_change) < 0.01:  # 最小交易阈值
                return None
            
            # 获取当前价格
            latest_data = self.data_manager.get_latest_data(signal.symbol)
            if latest_data is None:
                return None
            
            current_price = latest_data.price
            
            # 计算交易数量
            portfolio_value = self._calculate_portfolio_value()
            trade_value = abs(position_change) * portfolio_value
            quantity = int(trade_value / current_price)
            
            if quantity == 0:
                return None
            
            # 创建订单
            order = self.order_manager.create_order(
                symbol=signal.symbol,
                side='BUY' if position_change > 0 else 'SELL',
                quantity=quantity,
                order_type='LIMIT',
                price=current_price,
                time_in_force='IOC'
            )
            
            return order
            
        except Exception as e:
            self.logger.error(f"生成订单失败: {e}")
            return None
    
    async def _execute_order(self, order: Order) -> None:
        """执行订单"""
        try:
            execution_start = time.time()
            
            if self.broker_api is None or self.order_manager is None:
                return
            
            # 验证订单
            if not self.order_manager.validate_order(order):
                self.logger.warning(f"订单验证失败: {order.order_id}")
                return
            
            self.logger.info(f"执行订单: {order.side} {order.quantity} {order.symbol} @ {order.price}")
            
            # 提交订单
            order_id = await self.broker_api.submit_order(order)
            order.order_id = order_id
            order.status = "SUBMITTED"
            
            # 添加到待处理订单
            self.pending_orders[order_id] = order
            
            # 监控订单执行（简化版本）
            try:
                execution_result = await self.broker_api.wait_for_execution(order_id, timeout=5)
                
                if execution_result['status'] == 'FILLED':
                    # 更新仓位
                    position_change = order.quantity if order.side == 'BUY' else -order.quantity
                    portfolio_value = self._calculate_portfolio_value()
                    position_ratio_change = (position_change * order.price) / portfolio_value
                    
                    self.current_positions[order.symbol] += position_ratio_change
                    
                    # 记录交易
                    trade_record = {
                        'timestamp': datetime.now(),
                        'order_id': order_id,
                        'symbol': order.symbol,
                        'side': order.side,
                        'quantity': execution_result['executed_quantity'],
                        'price': execution_result['executed_price'],
                        'value': execution_result['executed_quantity'] * execution_result['executed_price']
                    }
                    
                    self.trade_history.append(trade_record)
                    self.system_state.total_trades_today += 1
                    
                    # 触发交易回调
                    for callback in self.trade_callbacks:
                        try:
                            callback(trade_record)
                        except Exception as e:
                            self.logger.error(f"交易回调错误: {e}")
                    
                    self.logger.info(f"订单执行成功: {trade_record}")
                    self.performance_metrics.orders_executed += 1
                else:
                    self.logger.warning(f"订单执行失败: {execution_result}")
                    self.performance_metrics.orders_failed += 1
                
                # 移除待处理订单
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"订单执行超时: {order_id}")
                self.performance_metrics.orders_failed += 1
            
            # 更新执行延迟统计
            execution_time = (time.time() - execution_start) * 1000
            self.performance_metrics.order_execution_latency_ms = execution_time
            
        except Exception as e:
            self.logger.error(f"订单执行失败: {e}")
            self.performance_metrics.orders_failed += 1
            self._handle_error("order_execution", e)
    
    def _calculate_portfolio_value(self) -> float:
        """计算投资组合总价值"""
        try:
            if self.broker_api is None:
                return self.initial_capital
            
            # 从经纪商获取账户信息
            account_info = self.broker_api.get_account_info()
            return account_info.get('total_value', self.initial_capital)
            
        except Exception as e:
            self.logger.error(f"计算投资组合价值失败: {e}")
            return self.initial_capital
    
    def _start_heartbeat_monitor(self) -> None:
        """启动心跳监控"""
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self.heartbeat_thread.start()
        self.logger.info("心跳监控已启动")
    
    def _heartbeat_loop(self) -> None:
        """心跳监控循环"""
        while self.system_state.is_running:
            try:
                self.system_state.last_heartbeat = datetime.now()
                
                # 检查组件状态
                self._check_component_health()
                
                # 更新系统状态
                self._update_system_metrics()
                
                time.sleep(5)  # 每5秒检查一次
                
            except Exception as e:
                self.logger.error(f"心跳监控错误: {e}")
    
    def _check_component_health(self) -> None:
        """检查组件健康状态"""
        # 检查数据连接
        if self.data_manager:
            status = self.data_manager.get_connection_status()
            self.system_state.data_connection_ok = status.get('running', False)
        
        # 检查模型服务
        if self.model_service:
            metrics = self.model_service.get_metrics()
            self.system_state.model_service_ok = metrics.get('is_running', False)
        
        # 检查经纪商连接
        if self.broker_api:
            self.system_state.broker_connection_ok = self.broker_api.is_connected()
    
    def _update_system_metrics(self) -> None:
        """更新系统指标"""
        try:
            # 计算当前PnL
            current_value = self._calculate_portfolio_value()
            self.system_state.current_pnl = current_value - self.initial_capital
            
            # 计算日PnL (简化版本)
            # 实际应用中应该基于当日开始时的价值计算
            self.system_state.daily_pnl = self.system_state.current_pnl
            
        except Exception as e:
            self.logger.error(f"更新系统指标失败: {e}")
    
    def _on_data_error(self, feed_name: str, error: Exception) -> None:
        """数据错误回调"""
        self.logger.error(f"数据源错误 {feed_name}: {error}")
        self.system_state.data_connection_ok = False
        self._handle_error("data_connection", error)
    
    def _handle_error(self, error_type: str, error: Exception) -> None:
        """处理系统错误"""
        error_msg = f"{error_type}: {str(error)}"
        self.system_state.last_error = error_msg
        
        # 触发错误回调
        for callback in self.error_callbacks:
            try:
                callback(error_type, error)
            except Exception as e:
                self.logger.error(f"错误回调失败: {e}")
    
    def start_trading(self) -> None:
        """启动交易"""
        if not self._verify_system_ready():
            raise RuntimeError("系统未准备就绪，无法启动交易")
        
        self.system_state.is_trading_active = True
        self.logger.info("实时交易已启动")
    
    def stop_trading(self) -> None:
        """停止交易"""
        self.system_state.is_trading_active = False
        self.logger.info("实时交易已停止")
    
    async def stop_system(self) -> None:
        """停止系统"""
        try:
            self.logger.info("停止实时交易系统...")
            
            # 停止交易
            self.stop_trading()
            
            # 设置停止标志
            self.system_state.is_running = False
            
            # 等待线程结束
            if self.signal_processor_thread and self.signal_processor_thread.is_alive():
                self.signal_processor_thread.join(timeout=5)
            
            if self.heartbeat_thread and self.heartbeat_thread.is_alive():
                self.heartbeat_thread.join(timeout=5)
            
            # 停止组件
            if self.data_manager:
                self.data_manager.stop()
            
            if self.model_service:
                self.model_service.stop()
            
            if self.broker_api:
                await self.broker_api.disconnect()
            
            self.logger.info("实时交易系统已停止")
            
        except Exception as e:
            self.logger.error(f"停止系统失败: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'system_state': {
                'is_running': self.system_state.is_running,
                'is_trading_active': self.system_state.is_trading_active,
                'last_heartbeat': self.system_state.last_heartbeat.isoformat() if self.system_state.last_heartbeat else None,
                'data_connection_ok': self.system_state.data_connection_ok,
                'model_service_ok': self.system_state.model_service_ok,
                'broker_connection_ok': self.system_state.broker_connection_ok,
                'total_trades_today': self.system_state.total_trades_today,
                'current_pnl': self.system_state.current_pnl,
                'daily_pnl': self.system_state.daily_pnl,
                'last_error': self.system_state.last_error
            },
            'performance_metrics': {
                'data_latency_ms': self.performance_metrics.data_latency_ms,
                'inference_latency_ms': self.performance_metrics.inference_latency_ms,
                'order_execution_latency_ms': self.performance_metrics.order_execution_latency_ms,
                'total_latency_ms': self.performance_metrics.total_latency_ms,
                'signals_processed': self.performance_metrics.signals_processed,
                'orders_executed': self.performance_metrics.orders_executed,
                'orders_failed': self.performance_metrics.orders_failed,
                'success_rate': (
                    self.performance_metrics.orders_executed / 
                    max(1, self.performance_metrics.orders_executed + self.performance_metrics.orders_failed)
                )
            },
            'trading_status': {
                'current_positions': self.current_positions.copy(),
                'pending_orders_count': len(self.pending_orders),
                'trade_history_count': len(self.trade_history),
                'trading_mode': self.trading_mode.value
            }
        }
    
    def add_signal_callback(self, callback: Callable[[TradingSignal], None]) -> None:
        """添加信号回调"""
        self.signal_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """添加交易回调"""
        self.trade_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str, Exception], None]) -> None:
        """添加错误回调"""
        self.error_callbacks.append(callback)
    
    def cleanup(self) -> None:
        """清理资源"""
        asyncio.run(self.stop_system())
        
        # 清理数据
        self.signal_queue = Queue(maxsize=1000)
        self.current_positions.clear()
        self.pending_orders.clear()
        self.trade_history.clear()
        
        # 清理回调
        self.signal_callbacks.clear()
        self.trade_callbacks.clear()
        self.error_callbacks.clear()
        
        self.logger.info("实时交易系统资源已清理")