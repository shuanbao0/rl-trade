"""
实时数据管理组件
处理实时市场数据接收、缓存和分发

主要功能:
1. WebSocket数据连接管理
2. 数据缓冲区和队列管理
3. 数据验证和清洗
4. 重连机制和错误恢复
5. 数据回调和事件分发
6. 性能监控和统计
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from collections import deque
from queue import Queue, Empty
import pandas as pd
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from ..utils.logger import setup_logger, get_default_log_file
from ..utils.config import Config


@dataclass
class MarketData:
    """市场数据结构"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close
        }


@dataclass
class DataFeed:
    """数据源配置"""
    name: str
    url: str
    symbols: List[str]
    reconnect_interval: int = 5
    max_reconnect_attempts: int = 10
    heartbeat_interval: int = 30
    buffer_size: int = 1000
    enabled: bool = True


@dataclass
class ConnectionStats:
    """连接统计信息"""
    connected: bool = False
    connect_time: Optional[datetime] = None
    disconnect_time: Optional[datetime] = None
    reconnect_count: int = 0
    messages_received: int = 0
    messages_processed: int = 0
    last_message_time: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    
    def reset(self):
        """重置统计"""
        self.messages_received = 0
        self.messages_processed = 0
        self.errors.clear()


class RealTimeDataManager:
    """
    实时数据管理器
    
    管理多个数据源的实时数据接收和分发:
    - WebSocket连接管理
    - 数据缓冲和队列
    - 回调机制
    - 错误处理和重连
    - 性能监控
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        buffer_size: int = 10000,
        max_latency_ms: float = 100.0
    ):
        """
        初始化实时数据管理器
        
        Args:
            config: 配置对象
            buffer_size: 数据缓冲区大小
            max_latency_ms: 最大允许延迟(毫秒)
        """
        self.config = config or Config()
        self.buffer_size = buffer_size
        self.max_latency_ms = max_latency_ms
        
        # 初始化日志
        self.logger = setup_logger(
            name="RealTimeDataManager",
            level="INFO",
            log_file=get_default_log_file("real_time_data_manager")
        )
        
        # 数据源配置
        self.data_feeds: Dict[str, DataFeed] = {}
        
        # 连接管理
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.connection_stats: Dict[str, ConnectionStats] = {}
        
        # 数据缓冲
        self.data_buffer: Dict[str, deque] = {}  # symbol -> deque of MarketData
        self.latest_data: Dict[str, MarketData] = {}  # symbol -> latest MarketData
        
        # 事件队列
        self.event_queue = Queue(maxsize=buffer_size)
        
        # 回调函数
        self.data_callbacks: List[Callable[[MarketData], None]] = []
        self.error_callbacks: List[Callable[[str, Exception], None]] = []
        
        # 控制状态
        self.running = False
        self.threads: List[threading.Thread] = []
        
        # 性能统计
        self.performance_stats = {
            'total_messages': 0,
            'messages_per_second': 0.0,
            'average_latency_ms': 0.0,
            'max_latency_ms': 0.0,
            'dropped_messages': 0,
            'start_time': None
        }
        
        self.logger.info("RealTimeDataManager 初始化完成")
    
    def add_data_feed(self, data_feed: DataFeed) -> None:
        """
        添加数据源
        
        Args:
            data_feed: 数据源配置
        """
        self.data_feeds[data_feed.name] = data_feed
        self.connection_stats[data_feed.name] = ConnectionStats()
        
        # 为每个symbol初始化缓冲区
        for symbol in data_feed.symbols:
            if symbol not in self.data_buffer:
                self.data_buffer[symbol] = deque(maxlen=self.buffer_size)
        
        self.logger.info(f"添加数据源: {data_feed.name}, 订阅: {data_feed.symbols}")
    
    def add_data_callback(self, callback: Callable[[MarketData], None]) -> None:
        """
        添加数据回调函数
        
        Args:
            callback: 回调函数，接收MarketData参数
        """
        self.data_callbacks.append(callback)
        callback_name = getattr(callback, '__name__', str(callback))
        self.logger.debug(f"添加数据回调: {callback_name}")
    
    def add_error_callback(self, callback: Callable[[str, Exception], None]) -> None:
        """
        添加错误回调函数
        
        Args:
            callback: 错误回调函数，接收(data_feed_name, exception)参数
        """
        self.error_callbacks.append(callback)
        callback_name = getattr(callback, '__name__', str(callback))
        self.logger.debug(f"添加错误回调: {callback_name}")
    
    async def connect_websocket(self, data_feed: DataFeed) -> None:
        """
        连接WebSocket数据源
        
        Args:
            data_feed: 数据源配置
        """
        feed_name = data_feed.name
        stats = self.connection_stats[feed_name]
        
        retry_count = 0
        
        while self.running and retry_count < data_feed.max_reconnect_attempts:
            try:
                self.logger.info(f"连接数据源: {feed_name} -> {data_feed.url}")
                
                async with websockets.connect(
                    data_feed.url,
                    ping_interval=data_feed.heartbeat_interval,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    
                    self.connections[feed_name] = websocket
                    stats.connected = True
                    stats.connect_time = datetime.now()
                    stats.reconnect_count = retry_count
                    
                    self.logger.info(f"数据源 {feed_name} 连接成功")
                    
                    # 订阅symbols
                    await self.subscribe_symbols(websocket, data_feed.symbols)
                    
                    # 开始接收数据
                    await self.receive_data(websocket, data_feed)
                    
            except (ConnectionClosed, WebSocketException) as e:
                stats.connected = False
                stats.disconnect_time = datetime.now()
                error_msg = f"WebSocket连接异常: {e}"
                stats.errors.append(error_msg)
                self.logger.warning(error_msg)
                
                # 通知错误回调
                for callback in self.error_callbacks:
                    try:
                        callback(feed_name, e)
                    except Exception as cb_error:
                        self.logger.error(f"错误回调异常: {cb_error}")
                
            except Exception as e:
                stats.connected = False
                error_msg = f"连接异常: {e}"
                stats.errors.append(error_msg)
                self.logger.error(error_msg)
                
                for callback in self.error_callbacks:
                    try:
                        callback(feed_name, e)
                    except Exception as cb_error:
                        self.logger.error(f"错误回调异常: {cb_error}")
            
            # 重连延迟
            if self.running and retry_count < data_feed.max_reconnect_attempts - 1:
                retry_count += 1
                await asyncio.sleep(data_feed.reconnect_interval)
                self.logger.info(f"尝试重连 {feed_name} (第{retry_count}次)")
        
        if retry_count >= data_feed.max_reconnect_attempts:
            self.logger.error(f"数据源 {feed_name} 重连次数超限，停止重连")
    
    async def subscribe_symbols(self, websocket, symbols: List[str]) -> None:
        """
        订阅交易对
        
        Args:
            websocket: WebSocket连接
            symbols: 交易对列表
        """
        # 示例订阅消息格式（需要根据实际API调整）
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [f"{symbol.lower()}@ticker" for symbol in symbols],
            "id": int(time.time())
        }
        
        await websocket.send(json.dumps(subscribe_msg))
        self.logger.info(f"订阅消息已发送: {symbols}")
    
    async def receive_data(self, websocket, data_feed: DataFeed) -> None:
        """
        接收数据循环
        
        Args:
            websocket: WebSocket连接
            data_feed: 数据源配置
        """
        feed_name = data_feed.name
        stats = self.connection_stats[feed_name]
        
        while self.running:
            try:
                # 接收消息
                message = await websocket.recv()
                receive_time = datetime.now()
                
                stats.messages_received += 1
                stats.last_message_time = receive_time
                
                # 解析数据
                market_data = self.parse_message(message, receive_time)
                
                if market_data:
                    # 检查延迟
                    latency_ms = (receive_time - market_data.timestamp).total_seconds() * 1000
                    
                    if latency_ms <= self.max_latency_ms:
                        # 处理数据
                        self.process_market_data(market_data)
                        stats.messages_processed += 1
                        
                        # 更新性能统计
                        self.update_performance_stats(latency_ms)
                    else:
                        self.performance_stats['dropped_messages'] += 1
                        self.logger.warning(f"数据延迟过高，丢弃: {latency_ms:.2f}ms > {self.max_latency_ms}ms")
                
            except ConnectionClosed:
                self.logger.warning(f"WebSocket连接 {feed_name} 已关闭")
                break
            except Exception as e:
                error_msg = f"接收数据异常: {e}"
                stats.errors.append(error_msg)
                self.logger.error(error_msg)
                break
    
    def parse_message(self, message: str, receive_time: datetime) -> Optional[MarketData]:
        """
        解析WebSocket消息
        
        Args:
            message: 原始消息
            receive_time: 接收时间
            
        Returns:
            Optional[MarketData]: 解析后的市场数据
        """
        try:
            data = json.loads(message)
            
            # 示例解析逻辑（需要根据实际API格式调整）
            if 'stream' in data and 'data' in data:
                stream_data = data['data']
                
                # 解析ticker数据
                if '@ticker' in data['stream']:
                    symbol = stream_data.get('s', '').upper()
                    
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(stream_data.get('E', 0) / 1000),
                        price=float(stream_data.get('c', 0)),
                        volume=float(stream_data.get('v', 0)),
                        bid=float(stream_data.get('b', 0)) if stream_data.get('b') else None,
                        ask=float(stream_data.get('a', 0)) if stream_data.get('a') else None,
                        open=float(stream_data.get('o', 0)) if stream_data.get('o') else None,
                        high=float(stream_data.get('h', 0)) if stream_data.get('h') else None,
                        low=float(stream_data.get('l', 0)) if stream_data.get('l') else None,
                        close=float(stream_data.get('c', 0)) if stream_data.get('c') else None
                    )
                    
                    return market_data
            
            # 如果是ping/pong或其他非数据消息，返回None
            return None
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"消息解析失败: {e}, 消息: {message[:200]}")
            return None
    
    def process_market_data(self, market_data: MarketData) -> None:
        """
        处理市场数据
        
        Args:
            market_data: 市场数据
        """
        symbol = market_data.symbol
        
        # 确保为symbol初始化缓冲区
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = deque(maxlen=self.buffer_size)
        
        # 更新缓冲区
        self.data_buffer[symbol].append(market_data)
        
        # 更新最新数据
        self.latest_data[symbol] = market_data
        
        # 添加到事件队列
        try:
            self.event_queue.put_nowait(market_data)
        except:
            # 队列满了，移除最旧的数据
            try:
                self.event_queue.get_nowait()
                self.event_queue.put_nowait(market_data)
            except Empty:
                pass
        
        # 触发回调
        for callback in self.data_callbacks:
            try:
                callback(market_data)
            except Exception as e:
                self.logger.error(f"数据回调异常: {e}")
    
    def update_performance_stats(self, latency_ms: float) -> None:
        """
        更新性能统计
        
        Args:
            latency_ms: 延迟时间(毫秒)
        """
        stats = self.performance_stats
        stats['total_messages'] += 1
        
        # 更新延迟统计
        if stats['average_latency_ms'] == 0:
            stats['average_latency_ms'] = latency_ms
        else:
            # 指数移动平均
            alpha = 0.1
            stats['average_latency_ms'] = (1 - alpha) * stats['average_latency_ms'] + alpha * latency_ms
        
        if latency_ms > stats['max_latency_ms']:
            stats['max_latency_ms'] = latency_ms
        
        # 计算消息速率
        if stats['start_time']:
            elapsed = (datetime.now() - stats['start_time']).total_seconds()
            if elapsed > 0:
                stats['messages_per_second'] = stats['total_messages'] / elapsed
    
    def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """
        获取指定symbol的最新数据
        
        Args:
            symbol: 交易对符号
            
        Returns:
            Optional[MarketData]: 最新市场数据
        """
        return self.latest_data.get(symbol.upper())
    
    def get_historical_data(self, symbol: str, count: int = 100) -> List[MarketData]:
        """
        获取历史数据
        
        Args:
            symbol: 交易对符号
            count: 获取数量
            
        Returns:
            List[MarketData]: 历史数据列表
        """
        symbol = symbol.upper()
        if symbol in self.data_buffer:
            buffer = self.data_buffer[symbol]
            return list(buffer)[-count:] if len(buffer) >= count else list(buffer)
        return []
    
    def get_data_frame(self, symbol: str, count: int = 100, window_size: int = None) -> pd.DataFrame:
        """
        获取DataFrame格式的历史数据
        
        Args:
            symbol: 交易对符号
            count: 获取数量
            window_size: 窗口大小（与count等价，用于兼容性）
            
        Returns:
            pd.DataFrame: 数据框
        """
        # 兼容window_size参数
        if window_size is not None:
            count = window_size
        data_list = self.get_historical_data(symbol, count)
        
        if not data_list:
            return pd.DataFrame()
        
        rows = []
        for data in data_list:
            rows.append({
                'timestamp': data.timestamp,
                'symbol': data.symbol,
                'price': data.price,
                'volume': data.volume,
                'bid': data.bid,
                'ask': data.ask,
                'open': data.open,
                'high': data.high,
                'low': data.low,
                'close': data.close
            })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            # 添加测试期望的列名（兼容性）
            if 'close' in df.columns:
                df['Close'] = df['close']
            if 'volume' in df.columns:
                df['Volume'] = df['volume']
        
        return df
    
    def start(self) -> None:
        """启动数据管理器"""
        if self.running:
            self.logger.warning("数据管理器已在运行")
            return
        
        self.running = True
        self.performance_stats['start_time'] = datetime.now()
        
        # 启动WebSocket连接线程
        for data_feed in self.data_feeds.values():
            if data_feed.enabled:
                thread = threading.Thread(
                    target=self._run_websocket_connection,
                    args=(data_feed,),
                    daemon=True
                )
                thread.start()
                self.threads.append(thread)
        
        self.logger.info("实时数据管理器已启动")
    
    def _run_websocket_connection(self, data_feed: DataFeed) -> None:
        """运行WebSocket连接的线程函数"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.connect_websocket(data_feed))
        finally:
            loop.close()
    
    def stop(self) -> None:
        """停止数据管理器"""
        if not self.running:
            return
        
        self.running = False
        
        # 关闭WebSocket连接
        for connection in self.connections.values():
            if hasattr(connection, 'close'):
                try:
                    asyncio.create_task(connection.close())
                except:
                    pass
        
        # 等待线程结束
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)
        
        self.connections.clear()
        self.threads.clear()
        
        self.logger.info("实时数据管理器已停止")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        获取连接状态
        
        Returns:
            Dict[str, Any]: 连接状态信息
        """
        status = {
            'running': self.running,
            'data_feeds': {},
            'data_feeds_count': len(self.data_feeds),
            'performance': self.performance_stats.copy(),
            'buffer_usage': {},
            'buffer_size': self.buffer_size,  # 测试需要的字段
            'latency_ms': self.performance_stats.get('average_latency_ms', 0.0)  # 测试需要的字段
        }
        
        # 数据源状态
        for name, stats in self.connection_stats.items():
            status['data_feeds'][name] = {
                'connected': stats.connected,
                'reconnect_count': stats.reconnect_count,
                'messages_received': stats.messages_received,
                'messages_processed': stats.messages_processed,
                'last_message_time': stats.last_message_time.isoformat() if stats.last_message_time else None,
                'error_count': len(stats.errors),
                'recent_errors': stats.errors[-5:] if stats.errors else []
            }
        
        # 缓冲区使用情况
        for symbol, buffer in self.data_buffer.items():
            status['buffer_usage'][symbol] = {
                'current_size': len(buffer),
                'max_size': buffer.maxlen,
                'usage_ratio': len(buffer) / buffer.maxlen if buffer.maxlen else 0
            }
        
        return status
    
    def cleanup(self) -> None:
        """清理资源"""
        self.stop()
        self.data_buffer.clear()
        self.latest_data.clear()
        
        # 清空队列
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except Empty:
                break
        
        self.data_callbacks.clear()
        self.error_callbacks.clear()
        
        self.logger.info("实时数据管理器资源已清理")
    
    def remove_data_feed(self, name: str) -> bool:
        """
        移除数据源
        
        Args:
            name: 数据源名称
            
        Returns:
            bool: 移除是否成功
        """
        if name in self.data_feeds:
            # 停止相关连接
            if name in self.connections:
                connection = self.connections[name]
                if hasattr(connection, 'close'):
                    try:
                        asyncio.create_task(connection.close())
                    except:
                        pass
                del self.connections[name]
            
            # 清理统计信息
            if name in self.connection_stats:
                del self.connection_stats[name]
            
            # 移除数据源
            del self.data_feeds[name]
            self.logger.info(f"移除数据源: {name}")
            return True
        
        return False
    
    def _process_market_data(self, market_data: MarketData) -> None:
        """内部方法：处理市场数据"""
        self.process_market_data(market_data)
    
    @property 
    def is_running(self) -> bool:
        """获取运行状态"""
        return self.running
    
    def _handle_error(self, feed_name: str, error: Exception) -> None:
        """处理错误（测试兼容方法）"""
        # 记录错误到统计信息
        if feed_name in self.connection_stats:
            self.connection_stats[feed_name].errors.append(str(error))
        
        # 调用错误回调
        for callback in self.error_callbacks:
            try:
                callback(feed_name, error)
            except Exception as cb_error:
                self.logger.error(f"错误回调异常: {cb_error}")