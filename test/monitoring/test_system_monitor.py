"""
测试系统监控模块
"""

import pytest
import time
from unittest.mock import patch, MagicMock
from src.monitoring.system_monitor import SystemMonitor
from src.utils.config import Config


class TestSystemMonitor:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.config = Config()
        self.monitor = SystemMonitor(config=self.config)
    
    def teardown_method(self):
        """每个测试方法后的清理"""
        if hasattr(self.monitor, 'stop'):
            self.monitor.stop()
    
    def test_monitor_initialization(self):
        """测试监控器初始化"""
        assert self.monitor.config is not None
        assert hasattr(self.monitor, 'metrics_history')
        assert hasattr(self.monitor, 'is_running')
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_collect_system_metrics(self, mock_memory, mock_cpu):
        """测试系统指标收集"""
        # 模拟psutil返回值
        mock_cpu.return_value = 45.5
        mock_memory_obj = MagicMock()
        mock_memory_obj.percent = 60.0
        mock_memory_obj.available = 4 * 1024**3  # 4GB
        mock_memory_obj.total = 8 * 1024**3     # 8GB
        mock_memory.return_value = mock_memory_obj
        
        metrics = self.monitor.collect_system_metrics()
        
        assert 'cpu_percent' in metrics
        assert 'memory_percent' in metrics
        assert 'memory_available_gb' in metrics
        assert 'timestamp' in metrics
        
        assert metrics['cpu_percent'] == 45.5
        assert metrics['memory_percent'] == 60.0
    
    @patch('psutil.disk_usage')
    def test_collect_disk_metrics(self, mock_disk_usage):
        """测试磁盘指标收集"""
        mock_usage = MagicMock()
        mock_usage.total = 100 * 1024**3     # 100GB
        mock_usage.used = 60 * 1024**3       # 60GB
        mock_usage.free = 40 * 1024**3       # 40GB
        mock_usage.percent = 60.0
        mock_disk_usage.return_value = mock_usage
        
        metrics = self.monitor.collect_disk_metrics()
        
        assert 'disk_usage_percent' in metrics
        assert 'disk_free_gb' in metrics
        assert 'disk_total_gb' in metrics
        
        assert metrics['disk_usage_percent'] == 60.0
    
    @patch('psutil.net_io_counters')
    def test_collect_network_metrics(self, mock_net_io):
        """测试网络指标收集"""
        mock_io = MagicMock()
        mock_io.bytes_sent = 1024 * 1024     # 1MB
        mock_io.bytes_recv = 2 * 1024 * 1024 # 2MB
        mock_io.packets_sent = 100
        mock_io.packets_recv = 200
        mock_net_io.return_value = mock_io
        
        metrics = self.monitor.collect_network_metrics()
        
        assert 'bytes_sent_mb' in metrics
        assert 'bytes_recv_mb' in metrics
        assert 'packets_sent' in metrics
        assert 'packets_recv' in metrics
    
    def test_add_custom_metric(self):
        """测试添加自定义指标"""
        self.monitor.add_custom_metric('trading_signals_count', 150)
        self.monitor.add_custom_metric('model_accuracy', 0.85)
        
        custom_metrics = self.monitor.get_custom_metrics()
        
        assert 'trading_signals_count' in custom_metrics
        assert 'model_accuracy' in custom_metrics
        assert custom_metrics['trading_signals_count'] == 150
        assert custom_metrics['model_accuracy'] == 0.85
    
    def test_metrics_history(self):
        """测试指标历史记录"""
        # 添加一些历史数据
        for i in range(5):
            metrics = {
                'cpu_percent': 30 + i * 10,
                'memory_percent': 50 + i * 5,
                'timestamp': time.time() + i
            }
            self.monitor.add_metrics_to_history(metrics)
        
        history = self.monitor.get_metrics_history()
        
        assert len(history) == 5
        assert all('cpu_percent' in entry for entry in history)
        assert all('memory_percent' in entry for entry in history)
    
    def test_alert_thresholds(self):
        """测试告警阈值检查"""
        # 设置告警阈值
        self.monitor.set_alert_threshold('cpu_percent', 80.0)
        self.monitor.set_alert_threshold('memory_percent', 85.0)
        
        # 测试超过阈值的指标
        high_metrics = {
            'cpu_percent': 90.0,
            'memory_percent': 75.0
        }
        
        alerts = self.monitor.check_alert_thresholds(high_metrics)
        
        assert len(alerts) == 1
        assert alerts[0]['metric'] == 'cpu_percent'
        assert alerts[0]['value'] == 90.0
        assert alerts[0]['threshold'] == 80.0
    
    def test_performance_tracking(self):
        """测试性能跟踪"""
        # 开始跟踪
        self.monitor.start_performance_tracking('test_operation')
        
        # 模拟一些工作
        time.sleep(0.1)
        
        # 结束跟踪
        duration = self.monitor.end_performance_tracking('test_operation')
        
        assert duration >= 0.1
        
        # 获取性能统计
        stats = self.monitor.get_performance_stats()
        assert 'test_operation' in stats
    
    def test_health_check(self):
        """测试健康检查"""
        # 模拟健康检查函数
        def mock_database_check():
            return True, "Database connection OK"
        
        def mock_api_check():
            return False, "API endpoint unreachable"
        
        # 注册健康检查
        self.monitor.register_health_check('database', mock_database_check)
        self.monitor.register_health_check('api', mock_api_check)
        
        # 执行健康检查
        health_status = self.monitor.run_health_checks()
        
        assert 'database' in health_status
        assert 'api' in health_status
        assert health_status['database']['status'] is True
        assert health_status['api']['status'] is False
    
    def test_metrics_export(self):
        """测试指标导出"""
        # 添加一些指标
        for i in range(3):
            metrics = {
                'cpu_percent': 30 + i * 10,
                'memory_percent': 50 + i * 5,
                'timestamp': time.time() + i
            }
            self.monitor.add_metrics_to_history(metrics)
        
        # 导出为JSON格式
        exported_data = self.monitor.export_metrics('json')
        
        assert isinstance(exported_data, (str, dict))
        
        # 导出为CSV格式
        csv_data = self.monitor.export_metrics('csv')
        
        assert isinstance(csv_data, str)
        assert 'cpu_percent' in csv_data
        assert 'memory_percent' in csv_data
    
    def test_monitoring_start_stop(self):
        """测试监控启动和停止"""
        # 启动监控
        self.monitor.start_monitoring(interval=1)
        
        assert self.monitor.is_running is True
        
        # 等待一小段时间让监控运行
        time.sleep(0.5)
        
        # 停止监控
        self.monitor.stop_monitoring()
        
        assert self.monitor.is_running is False
    
    def test_callback_registration(self):
        """测试回调函数注册"""
        callback_data = []
        
        def metrics_callback(metrics):
            callback_data.append(metrics)
        
        # 注册回调
        self.monitor.register_metrics_callback(metrics_callback)
        
        # 触发回调
        test_metrics = {'cpu_percent': 50.0, 'memory_percent': 60.0}
        self.monitor._trigger_callbacks(test_metrics)
        
        assert len(callback_data) == 1
        assert callback_data[0] == test_metrics
    
    def test_metrics_aggregation(self):
        """测试指标聚合"""
        # 添加多个数据点
        for i in range(10):
            metrics = {
                'cpu_percent': 40 + (i % 3) * 10,  # 40, 50, 60的循环
                'memory_percent': 50 + i,
                'timestamp': time.time() + i
            }
            self.monitor.add_metrics_to_history(metrics)
        
        # 计算聚合统计
        aggregated = self.monitor.get_aggregated_metrics(window_minutes=60)
        
        assert 'cpu_percent' in aggregated
        assert 'memory_percent' in aggregated
        assert 'avg' in aggregated['cpu_percent']
        assert 'min' in aggregated['cpu_percent']
        assert 'max' in aggregated['cpu_percent']