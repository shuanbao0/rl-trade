"""
测试告警管理器模块
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open
from src.monitoring.alert_manager import AlertManager, Alert, AlertLevel, AlertRule
from src.utils.config import Config


class TestAlertManager:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.config = Config()
        self.alert_manager = AlertManager(config=self.config)
    
    def teardown_method(self):
        """每个测试方法后的清理"""
        if hasattr(self.alert_manager, 'stop'):
            self.alert_manager.stop()
    
    def test_alert_manager_initialization(self):
        """测试告警管理器初始化"""
        assert self.alert_manager.config is not None
        assert hasattr(self.alert_manager, 'alerts')
        assert hasattr(self.alert_manager, 'rules')
        assert hasattr(self.alert_manager, 'channels')
        assert len(self.alert_manager.rules) > 0  # 应该有默认规则
    
    def test_create_alert(self):
        """测试创建告警"""
        alert = self.alert_manager.create_alert(
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="This is a test alert",
            source="test_system",
            labels={"component": "trading"},
            annotations={"runbook": "check_trading_system"}
        )
        
        assert alert is not None
        assert alert.level == AlertLevel.WARNING
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test alert"
        assert alert.source == "test_system"
        assert alert.labels["component"] == "trading"
        assert alert.annotations["runbook"] == "check_trading_system"
        assert alert.alert_id in self.alert_manager.alerts
    
    def test_add_alert_rule(self):
        """测试添加告警规则"""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            condition="cpu_percent > 90",
            level=AlertLevel.CRITICAL,
            cooldown_minutes=5,
            labels={"team": "devops"}
        )
        
        self.alert_manager.add_rule(rule)
        
        assert "test_rule" in self.alert_manager.rules
        assert self.alert_manager.rules["test_rule"].name == "Test Rule"
    
    def test_remove_alert_rule(self):
        """测试移除告警规则"""
        # 先添加一个规则
        rule = AlertRule(
            rule_id="temp_rule",
            name="Temporary Rule",
            condition="memory_percent > 95",
            level=AlertLevel.WARNING
        )
        self.alert_manager.add_rule(rule)
        
        # 然后移除它
        result = self.alert_manager.remove_rule("temp_rule")
        
        assert result is True
        assert "temp_rule" not in self.alert_manager.rules
    
    def test_resolve_alert(self):
        """测试解决告警"""
        # 创建一个告警
        alert = self.alert_manager.create_alert(
            level=AlertLevel.WARNING,
            title="Resolvable Alert",
            message="This alert will be resolved"
        )
        
        # 解决告警
        result = self.alert_manager.resolve_alert(alert.alert_id)
        
        assert result is True
        resolved_alert = self.alert_manager.alerts[alert.alert_id]
        assert resolved_alert.status.value == "resolved"
        assert resolved_alert.resolve_time is not None
    
    def test_suppress_alert(self):
        """测试抑制告警"""
        # 创建一个告警
        alert = self.alert_manager.create_alert(
            level=AlertLevel.INFO,
            title="Suppressible Alert",
            message="This alert will be suppressed"
        )
        
        # 抑制告警30分钟
        result = self.alert_manager.suppress_alert(alert.alert_id, 30)
        
        assert result is True
        suppressed_alert = self.alert_manager.alerts[alert.alert_id]
        assert suppressed_alert.status.value == "suppressed"
        assert suppressed_alert.suppressed_until is not None
    
    def test_get_active_alerts(self):
        """测试获取活跃告警"""
        # 创建几个不同状态的告警
        active_alert = self.alert_manager.create_alert(
            level=AlertLevel.CRITICAL,
            title="Active Alert",
            message="This is an active alert"
        )
        
        resolved_alert = self.alert_manager.create_alert(
            level=AlertLevel.WARNING,
            title="Resolved Alert",
            message="This alert was resolved"
        )
        self.alert_manager.resolve_alert(resolved_alert.alert_id)
        
        # 获取活跃告警
        active_alerts = self.alert_manager.get_active_alerts()
        
        assert len(active_alerts) >= 1
        assert any(alert.alert_id == active_alert.alert_id for alert in active_alerts)
        assert not any(alert.alert_id == resolved_alert.alert_id for alert in active_alerts)
    
    def test_get_active_alerts_by_level(self):
        """测试按级别获取活跃告警"""
        # 创建不同级别的告警
        critical_alert = self.alert_manager.create_alert(
            level=AlertLevel.CRITICAL,
            title="Critical Alert",
            message="Critical issue"
        )
        
        warning_alert = self.alert_manager.create_alert(
            level=AlertLevel.WARNING,
            title="Warning Alert",
            message="Warning issue"
        )
        
        # 获取关键级别的告警
        critical_alerts = self.alert_manager.get_active_alerts(AlertLevel.CRITICAL)
        
        assert len(critical_alerts) >= 1
        assert all(alert.level == AlertLevel.CRITICAL for alert in critical_alerts)
    
    def test_get_alert_history(self):
        """测试获取告警历史"""
        # 创建几个告警
        for i in range(3):
            self.alert_manager.create_alert(
                level=AlertLevel.INFO,
                title=f"Historical Alert {i}",
                message=f"Alert number {i}"
            )
        
        # 获取24小时内的告警历史
        history = self.alert_manager.get_alert_history(24)
        
        assert len(history) >= 3
        assert all(isinstance(alert, Alert) for alert in history)
    
    def test_get_alert_statistics(self):
        """测试获取告警统计"""
        # 创建不同类型的告警
        self.alert_manager.create_alert(AlertLevel.CRITICAL, "Critical 1", "Message")
        self.alert_manager.create_alert(AlertLevel.WARNING, "Warning 1", "Message")
        self.alert_manager.create_alert(AlertLevel.INFO, "Info 1", "Message")
        
        stats = self.alert_manager.get_alert_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_active' in stats
        assert 'by_level' in stats
        assert 'by_source' in stats
        assert 'rules_enabled' in stats
        assert 'channels_enabled' in stats
        
        # 验证按级别统计
        assert 'critical' in stats['by_level']
        assert 'warning' in stats['by_level']
        assert 'info' in stats['by_level']
    
    @patch('builtins.print')
    def test_console_notification(self, mock_print):
        """测试控制台通知"""
        # 创建一个告警
        self.alert_manager.create_alert(
            level=AlertLevel.WARNING,
            title="Console Test",
            message="Testing console notification",
            labels={"component": "test"}
        )
        
        # 验证print被调用（控制台通知）
        mock_print.assert_called()
    
    def test_alert_callback(self):
        """测试告警回调"""
        callback_data = []
        
        def test_callback(alert):
            callback_data.append(alert)
        
        # 添加回调
        self.alert_manager.add_alert_callback(test_callback)
        
        # 创建告警
        alert = self.alert_manager.create_alert(
            level=AlertLevel.INFO,
            title="Callback Test",
            message="Testing callback functionality"
        )
        
        # 验证回调被调用
        assert len(callback_data) == 1
        assert callback_data[0].alert_id == alert.alert_id
    
    def test_similar_alert_detection(self):
        """测试相似告警检测"""
        # 创建第一个告警
        alert1 = self.alert_manager.create_alert(
            level=AlertLevel.WARNING,
            title="Duplicate Test",
            message="First occurrence",
            source="test_source"
        )
        
        # 创建相似的告警（应该被跳过）
        alert2 = self.alert_manager.create_alert(
            level=AlertLevel.WARNING,
            title="Duplicate Test",
            message="Second occurrence",
            source="test_source"
        )
        
        # 第二个告警应该返回第一个告警的引用
        assert alert2.alert_id == alert1.alert_id
    
    def test_export_alerts_to_json(self):
        """测试导出告警到JSON"""
        # 创建几个告警
        for i in range(2):
            self.alert_manager.create_alert(
                level=AlertLevel.INFO,
                title=f"Export Test {i}",
                message=f"Export test alert {i}"
            )
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                self.alert_manager.export_alerts_to_json('test_export.json', hours=24)
                
                mock_file.assert_called_once()
                mock_json_dump.assert_called_once()
    
    def test_generate_alert_report(self):
        """测试生成告警报告"""
        # 创建一些告警
        self.alert_manager.create_alert(AlertLevel.CRITICAL, "Report Test", "Message")
        
        report = self.alert_manager.generate_alert_report()
        
        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'statistics' in report
        assert 'active_alerts' in report
        assert 'recent_alerts' in report
        assert 'rules' in report
        assert 'channels' in report
    
    def test_alert_manager_start_stop(self):
        """测试告警管理器启动和停止"""
        # 启动告警管理器
        self.alert_manager.start()
        assert self.alert_manager.running is True
        
        # 停止告警管理器
        self.alert_manager.stop()
        assert self.alert_manager.running is False
    
    def test_alert_suppression_time_based(self):
        """测试基于时间的告警抑制"""
        # 创建并抑制告警
        alert = self.alert_manager.create_alert(
            level=AlertLevel.WARNING,
            title="Time Suppression Test",
            message="Testing time-based suppression"
        )
        
        # 抑制1分钟
        alert.suppress(1)
        
        # 验证抑制状态
        assert alert.is_suppressed is True
        assert alert.suppressed_until > datetime.now()
    
    def test_alert_to_dict_conversion(self):
        """测试告警转字典"""
        alert = self.alert_manager.create_alert(
            level=AlertLevel.ERROR,
            title="Dict Test",
            message="Testing dictionary conversion",
            labels={"env": "test"},
            annotations={"doc": "test_doc"}
        )
        
        alert_dict = alert.to_dict()
        
        assert isinstance(alert_dict, dict)
        assert alert_dict['level'] == 'error'
        assert alert_dict['title'] == "Dict Test"
        assert alert_dict['message'] == "Testing dictionary conversion"
        assert alert_dict['labels']['env'] == "test"
        assert alert_dict['annotations']['doc'] == "test_doc"