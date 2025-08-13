"""
告警管理器
处理系统告警、通知分发、告警策略和历史记录

主要功能:
1. 告警规则管理
2. 告警触发和处理
3. 通知渠道管理
4. 告警历史记录
5. 告警抑制和聚合
6. 告警升级策略
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

from ..utils.logger import setup_logger, get_default_log_file
from ..utils.config import Config


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """告警状态"""
    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """告警对象"""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    source: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.FIRING
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    resolve_time: Optional[datetime] = None
    suppressed_until: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'alert_id': self.alert_id,
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'labels': self.labels,
            'annotations': self.annotations,
            'resolve_time': self.resolve_time.isoformat() if self.resolve_time else None,
            'suppressed_until': self.suppressed_until.isoformat() if self.suppressed_until else None
        }
    
    @property
    def is_active(self) -> bool:
        """是否为活跃告警"""
        return self.status == AlertStatus.FIRING
    
    @property
    def is_suppressed(self) -> bool:
        """是否被抑制"""
        if self.status == AlertStatus.SUPPRESSED:
            return True
        if self.suppressed_until and datetime.now() < self.suppressed_until:
            return True
        return False
    
    def suppress(self, duration_minutes: int) -> None:
        """抑制告警"""
        self.status = AlertStatus.SUPPRESSED
        self.suppressed_until = datetime.now() + timedelta(minutes=duration_minutes)
    
    def resolve(self) -> None:
        """解决告警"""
        self.status = AlertStatus.RESOLVED
        self.resolve_time = datetime.now()


@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    condition: str  # 条件表达式
    level: AlertLevel
    enabled: bool = True
    cooldown_minutes: int = 5  # 冷却期
    max_alerts_per_hour: int = 10  # 每小时最大告警数
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class NotificationChannel:
    """通知渠道"""
    channel_id: str
    name: str
    type: str  # email, webhook, slack, etc.
    config: Dict[str, Any]
    enabled: bool = True
    min_level: AlertLevel = AlertLevel.WARNING


class AlertManager:
    """
    告警管理器
    
    提供完整的告警管理功能:
    - 告警规则管理
    - 告警处理和通知
    - 告警历史记录
    - 告警抑制和去重
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        max_alerts: int = 10000,
        retention_days: int = 30
    ):
        """
        初始化告警管理器
        
        Args:
            config: 配置对象
            max_alerts: 最大告警数量
            retention_days: 告警保留天数
        """
        self.config = config or Config()
        self.max_alerts = max_alerts
        self.retention_days = retention_days
        
        # 初始化日志
        self.logger = setup_logger(
            name="AlertManager",
            level="INFO",
            log_file=get_default_log_file("alert_manager")
        )
        
        # 告警存储
        self.alerts: Dict[str, Alert] = {}  # alert_id -> Alert
        self.alert_history: deque = deque(maxlen=max_alerts)
        
        # 告警规则
        self.rules: Dict[str, AlertRule] = {}
        
        # 通知渠道
        self.channels: Dict[str, NotificationChannel] = {}
        
        # 告警计数和限制
        self.alert_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))  # rule_id -> timestamps
        self.last_alert_time: Dict[str, datetime] = {}  # rule_id -> last_alert_time
        
        # 运行状态
        self.running = False
        self.cleanup_thread: Optional[threading.Thread] = None
        
        # 回调函数
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # 初始化默认规则和通道
        self._init_default_rules()
        self._init_default_channels()
        
        self.logger.info("AlertManager 初始化完成")
    
    def _init_default_rules(self) -> None:
        """初始化默认告警规则"""
        default_rules = [
            AlertRule(
                rule_id="high_cpu",
                name="CPU使用率过高",
                condition="cpu_percent > 90",
                level=AlertLevel.CRITICAL,
                cooldown_minutes=5,
                labels={"category": "system"},
                annotations={"description": "CPU使用率超过90%"}
            ),
            AlertRule(
                rule_id="high_memory",
                name="内存使用率过高",
                condition="memory_percent > 85",
                level=AlertLevel.WARNING,
                cooldown_minutes=5,
                labels={"category": "system"},
                annotations={"description": "内存使用率超过85%"}
            ),
            AlertRule(
                rule_id="high_latency",
                name="数据延迟过高",
                condition="data_latency_ms > 200",
                level=AlertLevel.WARNING,
                cooldown_minutes=2,
                labels={"category": "performance"},
                annotations={"description": "数据延迟超过200ms"}
            ),
            AlertRule(
                rule_id="trading_system_down",
                name="交易系统停止",
                condition="trading_system_status != 'healthy'",
                level=AlertLevel.CRITICAL,
                cooldown_minutes=1,
                labels={"category": "business"},
                annotations={"description": "交易系统状态异常"}
            ),
            AlertRule(
                rule_id="low_success_rate",
                name="交易成功率过低",
                condition="success_rate < 0.8",
                level=AlertLevel.WARNING,
                cooldown_minutes=10,
                labels={"category": "business"},
                annotations={"description": "交易成功率低于80%"}
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
    
    def _init_default_channels(self) -> None:
        """初始化默认通知渠道"""
        # 控制台通知渠道
        self.channels["console"] = NotificationChannel(
            channel_id="console",
            name="控制台通知",
            type="console",
            config={},
            min_level=AlertLevel.INFO
        )
        
        # 邮件通知渠道（需要配置）
        email_config = self.config.get("alerts", {}).get("email", {})
        if email_config.get("enabled", False):
            self.channels["email"] = NotificationChannel(
                channel_id="email",
                name="邮件通知",
                type="email",
                config=email_config,
                min_level=AlertLevel.WARNING
            )
        
        # Webhook通知渠道（需要配置）
        webhook_config = self.config.get("alerts", {}).get("webhook", {})
        if webhook_config.get("enabled", False):
            self.channels["webhook"] = NotificationChannel(
                channel_id="webhook",
                name="Webhook通知",
                type="webhook",
                config=webhook_config,
                min_level=AlertLevel.WARNING
            )
    
    def start(self) -> None:
        """启动告警管理器"""
        if self.running:
            self.logger.warning("告警管理器已在运行")
            return
        
        self.running = True
        
        # 启动清理线程
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self.cleanup_thread.start()
        
        self.logger.info("告警管理器已启动")
    
    def stop(self) -> None:
        """停止告警管理器"""
        if not self.running:
            return
        
        self.running = False
        
        # 等待清理线程结束
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        
        self.logger.info("告警管理器已停止")
    
    def add_rule(self, rule: AlertRule) -> None:
        """添加告警规则"""
        self.rules[rule.rule_id] = rule
        self.logger.info(f"添加告警规则: {rule.name}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """移除告警规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.logger.info(f"移除告警规则: {rule_id}")
            return True
        return False
    
    def add_channel(self, channel: NotificationChannel) -> None:
        """添加通知渠道"""
        self.channels[channel.channel_id] = channel
        self.logger.info(f"添加通知渠道: {channel.name}")
    
    def remove_channel(self, channel_id: str) -> bool:
        """移除通知渠道"""
        if channel_id in self.channels:
            del self.channels[channel_id]
            self.logger.info(f"移除通知渠道: {channel_id}")
            return True
        return False
    
    def create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        source: str = "system",
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None
    ) -> Alert:
        """创建告警"""
        alert_id = f"{source}_{int(time.time() * 1000000)}"
        
        alert = Alert(
            alert_id=alert_id,
            level=level,
            title=title,
            message=message,
            source=source,
            timestamp=datetime.now(),
            labels=labels or {},
            annotations=annotations or {}
        )
        
        return self._process_alert(alert)
    
    def _process_alert(self, alert: Alert) -> Alert:
        """处理告警"""
        try:
            # 检查是否重复告警
            existing_alert = self._find_similar_alert(alert)
            if existing_alert and not existing_alert.is_suppressed:
                self.logger.debug(f"跳过重复告警: {alert.title}")
                return existing_alert
            
            # 存储告警
            self.alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # 触发回调
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"告警回调异常: {e}")
            
            # 发送通知
            self._send_notifications(alert)
            
            self.logger.info(f"处理告警: {alert.level.value.upper()} - {alert.title}")
            
            return alert
            
        except Exception as e:
            self.logger.error(f"处理告警失败: {e}")
            return alert
    
    def _find_similar_alert(self, alert: Alert) -> Optional[Alert]:
        """查找相似告警"""
        for existing_alert in self.alerts.values():
            if (existing_alert.source == alert.source and
                existing_alert.title == alert.title and
                existing_alert.is_active and
                (datetime.now() - existing_alert.timestamp).total_seconds() < 300):  # 5分钟内
                return existing_alert
        return None
    
    def _send_notifications(self, alert: Alert) -> None:
        """发送通知"""
        for channel in self.channels.values():
            if not channel.enabled:
                continue
            
            # 检查告警级别
            if alert.level.value < channel.min_level.value:
                continue
            
            try:
                if channel.type == "console":
                    self._send_console_notification(alert, channel)
                elif channel.type == "email":
                    self._send_email_notification(alert, channel)
                elif channel.type == "webhook":
                    self._send_webhook_notification(alert, channel)
                else:
                    self.logger.warning(f"未知通知渠道类型: {channel.type}")
                    
            except Exception as e:
                self.logger.error(f"发送通知失败 {channel.name}: {e}")
    
    def _send_console_notification(self, alert: Alert, channel: NotificationChannel) -> None:
        """发送控制台通知"""
        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨"
        }
        
        emoji = level_emoji.get(alert.level, "")
        print(f"\n{emoji} [{alert.level.value.upper()}] {alert.title}")
        print(f"📝 {alert.message}")
        print(f"🔗 来源: {alert.source}")
        print(f"⏰ 时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        if alert.labels:
            print(f"🏷️  标签: {alert.labels}")
        print("-" * 50)
    
    def _send_email_notification(self, alert: Alert, channel: NotificationChannel) -> None:
        """发送邮件通知"""
        config = channel.config
        
        # 邮件配置
        smtp_server = config.get("smtp_server")
        smtp_port = config.get("smtp_port", 587)
        username = config.get("username")
        password = config.get("password")
        to_emails = config.get("to_emails", [])
        
        if not all([smtp_server, username, password, to_emails]):
            self.logger.error("邮件配置不完整")
            return
        
        # 构建邮件
        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = ", ".join(to_emails)
        msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
        
        # 邮件内容
        body = f"""
告警详情:

级别: {alert.level.value.upper()}
标题: {alert.title}
消息: {alert.message}
来源: {alert.source}
时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

标签: {alert.labels}
注释: {alert.annotations}

---
TensorTrade 告警系统
        """.strip()
        
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # 发送邮件
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
    
    def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel) -> None:
        """发送Webhook通知"""
        config = channel.config
        url = config.get("url")
        
        if not url:
            self.logger.error("Webhook URL未配置")
            return
        
        # 构建请求数据
        payload = {
            "alert": alert.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "source": "tensortrade"
        }
        
        # 发送请求
        headers = config.get("headers", {"Content-Type": "application/json"})
        timeout = config.get("timeout", 10)
        
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Webhook请求失败: {response.status_code}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.resolve()
            self.logger.info(f"告警已解决: {alert.title}")
            return True
        return False
    
    def suppress_alert(self, alert_id: str, duration_minutes: int) -> bool:
        """抑制告警"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.suppress(duration_minutes)
            self.logger.info(f"告警已抑制 {duration_minutes} 分钟: {alert.title}")
            return True
        return False
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """获取活跃告警"""
        alerts = [alert for alert in self.alerts.values() if alert.is_active and not alert.is_suppressed]
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """获取告警历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计"""
        active_alerts = self.get_active_alerts()
        recent_alerts = self.get_alert_history(24)
        
        stats = {
            'total_active': len(active_alerts),
            'total_last_24h': len(recent_alerts),
            'by_level': {
                'critical': len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
                'warning': len([a for a in active_alerts if a.level == AlertLevel.WARNING]),
                'info': len([a for a in active_alerts if a.level == AlertLevel.INFO])
            },
            'by_source': {},
            'rules_enabled': len([r for r in self.rules.values() if r.enabled]),
            'channels_enabled': len([c for c in self.channels.values() if c.enabled])
        }
        
        # 按来源统计
        for alert in active_alerts:
            source = alert.source
            if source not in stats['by_source']:
                stats['by_source'][source] = 0
            stats['by_source'][source] += 1
        
        return stats
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    def _cleanup_loop(self) -> None:
        """清理循环"""
        while self.running:
            try:
                self._cleanup_old_alerts()
                self._cleanup_alert_counts()
            except Exception as e:
                self.logger.error(f"清理异常: {e}")
            
            time.sleep(3600)  # 每小时清理一次
    
    def _cleanup_old_alerts(self) -> None:
        """清理旧告警"""
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        
        # 清理已解决的旧告警
        alerts_to_remove = [
            alert_id for alert_id, alert in self.alerts.items()
            if alert.status == AlertStatus.RESOLVED and alert.resolve_time < cutoff_time
        ]
        
        for alert_id in alerts_to_remove:
            del self.alerts[alert_id]
        
        if alerts_to_remove:
            self.logger.info(f"清理了 {len(alerts_to_remove)} 个旧告警")
    
    def _cleanup_alert_counts(self) -> None:
        """清理告警计数"""
        cutoff_time = time.time() - 3600  # 1小时前
        
        for rule_id, timestamps in self.alert_counts.items():
            while timestamps and timestamps[0] < cutoff_time:
                timestamps.popleft()
    
    def export_alerts_to_json(self, filename: str, hours: int = 24) -> None:
        """导出告警到JSON文件"""
        try:
            alerts_data = self.get_alert_history(hours)
            
            data = {
                'export_time': datetime.now().isoformat(),
                'alerts_count': len(alerts_data),
                'hours': hours,
                'alerts': [alert.to_dict() for alert in alerts_data]
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"告警已导出到: {filename}, 共 {len(alerts_data)} 条记录")
            
        except Exception as e:
            self.logger.error(f"导出告警失败: {e}")
            raise
    
    def generate_alert_report(self) -> Dict[str, Any]:
        """生成告警报告"""
        return {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_alert_statistics(),
            'active_alerts': [alert.to_dict() for alert in self.get_active_alerts()],
            'recent_alerts': [alert.to_dict() for alert in self.get_alert_history(24)],
            'rules': {
                rule_id: {
                    'name': rule.name,
                    'enabled': rule.enabled,
                    'level': rule.level.value,
                    'condition': rule.condition
                }
                for rule_id, rule in self.rules.items()
            },
            'channels': {
                channel_id: {
                    'name': channel.name,
                    'type': channel.type,
                    'enabled': channel.enabled,
                    'min_level': channel.min_level.value
                }
                for channel_id, channel in self.channels.items()
            }
        }