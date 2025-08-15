"""
数据源路由管理器

负责加载和管理数据源路由配置，支持：
1. 基于市场类型的默认路由策略
2. 特殊符号的自定义路由规则  
3. 时间间隔优先级配置
4. 数据质量要求管理
"""

import os
import yaml
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field

from ..sources.base import MarketType, DataInterval, DataQuality, DataSource
from ...utils.logger import setup_logger


@dataclass
class RoutingRule:
    """路由规则"""
    primary: DataSource                        # 主数据源
    fallback: List[DataSource] = field(default_factory=list)  # 备选数据源
    description: str = ""               # 规则描述
    
    def get_all_sources(self) -> List[DataSource]:
        """获取所有数据源（按优先级）"""
        return [self.primary] + self.fallback
    
    def get_all_source_values(self) -> List[str]:
        """获取所有数据源的字符串值（向后兼容）"""
        return [source.value for source in self.get_all_sources()]


@dataclass
class QualityRequirement:
    """数据质量要求"""
    min_quality: DataQuality
    preferred_sources: List[DataSource] = field(default_factory=list)


@dataclass
class IntervalPreference:
    """时间间隔偏好配置"""
    intervals: List[str]
    preferred_sources: List[DataSource]


class RoutingManager:
    """数据源路由管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化路由管理器
        
        Args:
            config_file: 配置文件路径，如果为None则使用默认配置
        """
        self.logger = setup_logger("RoutingManager")
        
        # 默认配置文件路径
        if config_file is None:
            config_file = self._get_default_config_path()
        
        self.config_file = config_file
        self.config = {}
        
        # 路由规则缓存
        self._routing_rules: Dict[MarketType, RoutingRule] = {}
        self._symbol_overrides: Dict[str, RoutingRule] = {}
        self._interval_preferences: List[IntervalPreference] = []
        self._quality_requirements: Dict[MarketType, QualityRequirement] = {}
        self._global_settings: Dict[str, Any] = {}
        self._source_priorities: List[DataSource] = []
        
        # 加载配置
        self._load_config()
        
        self.logger.info(f"RoutingManager initialized with config: {config_file}")
    
    def _get_default_config_path(self) -> str:
        """获取默认配置文件路径"""
        # 配置文件在数据模块内部的config文件夹中
        current_dir = Path(__file__).parent.parent  # src/data (routing_manager is in managers/)
        config_path = current_dir / "config" / "data_routing.yaml"
        
        return str(config_path)
    
    def _load_config(self):
        """加载路由配置"""
        try:
            if not os.path.exists(self.config_file):
                self.logger.warning(f"Config file not found: {self.config_file}, using default routing")
                self._load_default_config()
                return
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            self._parse_config()
            self.logger.info("Successfully loaded routing configuration")
            
        except Exception as e:
            self.logger.error(f"Failed to load config file {self.config_file}: {e}")
            self.logger.info("Using default routing configuration")
            self._load_default_config()
    
    def _parse_config(self):
        """解析配置文件"""
        
        # 解析路由策略
        routing_strategy = self.config.get('routing_strategy', {})
        for market_type_str, rule_config in routing_strategy.items():
            try:
                market_type = MarketType(market_type_str)
                # 转换主数据源
                primary_source = DataSource.from_string(rule_config['primary'])
                
                # 转换备选数据源
                fallback_sources = []
                for fallback_str in rule_config.get('fallback', []):
                    try:
                        fallback_sources.append(DataSource.from_string(fallback_str))
                    except ValueError:
                        self.logger.warning(f"Invalid fallback data source: {fallback_str}")
                
                routing_rule = RoutingRule(
                    primary=primary_source,
                    fallback=fallback_sources,
                    description=rule_config.get('description', '')
                )
                self._routing_rules[market_type] = routing_rule
                
            except ValueError as e:
                self.logger.warning(f"Invalid market type in config: {market_type_str}")
        
        # 解析全局设置
        self._global_settings = self.config.get('global_settings', {})
        
        # 解析数据源优先级
        source_priorities_str = self.config.get('source_priorities', [])
        self._source_priorities = []
        for source_str in source_priorities_str:
            try:
                self._source_priorities.append(DataSource.from_string(source_str))
            except ValueError:
                self.logger.warning(f"Invalid source priority: {source_str}")
        
        # 解析符号覆盖规则
        symbol_overrides = self.config.get('symbol_overrides', {})
        for symbol, rule_config in symbol_overrides.items():
            # 转换主数据源
            try:
                primary_source = DataSource.from_string(rule_config['primary'])
                
                # 转换备选数据源
                fallback_sources = []
                for fallback_str in rule_config.get('fallback', []):
                    try:
                        fallback_sources.append(DataSource.from_string(fallback_str))
                    except ValueError:
                        self.logger.warning(f"Invalid symbol override fallback: {fallback_str}")
                
                routing_rule = RoutingRule(
                    primary=primary_source,
                    fallback=fallback_sources,
                    description=f"Override rule for {symbol}"
                )
            except ValueError:
                self.logger.warning(f"Invalid symbol override primary source for {symbol}: {rule_config['primary']}")
                continue
            self._symbol_overrides[symbol.upper()] = routing_rule
        
        # 解析时间间隔偏好
        interval_preferences = self.config.get('interval_preferences', {})
        for pref_name, pref_config in interval_preferences.items():
            # 转换偏好数据源
            preferred_sources = []
            for source_str in pref_config.get('preferred_sources', []):
                try:
                    preferred_sources.append(DataSource.from_string(source_str))
                except ValueError:
                    self.logger.warning(f"Invalid interval preference source: {source_str}")
            
            interval_pref = IntervalPreference(
                intervals=pref_config.get('intervals', []),
                preferred_sources=preferred_sources
            )
            self._interval_preferences.append(interval_pref)
        
        # 解析数据质量要求
        quality_requirements = self.config.get('quality_requirements', {})
        for market_type_str, quality_config in quality_requirements.items():
            try:
                market_type = MarketType(market_type_str)
                min_quality_str = quality_config.get('min_quality', 'MEDIUM')
                min_quality = DataQuality(min_quality_str)
                
                # 转换偏好数据源
                preferred_sources = []
                for source_str in quality_config.get('preferred_sources', []):
                    try:
                        preferred_sources.append(DataSource.from_string(source_str))
                    except ValueError:
                        self.logger.warning(f"Invalid quality requirement source: {source_str}")
                
                quality_req = QualityRequirement(
                    min_quality=min_quality,
                    preferred_sources=preferred_sources
                )
                self._quality_requirements[market_type] = quality_req
                
            except ValueError as e:
                self.logger.warning(f"Invalid quality config for {market_type_str}: {e}")
    
    def _load_default_config(self):
        """加载默认路由配置"""
        # 默认路由规则
        self._routing_rules = {
            MarketType.STOCK: RoutingRule(DataSource.YFINANCE, [], 'Default stock routing'),
            MarketType.FOREX: RoutingRule(DataSource.TRUEFX, [DataSource.OANDA, DataSource.FXMINUTE], 'Default forex routing'),
            MarketType.CRYPTO: RoutingRule(DataSource.YFINANCE, [], 'Default crypto routing'),
            MarketType.COMMODITIES: RoutingRule(DataSource.OANDA, [DataSource.YFINANCE], 'Default commodities routing'),
            MarketType.INDEX: RoutingRule(DataSource.YFINANCE, [], 'Default index routing'),
            MarketType.ETF: RoutingRule(DataSource.YFINANCE, [], 'Default ETF routing'),
            MarketType.FUTURES: RoutingRule(DataSource.OANDA, [], 'Default futures routing'),
            MarketType.OPTIONS: RoutingRule(DataSource.OANDA, [], 'Default options routing'),
            MarketType.BONDS: RoutingRule(DataSource.YFINANCE, [], 'Default bonds routing'),
        }
        
        # 默认全局设置
        self._global_settings = {
            'default_market_type': 'STOCK',
            'enable_source_compatibility_check': True,
            'enable_smart_fallback': True,
            'max_retries_per_source': 3,
            'retry_delay_seconds': 1.0,
            'health_check_interval_minutes': 30
        }
        
        # 默认数据源优先级
        self._source_priorities = [DataSource.YFINANCE, DataSource.TRUEFX, DataSource.OANDA, DataSource.FXMINUTE]
    
    def get_routing_rule(self, symbol: str, market_type: MarketType) -> RoutingRule:
        """
        获取指定符号和市场类型的路由规则
        
        Args:
            symbol: 交易符号
            market_type: 市场类型
            
        Returns:
            路由规则
        """
        symbol = symbol.upper().strip()
        
        # 首先检查符号特定的覆盖规则
        if symbol in self._symbol_overrides:
            self.logger.debug(f"Using symbol override rule for {symbol}")
            return self._symbol_overrides[symbol]
        
        # 使用基于市场类型的默认规则
        if market_type in self._routing_rules:
            return self._routing_rules[market_type]
        
        # 如果没有找到规则，使用股票市场的规则作为默认
        self.logger.warning(f"No routing rule found for market type {market_type}, using STOCK default")
        return self._routing_rules.get(MarketType.STOCK, RoutingRule(DataSource.YFINANCE))
    
    def get_optimal_sources(
        self, 
        symbol: str, 
        market_type: MarketType, 
        interval: str = "1d",
        quality_requirement: Optional[DataQuality] = None,
        check_compatibility: bool = True
    ) -> List[str]:
        """
        获取最优数据源列表（考虑时间间隔和质量要求）- 向后兼容版本
        
        Args:
            symbol: 交易符号
            market_type: 市场类型
            interval: 时间间隔
            quality_requirement: 数据质量要求
            check_compatibility: 是否进行兼容性检查
            
        Returns:
            按优先级排序的数据源字符串列表（向后兼容）
        """
        sources_enum = self.get_optimal_sources_enum(
            symbol=symbol,
            market_type=market_type,
            interval=interval,
            quality_requirement=quality_requirement,
            check_compatibility=check_compatibility
        )
        return [source.value for source in sources_enum]
    
    def get_optimal_sources_enum(
        self, 
        symbol: str, 
        market_type: MarketType, 
        interval: str = "1d",
        quality_requirement: Optional[DataQuality] = None,
        check_compatibility: bool = True
    ) -> List[DataSource]:
        """
        获取最优数据源列表（考虑时间间隔和质量要求）- 枚举版本
        
        Args:
            symbol: 交易符号
            market_type: 市场类型
            interval: 时间间隔
            quality_requirement: 数据质量要求
            check_compatibility: 是否进行兼容性检查
            
        Returns:
            按优先级排序的数据源枚举列表
        """
        # 获取基础路由规则
        base_rule = self.get_routing_rule(symbol, market_type)
        sources = base_rule.get_all_sources()
        
        # 考虑时间间隔偏好
        interval_sources = self._get_interval_preferred_sources(interval)
        if interval_sources:
            # 将时间间隔偏好的数据源提前
            optimized_sources = []
            for source in interval_sources:
                if source in sources and source not in optimized_sources:
                    optimized_sources.append(source)
            
            # 添加其他数据源
            for source in sources:
                if source not in optimized_sources:
                    optimized_sources.append(source)
            
            sources = optimized_sources
        
        # 考虑数据质量要求
        if quality_requirement is None:
            quality_req = self._quality_requirements.get(market_type)
            if quality_req:
                quality_requirement = quality_req.min_quality
        
        if quality_requirement:
            quality_sources = self._filter_by_quality(sources, quality_requirement)
            if quality_sources:
                sources = quality_sources
        
        # 如果启用兼容性检查，过滤出兼容的数据源
        if check_compatibility:
            compatible_sources = self._filter_by_compatibility(sources, market_type, interval, quality_requirement)
            if compatible_sources:
                sources = compatible_sources
            else:
                self.logger.warning(f"No compatible sources found for {symbol} ({market_type}, {interval})")
        
        return sources
    
    def _get_interval_preferred_sources(self, interval: str) -> List[DataSource]:
        """根据时间间隔获取偏好数据源"""
        for interval_pref in self._interval_preferences:
            if interval in interval_pref.intervals:
                return interval_pref.preferred_sources
        return []
    
    def _filter_by_quality(self, sources: List[DataSource], min_quality: DataQuality) -> List[DataSource]:
        """根据质量要求过滤数据源"""
        # 根据数据源的内置质量信息过滤
        filtered_sources = []
        for source in sources:
            if source.data_quality.value >= min_quality.value:
                filtered_sources.append(source)
            else:
                self.logger.debug(f"Source '{source.value}' quality ({source.data_quality}) below requirement ({min_quality})")
        
        return filtered_sources if filtered_sources else sources
    
    def _filter_by_compatibility(
        self, 
        sources: List[DataSource], 
        market_type: MarketType, 
        interval: str, 
        quality_requirement: Optional[DataQuality] = None
    ) -> List[DataSource]:
        """根据兼容性过滤数据源"""
        try:
            from ..advisors.compatibility_checker import get_compatibility_checker, CompatibilityRequest
            from ..sources.base import DataInterval
            
            # 转换时间间隔
            interval_map = {
                '1m': DataInterval.MINUTE_1,
                '2m': DataInterval.MINUTE_2,
                '5m': DataInterval.MINUTE_5,
                '15m': DataInterval.MINUTE_15,
                '30m': DataInterval.MINUTE_30,
                '1h': DataInterval.HOUR_1,
                '60m': DataInterval.HOUR_1,
                '90m': DataInterval.MINUTE_90,
                '1d': DataInterval.DAY_1,
                '3d': DataInterval.DAY_3,
                '5d': DataInterval.DAY_1,
                '1wk': DataInterval.WEEK_1,
                '1mo': DataInterval.MONTH_1,
                '3mo': DataInterval.MONTH_3
            }
            interval_enum = interval_map.get(interval, DataInterval.DAY_1)
            
            compatibility_checker = get_compatibility_checker()
            compatible_sources = []
            
            for source in sources:
                try:
                    request = CompatibilityRequest(
                        source=source.value,  # CompatibilityRequest 仍然使用字符串
                        market_type=market_type,
                        interval=interval_enum,
                        quality_requirement=quality_requirement
                    )
                    
                    result = compatibility_checker.check_compatibility(request)
                    
                    if result.is_compatible:
                        compatible_sources.append(source)
                    else:
                        self.logger.debug(f"Source '{source.value}' not compatible: {'; '.join(result.issues)}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to check compatibility for source '{source.value}': {e}")
                    # 如果检查失败，保守地认为该数据源可用
                    compatible_sources.append(source)
            
            return compatible_sources
            
        except ImportError:
            self.logger.warning("Compatibility checker not available, skipping compatibility filter")
            return sources
    
    def get_global_setting(self, key: str, default: Any = None) -> Any:
        """获取全局设置"""
        return self._global_settings.get(key, default)
    
    def get_source_priorities(self) -> List[str]:
        """获取数据源优先级列表（向后兼容版本）"""
        return [source.value for source in self._source_priorities]
    
    def get_source_priorities_enum(self) -> List[DataSource]:
        """获取数据源优先级枚举列表"""
        return self._source_priorities.copy()
    
    def is_source_compatibility_check_enabled(self) -> bool:
        """检查是否启用数据源兼容性检查"""
        return self.get_global_setting('enable_source_compatibility_check', True)
    
    def is_smart_fallback_enabled(self) -> bool:
        """检查是否启用智能降级"""
        return self.get_global_setting('enable_smart_fallback', True)
    
    def get_max_retries_per_source(self) -> int:
        """获取单个数据源的最大重试次数"""
        return self.get_global_setting('max_retries_per_source', 3)
    
    def get_retry_delay_seconds(self) -> float:
        """获取重试延迟"""
        return self.get_global_setting('retry_delay_seconds', 1.0)
    
    def get_default_market_type(self) -> MarketType:
        """获取默认市场类型"""
        default_str = self.get_global_setting('default_market_type', 'STOCK')
        try:
            return MarketType(default_str)
        except ValueError:
            return MarketType.STOCK
    
    def add_symbol_override(self, symbol: str, routing_rule: RoutingRule):
        """
        动态添加符号覆盖规则
        
        Args:
            symbol: 交易符号
            routing_rule: 路由规则
        """
        symbol = symbol.upper().strip()
        self._symbol_overrides[symbol] = routing_rule
        self.logger.info(f"Added symbol override for {symbol}: {routing_rule.primary.value}")
    
    def remove_symbol_override(self, symbol: str):
        """
        移除符号覆盖规则
        
        Args:
            symbol: 交易符号
        """
        symbol = symbol.upper().strip()
        if symbol in self._symbol_overrides:
            del self._symbol_overrides[symbol]
            self.logger.info(f"Removed symbol override for {symbol}")
    
    def get_routing_summary(self) -> Dict[str, Any]:
        """
        获取路由配置摘要
        
        Returns:
            路由配置摘要字典
        """
        return {
            'config_file': self.config_file,
            'market_type_rules': {
                market_type.value: {
                    'primary': rule.primary.value,
                    'fallback': [source.value for source in rule.fallback],
                    'description': rule.description
                }
                for market_type, rule in self._routing_rules.items()
            },
            'symbol_overrides_count': len(self._symbol_overrides),
            'interval_preferences_count': len(self._interval_preferences),
            'quality_requirements_count': len(self._quality_requirements),
            'source_priorities': [source.value for source in self._source_priorities],
            'global_settings': self._global_settings
        }
    
    def reload_config(self):
        """重新加载配置文件"""
        self.logger.info("Reloading routing configuration...")
        
        # 清空当前配置
        self._routing_rules.clear()
        self._symbol_overrides.clear()
        self._interval_preferences.clear()
        self._quality_requirements.clear()
        self._global_settings.clear()
        self._source_priorities.clear()
        
        # 重新加载
        self._load_config()
        
        self.logger.info("Routing configuration reloaded successfully")
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """
        验证配置的有效性
        
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查必要的市场类型是否都有配置
        required_markets = [MarketType.STOCK, MarketType.FOREX, MarketType.CRYPTO]
        for market_type in required_markets:
            if market_type not in self._routing_rules:
                errors.append(f"Missing routing rule for required market type: {market_type.value}")
        
        # 检查数据源是否存在（需要DataSourceRegistry）
        try:
            from ..sources import DataSourceRegistry
            all_sources = set()
            
            # 收集所有配置中使用的数据源
            for rule in self._routing_rules.values():
                all_sources.update(source.value for source in rule.get_all_sources())
            
            for rule in self._symbol_overrides.values():
                all_sources.update(source.value for source in rule.get_all_sources())
            
            # 检查数据源是否已注册
            for source_str in all_sources:
                if not DataSourceRegistry.is_registered(source_str):
                    errors.append(f"Data source '{source_str}' is not registered")
                    
        except ImportError:
            errors.append("Cannot validate data sources: DataSourceRegistry not available")
        
        # 检查全局设置的数据类型
        if 'max_retries_per_source' in self._global_settings:
            if not isinstance(self._global_settings['max_retries_per_source'], int):
                errors.append("max_retries_per_source must be an integer")
        
        if 'retry_delay_seconds' in self._global_settings:
            if not isinstance(self._global_settings['retry_delay_seconds'], (int, float)):
                errors.append("retry_delay_seconds must be a number")
        
        return len(errors) == 0, errors


# 全局实例（单例模式）
_global_routing_manager: Optional[RoutingManager] = None


def get_routing_manager() -> RoutingManager:
    """获取全局路由管理器实例"""
    global _global_routing_manager
    if _global_routing_manager is None:
        _global_routing_manager = RoutingManager()
    return _global_routing_manager


def set_routing_manager(routing_manager: RoutingManager):
    """设置全局路由管理器实例"""
    global _global_routing_manager
    _global_routing_manager = routing_manager