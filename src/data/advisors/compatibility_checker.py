"""
数据源兼容性检查器

提供全面的数据源兼容性验证功能：
1. 市场类型支持检查
2. 时间间隔支持检查  
3. 数据质量要求验证
4. 认证要求检查
5. 速率限制检查
6. 历史数据可用性检查
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..sources.base import MarketType, DataInterval, DataQuality, DataSourceCapabilities
from ..sources import DataSourceRegistry
from ...utils.logger import setup_logger


@dataclass
class CompatibilityResult:
    """兼容性检查结果"""
    is_compatible: bool                     # 是否兼容
    issues: List[str]                      # 问题列表
    warnings: List[str]                    # 警告列表
    recommendations: List[str]             # 建议列表
    compatibility_score: float             # 兼容性评分 (0-1)
    
    def has_critical_issues(self) -> bool:
        """是否有严重问题"""
        return not self.is_compatible
    
    def has_warnings(self) -> bool:
        """是否有警告"""
        return len(self.warnings) > 0


@dataclass
class CompatibilityRequest:
    """兼容性检查请求"""
    source: str                            # 数据源名称
    market_type: MarketType               # 市场类型
    interval: DataInterval                # 时间间隔
    symbol: Optional[str] = None          # 可选：特定符号
    period_days: Optional[int] = None     # 可选：历史数据天数要求
    quality_requirement: Optional[DataQuality] = None  # 可选：数据质量要求
    requires_realtime: bool = False       # 是否需要实时数据
    requires_streaming: bool = False      # 是否需要流式数据


class CompatibilityChecker:
    """数据源兼容性检查器"""
    
    def __init__(self):
        """初始化兼容性检查器"""
        self.logger = setup_logger("CompatibilityChecker")
        
        # 兼容性检查权重配置
        self.weights = {
            'market_support': 0.30,      # 市场类型支持权重
            'interval_support': 0.25,    # 时间间隔支持权重
            'quality_match': 0.20,       # 数据质量匹配权重
            'feature_support': 0.15,     # 功能支持权重（实时、流式等）
            'history_availability': 0.10  # 历史数据可用性权重
        }
    
    def check_compatibility(self, request: CompatibilityRequest) -> CompatibilityResult:
        """
        执行全面的兼容性检查
        
        Args:
            request: 兼容性检查请求
            
        Returns:
            兼容性检查结果
        """
        self.logger.debug(f"Checking compatibility for source '{request.source}' "
                         f"with market type '{request.market_type}' and interval '{request.interval}'")
        
        issues = []
        warnings = []
        recommendations = []
        scores = {}
        
        # 检查数据源是否存在
        if not DataSourceRegistry.is_registered(request.source):
            available = DataSourceRegistry.list_sources()
            issues.append(f"Data source '{request.source}' is not registered")
            if available:
                recommendations.append(f"Available sources: {', '.join(available)}")
            
            return CompatibilityResult(
                is_compatible=False,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations,
                compatibility_score=0.0
            )
        
        # 获取数据源能力
        try:
            source_class = DataSourceRegistry.get(request.source)
            capabilities = source_class({}).get_capabilities()
        except Exception as e:
            issues.append(f"Failed to get capabilities for source '{request.source}': {e}")
            return CompatibilityResult(
                is_compatible=False,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations,
                compatibility_score=0.0
            )
        
        # 1. 检查市场类型支持
        market_score, market_issues, market_warnings = self._check_market_support(
            capabilities, request.market_type
        )
        issues.extend(market_issues)
        warnings.extend(market_warnings)
        scores['market_support'] = market_score
        
        # 2. 检查时间间隔支持
        interval_score, interval_issues, interval_warnings = self._check_interval_support(
            capabilities, request.interval
        )
        issues.extend(interval_issues)
        warnings.extend(interval_warnings)
        scores['interval_support'] = interval_score
        
        # 3. 检查数据质量匹配
        quality_score, quality_warnings = self._check_quality_match(
            capabilities, request.quality_requirement
        )
        warnings.extend(quality_warnings)
        scores['quality_match'] = quality_score
        
        # 4. 检查功能支持
        feature_score, feature_issues, feature_warnings = self._check_feature_support(
            capabilities, request.requires_realtime, request.requires_streaming
        )
        issues.extend(feature_issues)
        warnings.extend(feature_warnings)
        scores['feature_support'] = feature_score
        
        # 5. 检查历史数据可用性
        history_score, history_warnings = self._check_history_availability(
            capabilities, request.period_days
        )
        warnings.extend(history_warnings)
        scores['history_availability'] = history_score
        
        # 计算总体兼容性评分
        compatibility_score = sum(
            scores[category] * weight 
            for category, weight in self.weights.items()
        )
        
        # 生成建议
        recommendations.extend(self._generate_recommendations(
            request, capabilities, scores
        ))
        
        # 判断是否兼容（没有严重问题）
        is_compatible = len(issues) == 0
        
        result = CompatibilityResult(
            is_compatible=is_compatible,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            compatibility_score=compatibility_score
        )
        
        self.logger.debug(f"Compatibility check result for '{request.source}': "
                         f"compatible={is_compatible}, score={compatibility_score:.2f}")
        
        return result
    
    def _check_market_support(
        self, 
        capabilities: DataSourceCapabilities, 
        market_type: MarketType
    ) -> Tuple[float, List[str], List[str]]:
        """检查市场类型支持"""
        issues = []
        warnings = []
        
        if market_type in capabilities.supported_markets:
            score = 1.0
        else:
            score = 0.0
            supported_markets = [m.value for m in capabilities.supported_markets]
            issues.append(
                f"Market type '{market_type.value}' not supported. "
                f"Supported markets: {supported_markets}"
            )
        
        return score, issues, warnings
    
    def _check_interval_support(
        self, 
        capabilities: DataSourceCapabilities, 
        interval: DataInterval
    ) -> Tuple[float, List[str], List[str]]:
        """检查时间间隔支持"""
        issues = []
        warnings = []
        
        if interval in capabilities.supported_intervals:
            score = 1.0
        else:
            score = 0.0
            supported_intervals = [i.value for i in capabilities.supported_intervals]
            issues.append(
                f"Interval '{interval.value}' not supported. "
                f"Supported intervals: {supported_intervals}"
            )
            
            # 建议最接近的时间间隔
            closest_interval = self._find_closest_interval(interval, capabilities.supported_intervals)
            if closest_interval:
                warnings.append(f"Consider using closest supported interval: {closest_interval.value}")
        
        return score, issues, warnings
    
    def _check_quality_match(
        self, 
        capabilities: DataSourceCapabilities, 
        quality_requirement: Optional[DataQuality]
    ) -> Tuple[float, List[str]]:
        """检查数据质量匹配"""
        warnings = []
        
        if quality_requirement is None:
            return 1.0, warnings
        
        source_quality = capabilities.data_quality
        
        # 数据质量等级映射
        quality_levels = {
            DataQuality.HIGH: 3,
            DataQuality.MEDIUM: 2,
            DataQuality.LOW: 1,
            DataQuality.UNKNOWN: 0
        }
        
        source_level = quality_levels.get(source_quality, 0)
        required_level = quality_levels.get(quality_requirement, 0)
        
        if source_level >= required_level:
            score = 1.0
        else:
            score = source_level / required_level if required_level > 0 else 0.0
            warnings.append(
                f"Data quality '{source_quality.value}' may not meet requirement '{quality_requirement.value}'"
            )
        
        return score, warnings
    
    def _check_feature_support(
        self, 
        capabilities: DataSourceCapabilities, 
        requires_realtime: bool, 
        requires_streaming: bool
    ) -> Tuple[float, List[str], List[str]]:
        """检查功能支持"""
        issues = []
        warnings = []
        score = 1.0
        
        if requires_realtime and not capabilities.has_realtime:
            score *= 0.5
            issues.append("Real-time data required but not supported")
        
        if requires_streaming and not capabilities.has_streaming:
            score *= 0.5
            issues.append("Streaming data required but not supported")
            warnings.append("Consider using polling instead of streaming")
        
        # 检查认证要求
        if capabilities.requires_auth:
            warnings.append("This data source requires authentication")
        
        # 检查是否免费
        if not capabilities.is_free:
            warnings.append("This data source may have usage costs")
        
        return score, issues, warnings
    
    def _check_history_availability(
        self, 
        capabilities: DataSourceCapabilities, 
        period_days: Optional[int]
    ) -> Tuple[float, List[str]]:
        """检查历史数据可用性"""
        warnings = []
        
        if period_days is None or capabilities.max_history_days is None:
            return 1.0, warnings
        
        max_days = capabilities.max_history_days
        
        if period_days <= max_days:
            score = 1.0
        else:
            score = max_days / period_days
            warnings.append(
                f"Requested {period_days} days of history, but source only provides {max_days} days"
            )
        
        return score, warnings
    
    def _find_closest_interval(
        self, 
        target: DataInterval, 
        supported: List[DataInterval]
    ) -> Optional[DataInterval]:
        """找到最接近的支持时间间隔"""
        # 简化实现：返回第一个找到的间隔
        # 在实际应用中可以根据时间长度进行更智能的匹配
        return supported[0] if supported else None
    
    def _generate_recommendations(
        self, 
        request: CompatibilityRequest, 
        capabilities: DataSourceCapabilities, 
        scores: Dict[str, float]
    ) -> List[str]:
        """生成兼容性建议"""
        recommendations = []
        
        # 基于评分提供建议
        if scores.get('market_support', 0) == 0:
            # 推荐支持该市场类型的其他数据源
            alternative_sources = self._find_alternative_sources(request.market_type)
            if alternative_sources:
                recommendations.append(
                    f"Try alternative sources for {request.market_type.value}: {', '.join(alternative_sources)}"
                )
        
        if scores.get('interval_support', 0) == 0:
            # 推荐支持该时间间隔的其他数据源
            alternative_sources = self._find_alternative_sources_for_interval(request.interval)
            if alternative_sources:
                recommendations.append(
                    f"Try alternative sources for {request.interval.value}: {', '.join(alternative_sources)}"
                )
        
        if scores.get('quality_match', 0) < 0.8:
            recommendations.append("Consider upgrading to a higher quality data source")
        
        # 如果有延迟信息，提供延迟相关建议
        if capabilities.latency_ms and capabilities.latency_ms > 1000:
            recommendations.append(f"Note: This source has high latency ({capabilities.latency_ms}ms)")
        
        return recommendations
    
    def _find_alternative_sources(self, market_type: MarketType) -> List[str]:
        """查找支持指定市场类型的替代数据源"""
        alternatives = []
        
        for source_name in DataSourceRegistry.list_sources():
            try:
                source_class = DataSourceRegistry.get(source_name)
                capabilities = source_class({}).get_capabilities()
                
                if market_type in capabilities.supported_markets:
                    alternatives.append(source_name)
                    
            except Exception:
                continue
        
        return alternatives
    
    def _find_alternative_sources_for_interval(self, interval: DataInterval) -> List[str]:
        """查找支持指定时间间隔的替代数据源"""
        alternatives = []
        
        for source_name in DataSourceRegistry.list_sources():
            try:
                source_class = DataSourceRegistry.get(source_name)
                capabilities = source_class({}).get_capabilities()
                
                if interval in capabilities.supported_intervals:
                    alternatives.append(source_name)
                    
            except Exception:
                continue
        
        return alternatives
    
    def check_multiple_sources(
        self, 
        sources: List[str], 
        market_type: MarketType, 
        interval: DataInterval,
        **kwargs
    ) -> Dict[str, CompatibilityResult]:
        """
        批量检查多个数据源的兼容性
        
        Args:
            sources: 数据源列表
            market_type: 市场类型
            interval: 时间间隔
            **kwargs: 其他兼容性检查参数
            
        Returns:
            数据源名称到兼容性结果的映射
        """
        results = {}
        
        for source in sources:
            request = CompatibilityRequest(
                source=source,
                market_type=market_type,
                interval=interval,
                **kwargs
            )
            results[source] = self.check_compatibility(request)
        
        return results
    
    def rank_sources_by_compatibility(
        self, 
        sources: List[str], 
        market_type: MarketType, 
        interval: DataInterval,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """
        根据兼容性评分排序数据源
        
        Args:
            sources: 数据源列表
            market_type: 市场类型
            interval: 时间间隔
            **kwargs: 其他兼容性检查参数
            
        Returns:
            按兼容性评分排序的(源名称, 评分)元组列表
        """
        results = self.check_multiple_sources(sources, market_type, interval, **kwargs)
        
        # 只包含兼容的数据源，按评分排序
        compatible_sources = [
            (source, result.compatibility_score)
            for source, result in results.items()
            if result.is_compatible
        ]
        
        return sorted(compatible_sources, key=lambda x: x[1], reverse=True)


# 全局实例
_global_compatibility_checker: Optional[CompatibilityChecker] = None


def get_compatibility_checker() -> CompatibilityChecker:
    """获取全局兼容性检查器实例"""
    global _global_compatibility_checker
    if _global_compatibility_checker is None:
        _global_compatibility_checker = CompatibilityChecker()
    return _global_compatibility_checker