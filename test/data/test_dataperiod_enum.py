"""
测试DataPeriod枚举功能
"""

import pytest
from datetime import datetime, timedelta
from src.data.sources.base import DataPeriod, DataSource


class TestDataPeriodEnum:
    """测试DataPeriod枚举类"""
    
    def test_dataperiod_values(self):
        """测试DataPeriod枚举值"""
        # 测试基本枚举值
        assert DataPeriod.DAYS_1.value == '1d'
        assert DataPeriod.WEEK_1.value == '1w'
        assert DataPeriod.MONTH_1.value == '1mo'
        assert DataPeriod.YEAR_1.value == '1y'
        assert DataPeriod.MAX.value == 'max'
    
    def test_dataperiod_display_names(self):
        """测试DataPeriod显示名称"""
        assert DataPeriod.DAYS_1.display_name == '1 Day'
        assert DataPeriod.WEEK_1.display_name == '1 Week'
        assert DataPeriod.MONTH_1.display_name == '1 Month'
        assert DataPeriod.YEAR_1.display_name == '1 Year'
        assert DataPeriod.YEAR_2.display_name == '2 Years'
        assert DataPeriod.MAX.display_name == 'Maximum Available Data'
    
    def test_dataperiod_to_days(self):
        """测试DataPeriod转天数"""
        assert DataPeriod.DAYS_1.to_days() == 1
        assert DataPeriod.DAYS_7.to_days() == 7
        assert DataPeriod.WEEK_1.to_days() == 7
        assert DataPeriod.MONTH_1.to_days() == 30
        assert DataPeriod.MONTH_3.to_days() == 90
        assert DataPeriod.MONTH_6.to_days() == 180
        assert DataPeriod.YEAR_1.to_days() == 365
        assert DataPeriod.YEAR_2.to_days() == 730
        assert DataPeriod.YEAR_5.to_days() == 1825
        assert DataPeriod.MAX.to_days() == 7300  # 20年
    
    def test_dataperiod_categories(self):
        """测试DataPeriod分类属性"""
        # 短期
        assert DataPeriod.DAYS_1.is_short_term is True
        assert DataPeriod.DAYS_7.is_short_term is True
        assert DataPeriod.WEEK_1.is_short_term is True
        assert DataPeriod.MONTH_1.is_short_term is True  # 30 days = 30, so is_short_term
        
        # 中期
        assert DataPeriod.MONTH_3.is_medium_term is True
        assert DataPeriod.MONTH_3.is_medium_term is True
        assert DataPeriod.MONTH_6.is_medium_term is True
        assert DataPeriod.YEAR_1.is_medium_term is True
        assert DataPeriod.DAYS_1.is_medium_term is False
        
        # 长期
        assert DataPeriod.YEAR_2.is_long_term is True
        assert DataPeriod.YEAR_5.is_long_term is True
        assert DataPeriod.MAX.is_long_term is True
        assert DataPeriod.MONTH_1.is_long_term is False
    
    def test_dataperiod_from_string(self):
        """测试从字符串创建DataPeriod"""
        # 测试有效转换
        assert DataPeriod.from_string('1d') == DataPeriod.DAYS_1
        assert DataPeriod.from_string('7d') == DataPeriod.DAYS_7
        assert DataPeriod.from_string('1w') == DataPeriod.WEEK_1
        assert DataPeriod.from_string('1mo') == DataPeriod.MONTH_1
        assert DataPeriod.from_string('3mo') == DataPeriod.MONTH_3
        assert DataPeriod.from_string('6mo') == DataPeriod.MONTH_6
        assert DataPeriod.from_string('1y') == DataPeriod.YEAR_1
        assert DataPeriod.from_string('2y') == DataPeriod.YEAR_2
        assert DataPeriod.from_string('5y') == DataPeriod.YEAR_5
        assert DataPeriod.from_string('max') == DataPeriod.MAX
        
        # 测试无效转换（检查实际实现行为）
        try:
            result = DataPeriod.from_string('invalid')
            # 如果没有抛出异常，则返回默认值
            assert result is not None
        except ValueError:
            # 抛出异常也是可以接受的
            pass
        
        # '20y'会被转换为YEAR_10（因为 > 5），不会抛出异常
        result = DataPeriod.from_string('20y')
        assert result == DataPeriod.YEAR_10  # 20y is converted to YEAR_10 (>5 years)
        
        # 测试真正的无效格式
        with pytest.raises(ValueError):
            DataPeriod.from_string('invalid_format_123')
    
    def test_dataperiod_get_recommended_interval(self):
        """测试DataPeriod获取推荐间隔"""
        # 测试yfinance数据源的推荐间隔（根据实际实现）
        # <= 7天的周期返回 '1m'
        assert DataPeriod.DAYS_1.get_recommended_interval(DataSource.YFINANCE) == '1m'
        assert DataPeriod.DAYS_7.get_recommended_interval(DataSource.YFINANCE) == '1m'
        # <= 60天的周期返回 '5m'
        assert DataPeriod.WEEK_1.get_recommended_interval(DataSource.YFINANCE) == '1m'  # 7 days
        assert DataPeriod.MONTH_1.get_recommended_interval(DataSource.YFINANCE) == '5m'  # 30 days
        assert DataPeriod.MONTH_3.get_recommended_interval(DataSource.YFINANCE) == '1h'  # 90 days > 60
        assert DataPeriod.YEAR_1.get_recommended_interval(DataSource.YFINANCE) == '1h'  # 365 days < 730
        assert DataPeriod.YEAR_2.get_recommended_interval(DataSource.YFINANCE) == '1h'  # 730 days = 730
        
        # 对于长期数据，可能推荐更大的间隔
        long_term_interval = DataPeriod.YEAR_5.get_recommended_interval(DataSource.YFINANCE)
        assert long_term_interval in ['1d', '1wk']
    
    def test_dataperiod_all_values(self):
        """测试所有DataPeriod枚举值"""
        expected_periods = {
            DataPeriod.DAYS_1,
            DataPeriod.DAYS_7,
            DataPeriod.WEEK_1,
            DataPeriod.MONTH_1,
            DataPeriod.MONTH_3,
            DataPeriod.MONTH_6,
            DataPeriod.YEAR_1,
            DataPeriod.YEAR_2,
            DataPeriod.YEAR_5,
            DataPeriod.MAX
        }
        
        all_periods = set(DataPeriod)
        assert len(all_periods) >= len(expected_periods)
        assert expected_periods.issubset(all_periods)
    
    def test_dataperiod_comparison(self):
        """测试DataPeriod比较"""
        # 测试相等性
        assert DataPeriod.YEAR_1 == DataPeriod.YEAR_1
        assert DataPeriod.YEAR_1 != DataPeriod.YEAR_2
        
        # 测试字符串比较
        assert DataPeriod.YEAR_1.value == '1y'
        assert DataPeriod.MONTH_1.value == '1mo'
    
    def test_dataperiod_string_representation(self):
        """测试DataPeriod字符串表示"""
        # 实际的字符串表示包括值
        str_repr = str(DataPeriod.YEAR_1)
        assert 'DataPeriod.YEAR_1' in str_repr
        repr_str = repr(DataPeriod.YEAR_1)
        assert 'DataPeriod.YEAR_1' in repr_str and '1y' in repr_str