#!/usr/bin/env python3
"""
测试DataSource枚举实现
验证枚举的创建、转换、工厂模式等功能
"""

import unittest
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.sources.base import DataSource, DataQuality, MarketType
from src.data.sources.factory import DataSourceRegistry, DataSourceFactory


class TestDataSourceEnum(unittest.TestCase):
    """DataSource枚举测试"""

    def test_enum_values(self):
        """测试枚举值"""
        self.assertEqual(DataSource.YFINANCE.value, "yfinance")
        self.assertEqual(DataSource.TRUEFX.value, "truefx")
        self.assertEqual(DataSource.OANDA.value, "oanda")
        self.assertEqual(DataSource.FXMINUTE.value, "fxminute")
        self.assertEqual(DataSource.HISTDATA.value, "histdata")
        self.assertEqual(DataSource.AUTO.value, "auto")

    def test_from_string_standard(self):
        """测试标准字符串转换"""
        self.assertEqual(DataSource.from_string("yfinance"), DataSource.YFINANCE)
        self.assertEqual(DataSource.from_string("truefx"), DataSource.TRUEFX)
        self.assertEqual(DataSource.from_string("oanda"), DataSource.OANDA)
        self.assertEqual(DataSource.from_string("fxminute"), DataSource.FXMINUTE)
        self.assertEqual(DataSource.from_string("histdata"), DataSource.HISTDATA)
        self.assertEqual(DataSource.from_string("auto"), DataSource.AUTO)

    def test_from_string_case_insensitive(self):
        """测试大小写不敏感转换"""
        self.assertEqual(DataSource.from_string("YFINANCE"), DataSource.YFINANCE)
        self.assertEqual(DataSource.from_string("TrueFX"), DataSource.TRUEFX)
        self.assertEqual(DataSource.from_string("Oanda"), DataSource.OANDA)

    def test_from_string_aliases(self):
        """测试别名转换"""
        self.assertEqual(DataSource.from_string("yahoo"), DataSource.YFINANCE)
        self.assertEqual(DataSource.from_string("yf"), DataSource.YFINANCE)
        self.assertEqual(DataSource.from_string("yahoo_finance"), DataSource.YFINANCE)
        self.assertEqual(DataSource.from_string("fx_minute"), DataSource.FXMINUTE)
        self.assertEqual(DataSource.from_string("fx-minute"), DataSource.FXMINUTE)
        self.assertEqual(DataSource.from_string("automatic"), DataSource.AUTO)
        self.assertEqual(DataSource.from_string("default"), DataSource.AUTO)

    def test_from_string_invalid(self):
        """测试无效字符串转换"""
        with self.assertRaises(ValueError) as context:
            DataSource.from_string("invalid_source")
        
        self.assertIn("Unknown data source", str(context.exception))

    def test_from_enum_passthrough(self):
        """测试枚举直接传入"""
        self.assertEqual(DataSource.from_string(DataSource.YFINANCE), DataSource.YFINANCE)

    def test_display_name(self):
        """测试友好显示名称"""
        self.assertEqual(DataSource.YFINANCE.display_name, "Yahoo Finance")
        self.assertEqual(DataSource.TRUEFX.display_name, "TrueFX")
        self.assertEqual(DataSource.OANDA.display_name, "Oanda")
        self.assertEqual(DataSource.FXMINUTE.display_name, "FX-1-Minute-Data")
        self.assertEqual(DataSource.HISTDATA.display_name, "HistData")
        self.assertEqual(DataSource.AUTO.display_name, "Auto-Select")

    def test_supported_markets(self):
        """测试支持的市场类型"""
        yfinance_markets = DataSource.YFINANCE.supported_markets
        self.assertIn(MarketType.STOCK, yfinance_markets)
        self.assertIn(MarketType.CRYPTO, yfinance_markets)
        self.assertIn(MarketType.ETF, yfinance_markets)
        self.assertNotIn(MarketType.FOREX, yfinance_markets)

        truefx_markets = DataSource.TRUEFX.supported_markets
        self.assertEqual(truefx_markets, [MarketType.FOREX])

        oanda_markets = DataSource.OANDA.supported_markets
        self.assertIn(MarketType.FOREX, oanda_markets)
        self.assertIn(MarketType.COMMODITIES, oanda_markets)

        auto_markets = DataSource.AUTO.supported_markets
        self.assertEqual(len(auto_markets), len(MarketType))  # AUTO支持所有市场

    def test_data_quality(self):
        """测试数据质量等级"""
        self.assertEqual(DataSource.YFINANCE.data_quality, DataQuality.MEDIUM)
        self.assertEqual(DataSource.TRUEFX.data_quality, DataQuality.HIGH)
        self.assertEqual(DataSource.OANDA.data_quality, DataQuality.HIGH)
        self.assertEqual(DataSource.FXMINUTE.data_quality, DataQuality.HIGH)
        self.assertEqual(DataSource.HISTDATA.data_quality, DataQuality.MEDIUM)
        self.assertEqual(DataSource.AUTO.data_quality, DataQuality.UNKNOWN)


class TestDataSourceRegistryEnum(unittest.TestCase):
    """DataSourceRegistry枚举支持测试"""

    def setUp(self):
        """每个测试前清理注册表"""
        DataSourceRegistry.clear()

    def tearDown(self):
        """每个测试后清理注册表"""
        DataSourceRegistry.clear()

    def test_register_with_enum(self):
        """测试使用枚举注册数据源"""
        from unittest.mock import MagicMock
        
        mock_source = MagicMock()
        DataSourceRegistry.register(DataSource.YFINANCE, mock_source)
        
        self.assertTrue(DataSourceRegistry.is_registered(DataSource.YFINANCE))
        self.assertEqual(DataSourceRegistry.get(DataSource.YFINANCE), mock_source)

    def test_register_with_string(self):
        """测试使用字符串注册数据源"""
        from unittest.mock import MagicMock
        
        mock_source = MagicMock()
        DataSourceRegistry.register("yfinance", mock_source)
        
        self.assertTrue(DataSourceRegistry.is_registered("yfinance"))
        self.assertTrue(DataSourceRegistry.is_registered(DataSource.YFINANCE))
        self.assertEqual(DataSourceRegistry.get("yfinance"), mock_source)

    def test_get_with_mixed_types(self):
        """测试混合类型获取"""
        from unittest.mock import MagicMock
        
        mock_source = MagicMock()
        DataSourceRegistry.register(DataSource.TRUEFX, mock_source)
        
        # 使用枚举和字符串都应该能获取到相同的结果
        self.assertEqual(DataSourceRegistry.get(DataSource.TRUEFX), mock_source)
        self.assertEqual(DataSourceRegistry.get("truefx"), mock_source)
        self.assertEqual(DataSourceRegistry.get("TRUEFX"), mock_source)
        self.assertEqual(DataSourceRegistry.get("TrueFX"), mock_source)

    def test_list_sources_returns_enums(self):
        """测试list_sources返回枚举列表"""
        from unittest.mock import MagicMock
        
        mock_source1 = MagicMock()
        mock_source2 = MagicMock()
        
        DataSourceRegistry.register(DataSource.YFINANCE, mock_source1)
        DataSourceRegistry.register(DataSource.TRUEFX, mock_source2)
        
        sources = DataSourceRegistry.list_sources()
        self.assertIsInstance(sources, list)
        self.assertIn(DataSource.YFINANCE, sources)
        self.assertIn(DataSource.TRUEFX, sources)
        self.assertEqual(len(sources), 2)

    def test_invalid_string_registration(self):
        """测试无效字符串注册"""
        from unittest.mock import MagicMock
        
        mock_source = MagicMock()
        # 应该不抛出异常，但会打印警告
        DataSourceRegistry.register("invalid_source", mock_source)
        
        # 无效源不应该被注册
        self.assertFalse(DataSourceRegistry.is_registered("invalid_source"))


class TestDataSourceFactoryEnum(unittest.TestCase):
    """DataSourceFactory枚举支持测试"""

    def setUp(self):
        """每个测试前清理注册表"""
        DataSourceRegistry.clear()

    def tearDown(self):
        """每个测试后清理注册表"""
        DataSourceRegistry.clear()

    def test_create_with_enum(self):
        """测试使用枚举创建数据源"""
        from unittest.mock import MagicMock
        
        mock_source_class = MagicMock()
        mock_instance = MagicMock()
        mock_source_class.return_value = mock_instance
        
        DataSourceRegistry.register(DataSource.YFINANCE, mock_source_class)
        
        result = DataSourceFactory.create_data_source(DataSource.YFINANCE, {"test": "config"})
        
        self.assertEqual(result, mock_instance)
        mock_source_class.assert_called_once()
        
        # 检查传入的配置
        call_args = mock_source_class.call_args[0][0]
        self.assertEqual(call_args['name'], 'yfinance')
        self.assertEqual(call_args['source_enum'], DataSource.YFINANCE)
        self.assertEqual(call_args['test'], 'config')

    def test_create_with_string(self):
        """测试使用字符串创建数据源"""
        from unittest.mock import MagicMock
        
        mock_source_class = MagicMock()
        mock_instance = MagicMock()
        mock_source_class.return_value = mock_instance
        
        DataSourceRegistry.register(DataSource.TRUEFX, mock_source_class)
        
        result = DataSourceFactory.create_data_source("truefx")
        
        self.assertEqual(result, mock_instance)
        mock_source_class.assert_called_once()
        
        # 检查传入的配置
        call_args = mock_source_class.call_args[0][0]
        self.assertEqual(call_args['name'], 'truefx')
        self.assertEqual(call_args['source_enum'], DataSource.TRUEFX)

    def test_create_with_invalid_source(self):
        """测试使用无效数据源创建"""
        with self.assertRaises(ValueError) as context:
            DataSourceFactory.create_data_source("invalid_source")
        
        self.assertIn("Unknown data source", str(context.exception))

    def test_create_unregistered_source(self):
        """测试创建未注册的数据源"""
        with self.assertRaises(ValueError) as context:
            DataSourceFactory.create_data_source(DataSource.OANDA)
        
        self.assertIn("Data source not registered", str(context.exception))


if __name__ == '__main__':
    unittest.main()