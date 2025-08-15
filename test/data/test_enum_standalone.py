#!/usr/bin/env python3
"""
独立测试DataSource枚举功能
避免循环依赖问题
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# 直接导入基础模块避免循环依赖
from src.data.sources.base import DataSource, DataQuality, MarketType


def test_enum_basic_functionality():
    """测试枚举基本功能"""
    print("🔧 测试DataSource枚举基本功能...")
    
    # 测试枚举值
    assert DataSource.YFINANCE.value == "yfinance"
    assert DataSource.TRUEFX.value == "truefx"
    assert DataSource.OANDA.value == "oanda"
    assert DataSource.FXMINUTE.value == "fxminute"
    assert DataSource.HISTDATA.value == "histdata"
    assert DataSource.AUTO.value == "auto"
    print("✅ 枚举值测试通过")
    
    # 测试字符串转换
    assert DataSource.from_string("yfinance") == DataSource.YFINANCE
    assert DataSource.from_string("TRUEFX") == DataSource.TRUEFX
    assert DataSource.from_string("yahoo") == DataSource.YFINANCE
    assert DataSource.from_string("fx_minute") == DataSource.FXMINUTE
    print("✅ 字符串转换测试通过")
    
    # 测试错误处理
    try:
        DataSource.from_string("invalid_source")
        assert False, "应该抛出ValueError"
    except ValueError as e:
        assert "Unknown data source" in str(e)
    print("✅ 错误处理测试通过")
    
    # 测试友好名称
    assert DataSource.YFINANCE.display_name == "Yahoo Finance"
    assert DataSource.TRUEFX.display_name == "TrueFX"
    assert DataSource.AUTO.display_name == "Auto-Select"
    print("✅ 友好名称测试通过")
    
    # 测试支持的市场类型
    yfinance_markets = DataSource.YFINANCE.supported_markets
    assert MarketType.STOCK in yfinance_markets
    assert MarketType.CRYPTO in yfinance_markets
    assert MarketType.FOREX not in yfinance_markets
    
    truefx_markets = DataSource.TRUEFX.supported_markets
    assert truefx_markets == [MarketType.FOREX]
    
    auto_markets = DataSource.AUTO.supported_markets
    assert len(auto_markets) == len(list(MarketType))
    print("✅ 支持的市场类型测试通过")
    
    # 测试数据质量
    assert DataSource.YFINANCE.data_quality == DataQuality.MEDIUM
    assert DataSource.TRUEFX.data_quality == DataQuality.HIGH
    assert DataSource.AUTO.data_quality == DataQuality.UNKNOWN
    print("✅ 数据质量测试通过")


def test_factory_basic():
    """测试工厂基本功能（不涉及实际数据源类）"""
    print("\n🏭 测试DataSourceRegistry基本功能...")
    
    from src.data.sources.factory import DataSourceRegistry
    from unittest.mock import MagicMock
    
    # 清理注册表
    DataSourceRegistry.clear()
    
    # 测试注册
    mock_source = MagicMock()
    DataSourceRegistry.register(DataSource.YFINANCE, mock_source)
    assert DataSourceRegistry.is_registered(DataSource.YFINANCE)
    assert DataSourceRegistry.is_registered("yfinance")
    print("✅ 注册功能测试通过")
    
    # 测试获取
    assert DataSourceRegistry.get(DataSource.YFINANCE) == mock_source
    assert DataSourceRegistry.get("yfinance") == mock_source
    assert DataSourceRegistry.get("YFINANCE") == mock_source
    print("✅ 获取功能测试通过")
    
    # 测试列表
    sources = DataSourceRegistry.list_sources()
    assert DataSource.YFINANCE in sources
    assert len(sources) == 1
    print("✅ 列表功能测试通过")
    
    # 测试注销
    DataSourceRegistry.unregister(DataSource.YFINANCE)
    assert not DataSourceRegistry.is_registered(DataSource.YFINANCE)
    print("✅ 注销功能测试通过")


if __name__ == '__main__':
    try:
        test_enum_basic_functionality()
        test_factory_basic()
        print("\n🎉 所有测试通过！DataSource枚举实现正确。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)