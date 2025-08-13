# 数据源工厂模式设计文档

## 1. 概述

本文档描述了使用抽象工厂模式重构TensorTrade数据下载模块的设计方案，支持多种金融数据源的灵活切换和扩展。

## 2. 设计目标

- **可扩展性**: 轻松添加新的数据源而不影响现有代码
- **统一接口**: 所有数据源通过统一接口访问
- **配置驱动**: 通过配置文件切换数据源
- **向后兼容**: 保持与现有系统的兼容性
- **错误处理**: 统一的错误处理和重试机制

## 3. 支持的数据源

| 数据源 | 数据类型 | 时间范围 | 访问方式 | 特点 |
|--------|----------|----------|----------|------|
| YFinance | OHLCV | 1970年起 | 免费API | 股票为主，简单易用 |
| TrueFX | Tick级 | 2009年起 | 免费注册 | 外汇数据，毫秒精度 |
| HistData | Tick级 | 2000年起 | 免费下载 | 需手动下载，历史数据丰富 |
| Dukascopy | Tick/分钟 | 2003年起 | 免费API | 专业外汇数据 |
| Oanda | Tick/分钟 | 2005年起 | 注册账户 | 实时+历史，API完善 |
| FXCM | Tick/分钟 | 2000年起 | 注册账户 | 专业交易平台 |

## 4. 架构设计

### 4.1 类图

```
                    ┌─────────────────────┐
                    │ DataSourceFactory   │
                    │ (Abstract Factory)  │
                    └─────────┬───────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
    ┌───────────▼──────────┐    ┌──────────▼──────────┐
    │ AbstractDataSource   │    │ DataSourceRegistry  │
    │   (Abstract Base)    │    │    (Registry)       │
    └───────────┬──────────┘    └─────────────────────┘
                │
    ┌───────────┼───────────────────────────┐
    │           │           │               │
┌───▼───┐ ┌────▼───┐ ┌────▼───┐    ┌─────▼────┐
│YFinance│ │TrueFX  │ │Oanda   │    │Dukascopy │
│Source  │ │Source  │ │Source  │... │Source    │
└────────┘ └────────┘ └────────┘    └──────────┘
```

### 4.2 核心组件

#### 4.2.1 抽象基类
```python
class AbstractDataSource(ABC):
    """数据源抽象基类"""
    
    @abstractmethod
    def fetch_historical_data(self, symbol, start_date, end_date, interval):
        """获取历史数据"""
        pass
    
    @abstractmethod
    def fetch_realtime_data(self, symbol):
        """获取实时数据"""
        pass
    
    @abstractmethod
    def validate_symbol(self, symbol):
        """验证标的代码"""
        pass
    
    @abstractmethod
    def get_supported_intervals(self):
        """获取支持的时间间隔"""
        pass
```

#### 4.2.2 工厂类
```python
class DataSourceFactory:
    """数据源工厂"""
    
    @staticmethod
    def create_data_source(source_type: str, config: Dict) -> AbstractDataSource:
        """创建数据源实例"""
        pass
    
    @staticmethod
    def register_source(source_type: str, source_class: Type[AbstractDataSource]):
        """注册新数据源"""
        pass
```

## 5. 数据源实现细节

### 5.1 YFinance数据源
- **优势**: 免费、易用、数据完整
- **限制**: 主要支持股票，外汇数据有限
- **实现要点**: 
  - 利用现有DataManager代码
  - 保持缓存机制
  - 支持代理设置

### 5.2 TrueFX数据源
- **优势**: 高精度tick数据，毫秒级时间戳
- **限制**: 仅外汇，需注册
- **实现要点**:
  - HTTP API集成
  - 认证会话管理
  - CSV/JSON格式解析
  - 增量更新支持

### 5.3 Oanda数据源
- **优势**: 专业API，实时+历史数据
- **限制**: 需要账户（可用免费模拟账户）
- **实现要点**:
  - oandapyV20库集成
  - OAuth认证
  - 流式数据支持
  - 速率限制处理

### 5.4 Dukascopy数据源
- **优势**: 专业外汇数据，历史悠久
- **限制**: API较复杂
- **实现要点**:
  - JForex API集成或HTTP下载
  - 二进制数据解析
  - 时区转换（GMT）

### 5.5 HistData数据源
- **优势**: 免费，历史数据丰富
- **限制**: 需手动下载，无实时数据
- **实现要点**:
  - 本地文件管理
  - CSV解析优化
  - 数据完整性检查

### 5.6 FXCM数据源
- **优势**: 专业交易平台数据
- **限制**: 需要账户
- **实现要点**:
  - REST API集成
  - WebSocket支持
  - 订单簿数据

## 6. 配置管理

### 6.1 配置文件结构
```yaml
data_sources:
  default: yfinance
  
  sources:
    yfinance:
      enabled: true
      cache_dir: ./cache/yfinance
      proxy: socks5://127.0.0.1:7891
      
    truefx:
      enabled: true
      username: ${TRUEFX_USERNAME}
      password: ${TRUEFX_PASSWORD}
      cache_dir: ./cache/truefx
      
    oanda:
      enabled: true
      account_id: ${OANDA_ACCOUNT_ID}
      access_token: ${OANDA_TOKEN}
      environment: practice  # practice/live
      
    dukascopy:
      enabled: false
      cache_dir: ./cache/dukascopy
      
    histdata:
      enabled: true
      data_dir: ./data/histdata
      auto_download: false
      
    fxcm:
      enabled: false
      token: ${FXCM_TOKEN}
      server: demo  # demo/real
```

### 6.2 环境变量
```bash
# 数据源凭证
TRUEFX_USERNAME=your_username
TRUEFX_PASSWORD=your_password
OANDA_ACCOUNT_ID=your_account
OANDA_TOKEN=your_token
FXCM_TOKEN=your_token

# 代理设置
USE_PROXY=true
PROXY_HOST=127.0.0.1
PROXY_PORT=7891
```

## 7. 数据格式统一

### 7.1 统一输出格式
```python
@dataclass
class MarketData:
    """统一的市场数据格式"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    tick_count: Optional[int] = None
```

### 7.2 数据转换器
```python
class DataConverter:
    """数据格式转换器"""
    
    @staticmethod
    def to_dataframe(data: List[MarketData]) -> pd.DataFrame:
        """转换为DataFrame"""
        pass
    
    @staticmethod
    def from_tick_to_ohlc(ticks: List[Dict], interval: str) -> pd.DataFrame:
        """Tick数据聚合为OHLC"""
        pass
```

## 8. 错误处理策略

### 8.1 错误类型
- **网络错误**: 自动重试，指数退避
- **认证错误**: 提示用户检查凭证
- **速率限制**: 自动降速，队列管理
- **数据质量**: 验证和清洗，记录异常

### 8.2 降级策略
```python
class DataSourceFallback:
    """数据源降级策略"""
    
    def __init__(self, primary_source, fallback_sources):
        self.primary = primary_source
        self.fallbacks = fallback_sources
    
    def fetch_with_fallback(self, *args, **kwargs):
        """带降级的数据获取"""
        try:
            return self.primary.fetch(*args, **kwargs)
        except Exception as e:
            for fallback in self.fallbacks:
                try:
                    return fallback.fetch(*args, **kwargs)
                except:
                    continue
            raise e
```

## 9. 性能优化

### 9.1 缓存策略
- **多级缓存**: 内存 → 本地文件 → 远程API
- **智能预取**: 根据使用模式预加载数据
- **增量更新**: 仅获取新数据，合并历史

### 9.2 并发处理
- **异步IO**: 使用asyncio进行并发请求
- **连接池**: 复用HTTP连接
- **批量请求**: 合并多个请求

## 10. 测试策略

### 10.1 单元测试
- 每个数据源的独立测试
- Mock外部API响应
- 数据验证测试

### 10.2 集成测试
- 多数据源切换测试
- 降级策略测试
- 性能基准测试

## 11. 迁移计划

### 第一阶段：基础架构
1. 创建抽象基类和工厂
2. 实现注册机制
3. 配置管理系统

### 第二阶段：数据源实现
1. YFinance适配（保持兼容）
2. TrueFX实现
3. Oanda实现
4. 其他数据源逐步添加

### 第三阶段：集成优化
1. 统一错误处理
2. 性能优化
3. 监控和日志

## 12. 使用示例

```python
# 创建数据源
from src.data.sources import DataSourceFactory

# 使用默认数据源
source = DataSourceFactory.create_default()
data = source.fetch_historical_data('EURUSD', '2024-01-01', '2024-12-31', '1h')

# 指定数据源
oanda_source = DataSourceFactory.create('oanda', config)
data = oanda_source.fetch_realtime_data('EURUSD')

# 多数据源聚合
aggregator = MultiSourceAggregator(['truefx', 'oanda'])
best_price = aggregator.get_best_price('EURUSD')
```

## 13. 监控和维护

### 13.1 监控指标
- API调用次数和延迟
- 数据质量分数
- 缓存命中率
- 错误率和类型分布

### 13.2 维护任务
- 定期清理过期缓存
- 更新API版本
- 监控数据源可用性
- 性能调优

## 14. 未来扩展

- 加密货币数据源（Binance, Coinbase）
- 新闻和情绪数据
- 基本面数据集成
- WebSocket实时流支持
- 分布式缓存（Redis）

## 15. 参考资源

- [YFinance Documentation](https://github.com/ranaroussi/yfinance)
- [TrueFX API Guide](https://www.truefx.com/dev/api/)
- [Oanda v20 API](https://developer.oanda.com/rest-live-v20/)
- [Dukascopy Historical Data](https://www.dukascopy.com/swiss/english/marketwatch/historical/)
- [HistData Downloads](https://www.histdata.com/)
- [FXCM REST API](https://github.com/fxcm/RestAPI)