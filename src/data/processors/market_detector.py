"""
市场类型自动检测器

根据symbol格式自动识别市场类型
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..sources.base import MarketType


@dataclass
class MarketPattern:
    """市场模式配置"""
    name: str
    market_type: MarketType
    patterns: List[str]          # 正则表达式模式
    exact_matches: List[str]     # 精确匹配
    length_rules: List[int]      # 长度规则
    priority: int = 50          # 优先级（数字越大优先级越高）


class MarketTypeDetector:
    """市场类型自动检测器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化市场类型检测器
        
        Args:
            config: 可选的配置字典，用于自定义检测规则
        """
        self.config = config or {}
        self.patterns = self._load_default_patterns()
        
        # 如果有配置，加载自定义模式
        if 'market_patterns' in self.config:
            self._load_custom_patterns(self.config['market_patterns'])
    
    def detect(self, symbol: str) -> MarketType:
        """
        检测symbol的市场类型
        
        Args:
            symbol: 交易代码
            
        Returns:
            检测到的市场类型，默认为STOCK
        """
        symbol_clean = symbol.upper().strip()
        
        # 按优先级排序后检测
        sorted_patterns = sorted(self.patterns, key=lambda x: x.priority, reverse=True)
        
        for pattern in sorted_patterns:
            if self._matches_pattern(symbol_clean, pattern):
                return pattern.market_type
        
        # 默认返回股票类型
        return MarketType.STOCK
    
    def _matches_pattern(self, symbol: str, pattern: MarketPattern) -> bool:
        """检查symbol是否匹配指定模式"""
        
        # 1. 检查精确匹配
        if symbol in pattern.exact_matches:
            return True
        
        # 2. 检查长度规则
        if pattern.length_rules and len(symbol) not in pattern.length_rules:
            return False
        
        # 3. 检查正则表达式模式
        for regex_pattern in pattern.patterns:
            if re.match(regex_pattern, symbol):
                return True
        
        return False
    
    def _load_default_patterns(self) -> List[MarketPattern]:
        """加载默认的市场模式"""
        return [
            # 外汇模式（优先级最高）
            MarketPattern(
                name="forex_6char",
                market_type=MarketType.FOREX,
                patterns=[
                    r'^[A-Z]{6}$',              # EURUSD格式
                    r'^[A-Z]{3}[A-Z]{3}$',      # EURUSD格式（明确分组）
                ],
                exact_matches=[
                    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
                    'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD', 'EURNZD',
                    'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPCAD', 'GBPNZD',
                    'AUDJPY', 'AUDCHF', 'AUDCAD', 'AUDNZD',
                    'NZDJPY', 'NZDCHF', 'NZDCAD', 'CHFJPY', 'CADCHF', 'CADJPY'
                ],
                length_rules=[6],
                priority=90
            ),
            
            MarketPattern(
                name="forex_with_separators",
                market_type=MarketType.FOREX,
                patterns=[
                    r'^[A-Z]{3}[/_\-][A-Z]{3}$',    # EUR/USD, EUR_USD, EUR-USD格式
                ],
                exact_matches=[],
                length_rules=[7],
                priority=85
            ),
            
            # 加密货币模式
            MarketPattern(
                name="crypto_with_usd",
                market_type=MarketType.CRYPTO,
                patterns=[
                    r'^[A-Z]+-USD$',            # BTC-USD格式
                    r'^[A-Z]+USD$',             # BTCUSD格式
                ],
                exact_matches=[
                    'BTC-USD', 'ETH-USD', 'LTC-USD', 'BCH-USD', 'ADA-USD',
                    'DOT-USD', 'LINK-USD', 'XRP-USD', 'DOGE-USD'
                ],
                length_rules=[],
                priority=80
            ),
            
            MarketPattern(
                name="crypto_with_btc",
                market_type=MarketType.CRYPTO,
                patterns=[
                    r'^[A-Z]+-BTC$',            # ETH-BTC格式
                    r'^[A-Z]+BTC$',             # ETHBTC格式
                ],
                exact_matches=[],
                length_rules=[],
                priority=75
            ),
            
            MarketPattern(
                name="crypto_common",
                market_type=MarketType.CRYPTO,
                patterns=[
                    r'^BTC[A-Z]*$',             # BTC开头
                    r'^ETH[A-Z]*$',             # ETH开头
                    r'^[A-Z]*COIN$',            # COIN结尾
                ],
                exact_matches=[
                    'BITCOIN', 'ETHEREUM', 'LITECOIN', 'RIPPLE'
                ],
                length_rules=[],
                priority=70
            ),
            
            # 商品模式
            MarketPattern(
                name="commodities",
                market_type=MarketType.COMMODITIES,
                patterns=[
                    r'^XAU[A-Z]{3}$',           # XAUUSD (黄金)
                    r'^XAG[A-Z]{3}$',           # XAGUSD (白银)
                    r'^[A-Z]+OIL$',             # CRUDE OIL, BRENT OIL
                    r'^WTI[A-Z]*$',             # WTI原油
                    r'^BCO[A-Z]*$',             # 布伦特原油
                ],
                exact_matches=[
                    'XAUUSD', 'XAGUSD', 'XAUEUR', 'XAUGBP', 'XAUCHF', 'XAUAUD',
                    'WTIUSD', 'BCOUSD', 'GOLD', 'SILVER', 'OIL'
                ],
                length_rules=[],
                priority=85
            ),
            
            # 指数模式
            MarketPattern(
                name="indices",
                market_type=MarketType.INDEX,
                patterns=[
                    r'^\^[A-Z]+$',              # ^GSPC, ^DJI, ^IXIC格式
                    r'^[A-Z]{3}[A-Z]{3}$',      # SPXUSD格式（如果不是外汇）
                ],
                exact_matches=[
                    '^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX',
                    'SPXUSD', 'JPXJPY', 'NSXUSD', 'FRXEUR', 'UKXGBP',
                    'GRXEUR', 'AUXAUD', 'HKXHKD', 'ETXEUR', 'UDXUSD'
                ],
                length_rules=[],
                priority=75
            ),
            
            # ETF模式
            MarketPattern(
                name="etf",
                market_type=MarketType.ETF,
                patterns=[
                    r'^[A-Z]{3,4}$',            # 3-4字符ETF
                ],
                exact_matches=[
                    'SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VTI', 'VEA', 'VWO',
                    'GLD', 'SLV', 'TLT', 'IEF', 'LQD', 'HYG'
                ],
                length_rules=[3, 4],
                priority=60
            ),
            
            # 股票模式（优先级最低，作为默认）
            MarketPattern(
                name="stocks",
                market_type=MarketType.STOCK,
                patterns=[
                    r'^[A-Z]{1,5}$',            # 1-5字符股票代码
                    r'^[A-Z]+\.[A-Z]{1,3}$',    # 带后缀的股票（如 BRK.A）
                ],
                exact_matches=[
                    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA',
                    'JPM', 'JNJ', 'V', 'PG', 'UNH', 'DIS', 'MA', 'HD', 'PFE',
                    'BAC', 'KO', 'ABBV', 'PEP', 'AVGO', 'TMO', 'COST', 'WMT'
                ],
                length_rules=[1, 2, 3, 4, 5],
                priority=10  # 最低优先级
            )
        ]
    
    def _load_custom_patterns(self, custom_config: Dict):
        """加载自定义模式配置"""
        for pattern_config in custom_config:
            try:
                pattern = MarketPattern(
                    name=pattern_config['name'],
                    market_type=MarketType(pattern_config['market_type']),
                    patterns=pattern_config.get('patterns', []),
                    exact_matches=pattern_config.get('exact_matches', []),
                    length_rules=pattern_config.get('length_rules', []),
                    priority=pattern_config.get('priority', 50)
                )
                self.patterns.append(pattern)
            except (KeyError, ValueError) as e:
                print(f"Warning: Invalid custom pattern config: {e}")
    
    def add_custom_pattern(self, pattern: MarketPattern):
        """添加自定义模式"""
        self.patterns.append(pattern)
        # 重新排序
        self.patterns.sort(key=lambda x: x.priority, reverse=True)
    
    def get_confidence(self, symbol: str) -> Dict[MarketType, float]:
        """
        获取各市场类型的置信度
        
        Args:
            symbol: 交易代码
            
        Returns:
            各市场类型的置信度字典
        """
        symbol_clean = symbol.upper().strip()
        confidence = {}
        
        for pattern in self.patterns:
            if self._matches_pattern(symbol_clean, pattern):
                market_type = pattern.market_type
                # 基于优先级计算置信度
                conf_score = pattern.priority / 100.0
                
                if market_type not in confidence:
                    confidence[market_type] = conf_score
                else:
                    # 取最高置信度
                    confidence[market_type] = max(confidence[market_type], conf_score)
        
        return confidence
    
    def batch_detect(self, symbols: List[str]) -> Dict[str, MarketType]:
        """
        批量检测多个symbol的市场类型
        
        Args:
            symbols: 交易代码列表
            
        Returns:
            symbol到市场类型的映射字典
        """
        return {symbol: self.detect(symbol) for symbol in symbols}


def create_default_detector() -> MarketTypeDetector:
    """创建默认的市场类型检测器"""
    return MarketTypeDetector()


# 模块级别的默认检测器实例
_default_detector = None

def get_default_detector() -> MarketTypeDetector:
    """获取默认的检测器实例（单例）"""
    global _default_detector
    if _default_detector is None:
        _default_detector = create_default_detector()
    return _default_detector


def detect_market_type(symbol: str) -> MarketType:
    """便捷函数：检测单个symbol的市场类型"""
    return get_default_detector().detect(symbol)


def detect_market_types(symbols: List[str]) -> Dict[str, MarketType]:
    """便捷函数：批量检测市场类型"""
    return get_default_detector().batch_detect(symbols)