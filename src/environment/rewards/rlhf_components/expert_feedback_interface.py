"""
专家交易员反馈数据收集接口

基于2024-2025年RLHF最新研究，实现智能化的专家反馈收集系统，
支持偏好对比、标量评分、批评文本等多种反馈类型。
"""

import time
import uuid
import json
import sqlite3
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum
from pathlib import Path
import logging
from collections import defaultdict, deque
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """反馈类型枚举"""
    PREFERENCE_PAIR = "preference_pair"
    SCALAR_RATING = "scalar_rating"
    CRITIQUE_TEXT = "critique_text"
    RISK_TOLERANCE = "risk_tolerance"
    MARKET_REGIME = "market_regime"
    BINARY_CHOICE = "binary_choice"

class ExpertiseLevel(Enum):
    """专家级别枚举"""
    JUNIOR = "junior"           # 1-3年经验
    SENIOR = "senior"           # 3-10年经验
    EXPERT = "expert"           # 10+年经验
    SPECIALIST = "specialist"   # 特定领域专家

class MarketRegime(Enum):
    """市场状态枚举"""
    BULL = "bull"              # 牛市
    BEAR = "bear"              # 熊市
    SIDEWAYS = "sideways"      # 横盘
    VOLATILE = "volatile"      # 高波动
    CRASH = "crash"            # 崩盘
    RECOVERY = "recovery"      # 复苏

@dataclass
class TradingScenario:
    """交易场景数据结构"""
    scenario_id: str
    market_state: Dict[str, float]      # 市场状态特征
    portfolio_state: Dict[str, float]   # 投资组合状态
    action_options: List[Dict]          # 可选择的行动
    context_info: Dict[str, str]        # 上下文信息
    market_regime: MarketRegime         # 市场状态
    timestamp: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradingScenario':
        """从字典创建实例"""
        if isinstance(data['market_regime'], str):
            data['market_regime'] = MarketRegime(data['market_regime'])
        return cls(**data)

@dataclass
class ExpertProfile:
    """专家档案"""
    expert_id: str
    name: str
    expertise_level: ExpertiseLevel
    specialization: List[str]           # 专业领域
    track_record: Dict[str, float]      # 历史表现
    reliability_score: float           # 可靠性评分 (0-1)
    bias_profile: Dict[str, float]      # 偏差特征
    preferred_timeframe: str            # 偏好时间框架
    contact_info: Dict[str, str]        # 联系信息
    active_since: float                 # 激活时间
    total_feedbacks: int               # 总反馈次数
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExpertProfile':
        """从字典创建实例"""
        if isinstance(data['expertise_level'], str):
            data['expertise_level'] = ExpertiseLevel(data['expertise_level'])
        return cls(**data)

@dataclass
class FeedbackData:
    """反馈数据结构"""
    feedback_id: str
    expert_id: str
    scenario_id: str
    feedback_type: FeedbackType
    content: Union[Dict, float, str]
    confidence_level: float             # 置信度 0-1
    reasoning: Optional[str]            # 推理解释
    timestamp: float
    processing_time: float              # 处理时间
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FeedbackData':
        """从字典创建实例"""
        if isinstance(data['feedback_type'], str):
            data['feedback_type'] = FeedbackType(data['feedback_type'])
        return cls(**data)

class FeedbackDatabase:
    """反馈数据库管理器"""
    
    def __init__(self, db_path: str = "expert_feedback.db"):
        self.db_path = db_path
        self.connection = None
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            # 专家档案表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS expert_profiles (
                    expert_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    expertise_level TEXT NOT NULL,
                    specialization TEXT,
                    track_record TEXT,
                    reliability_score REAL,
                    bias_profile TEXT,
                    preferred_timeframe TEXT,
                    contact_info TEXT,
                    active_since REAL,
                    total_feedbacks INTEGER DEFAULT 0,
                    created_at REAL DEFAULT (datetime('now')),
                    updated_at REAL DEFAULT (datetime('now'))
                )
            """)
            
            # 交易场景表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trading_scenarios (
                    scenario_id TEXT PRIMARY KEY,
                    market_state TEXT,
                    portfolio_state TEXT,
                    action_options TEXT,
                    context_info TEXT,
                    market_regime TEXT,
                    timestamp REAL,
                    metadata TEXT,
                    created_at REAL DEFAULT (datetime('now'))
                )
            """)
            
            # 反馈数据表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_data (
                    feedback_id TEXT PRIMARY KEY,
                    expert_id TEXT,
                    scenario_id TEXT,
                    feedback_type TEXT,
                    content TEXT,
                    confidence_level REAL,
                    reasoning TEXT,
                    timestamp REAL,
                    processing_time REAL,
                    metadata TEXT,
                    created_at REAL DEFAULT (datetime('now')),
                    FOREIGN KEY (expert_id) REFERENCES expert_profiles (expert_id),
                    FOREIGN KEY (scenario_id) REFERENCES trading_scenarios (scenario_id)
                )
            """)
            
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_expert ON feedback_data (expert_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_scenario ON feedback_data (scenario_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback_data (feedback_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback_data (timestamp)")
            
            conn.commit()
    
    def store_expert_profile(self, expert: ExpertProfile):
        """存储专家档案"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO expert_profiles 
                    (expert_id, name, expertise_level, specialization, track_record, 
                     reliability_score, bias_profile, preferred_timeframe, contact_info, 
                     active_since, total_feedbacks)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    expert.expert_id,
                    expert.name,
                    expert.expertise_level.value,
                    json.dumps(expert.specialization),
                    json.dumps(expert.track_record),
                    expert.reliability_score,
                    json.dumps(expert.bias_profile),
                    expert.preferred_timeframe,
                    json.dumps(expert.contact_info),
                    expert.active_since,
                    expert.total_feedbacks
                ))
                conn.commit()
    
    def store_trading_scenario(self, scenario: TradingScenario):
        """存储交易场景"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO trading_scenarios 
                    (scenario_id, market_state, portfolio_state, action_options, 
                     context_info, market_regime, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    scenario.scenario_id,
                    json.dumps(scenario.market_state),
                    json.dumps(scenario.portfolio_state),
                    json.dumps(scenario.action_options),
                    json.dumps(scenario.context_info),
                    scenario.market_regime.value,
                    scenario.timestamp,
                    json.dumps(scenario.metadata)
                ))
                conn.commit()
    
    def store_feedback(self, feedback: FeedbackData):
        """存储反馈数据"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO feedback_data 
                    (feedback_id, expert_id, scenario_id, feedback_type, content, 
                     confidence_level, reasoning, timestamp, processing_time, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.feedback_id,
                    feedback.expert_id,
                    feedback.scenario_id,
                    feedback.feedback_type.value,
                    json.dumps(feedback.content),
                    feedback.confidence_level,
                    feedback.reasoning,
                    feedback.timestamp,
                    feedback.processing_time,
                    json.dumps(feedback.metadata)
                ))
                
                # 更新专家反馈计数
                conn.execute("""
                    UPDATE expert_profiles 
                    SET total_feedbacks = total_feedbacks + 1,
                        updated_at = datetime('now')
                    WHERE expert_id = ?
                """, (feedback.expert_id,))
                
                conn.commit()
    
    def get_expert_feedback_history(self, expert_id: str, limit: int = 100) -> List[FeedbackData]:
        """获取专家反馈历史"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM feedback_data 
                WHERE expert_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (expert_id, limit))
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            feedbacks = []
            for row in rows:
                data = dict(zip(columns, row))
                data['content'] = json.loads(data['content'])
                data['metadata'] = json.loads(data['metadata'])
                
                # 移除数据库专用字段
                data.pop('created_at', None)
                
                feedbacks.append(FeedbackData.from_dict(data))
            
            return feedbacks

class TradingScenarioGenerator:
    """交易场景生成器"""
    
    def __init__(self):
        self.scenario_templates = self._load_scenario_templates()
        self.market_data_simulator = MarketDataSimulator()
    
    def generate_scenarios(self, 
                         expert: ExpertProfile,
                         num_scenarios: int = 5,
                         focus_area: Optional[str] = None) -> List[TradingScenario]:
        """生成个性化交易场景"""
        
        scenarios = []
        
        for i in range(num_scenarios):
            # 基于专家专长选择场景类型
            scenario_type = self._select_scenario_type(expert, focus_area)
            
            # 生成市场状态
            market_state = self.market_data_simulator.generate_market_state(scenario_type)
            
            # 生成投资组合状态
            portfolio_state = self._generate_portfolio_state(expert, market_state)
            
            # 生成行动选项
            action_options = self._generate_action_options(scenario_type, market_state, portfolio_state)
            
            # 确定市场状态
            market_regime = self._determine_market_regime(market_state)
            
            scenario = TradingScenario(
                scenario_id=f"scenario_{uuid.uuid4().hex[:8]}",
                market_state=market_state,
                portfolio_state=portfolio_state,
                action_options=action_options,
                context_info=self._generate_context_info(scenario_type, expert),
                market_regime=market_regime,
                timestamp=time.time(),
                metadata={
                    'expert_id': expert.expert_id,
                    'scenario_type': scenario_type,
                    'focus_area': focus_area,
                    'difficulty_level': self._calculate_difficulty_level(expert, scenario_type)
                }
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _load_scenario_templates(self) -> Dict[str, Dict]:
        """加载场景模板"""
        return {
            'portfolio_rebalancing': {
                'description': '投资组合再平衡决策',
                'complexity': 'medium',
                'required_expertise': ['portfolio_management', 'risk_management']
            },
            'risk_management': {
                'description': '风险管理决策',
                'complexity': 'high',
                'required_expertise': ['risk_management', 'derivatives']
            },
            'market_timing': {
                'description': '市场时机选择',
                'complexity': 'high',
                'required_expertise': ['technical_analysis', 'market_psychology']
            },
            'asset_allocation': {
                'description': '资产配置决策',
                'complexity': 'medium',
                'required_expertise': ['asset_allocation', 'macroeconomics']
            },
            'sector_rotation': {
                'description': '行业轮动策略',
                'complexity': 'medium',
                'required_expertise': ['sector_analysis', 'business_cycles']
            }
        }
    
    def _select_scenario_type(self, expert: ExpertProfile, focus_area: Optional[str]) -> str:
        """选择场景类型"""
        if focus_area and focus_area in self.scenario_templates:
            return focus_area
        
        # 基于专家专长随机选择
        suitable_scenarios = []
        for scenario_type, template in self.scenario_templates.items():
            if any(exp in template['required_expertise'] for exp in expert.specialization):
                suitable_scenarios.append(scenario_type)
        
        if suitable_scenarios:
            return np.random.choice(suitable_scenarios)
        else:
            return np.random.choice(list(self.scenario_templates.keys()))
    
    def _generate_portfolio_state(self, expert: ExpertProfile, market_state: Dict) -> Dict[str, float]:
        """生成投资组合状态"""
        return {
            'total_value': 1000000.0,  # 100万初始资金
            'cash_ratio': np.random.uniform(0.05, 0.20),
            'equity_ratio': np.random.uniform(0.60, 0.85),
            'bond_ratio': np.random.uniform(0.10, 0.25),
            'alternative_ratio': np.random.uniform(0.00, 0.10),
            'sector_concentration': {
                'technology': np.random.uniform(0.15, 0.35),
                'healthcare': np.random.uniform(0.10, 0.25),
                'financials': np.random.uniform(0.10, 0.20),
                'energy': np.random.uniform(0.05, 0.15),
                'consumer': np.random.uniform(0.10, 0.20)
            },
            'risk_metrics': {
                'var_95': np.random.uniform(0.02, 0.05),
                'max_drawdown': np.random.uniform(0.08, 0.15),
                'sharpe_ratio': np.random.uniform(0.8, 1.5),
                'beta': np.random.uniform(0.8, 1.2)
            }
        }
    
    def _generate_action_options(self, scenario_type: str, market_state: Dict, portfolio_state: Dict) -> List[Dict]:
        """生成行动选项"""
        actions = []
        
        if scenario_type == 'portfolio_rebalancing':
            actions = [
                {
                    'action_type': 'rebalance',
                    'description': '标准再平衡至目标配置',
                    'target_allocation': {
                        'equity': 0.70,
                        'bond': 0.25,
                        'cash': 0.05
                    },
                    'expected_cost': 0.002
                },
                {
                    'action_type': 'tactical_tilt',
                    'description': '战术性倾斜增加股票配置',
                    'target_allocation': {
                        'equity': 0.80,
                        'bond': 0.15,
                        'cash': 0.05
                    },
                    'expected_cost': 0.003
                },
                {
                    'action_type': 'maintain',
                    'description': '维持当前配置',
                    'target_allocation': portfolio_state,
                    'expected_cost': 0.000
                }
            ]
        
        elif scenario_type == 'risk_management':
            actions = [
                {
                    'action_type': 'hedge',
                    'description': '购买看跌期权对冲',
                    'hedge_ratio': 0.5,
                    'cost': 0.01,
                    'protection_level': 0.10
                },
                {
                    'action_type': 'reduce_exposure',
                    'description': '降低风险敞口',
                    'reduction_ratio': 0.3,
                    'target_beta': 0.7
                },
                {
                    'action_type': 'diversify',
                    'description': '增加资产多样化',
                    'new_positions': ['international_equity', 'commodities'],
                    'allocation_shift': 0.15
                }
            ]
        
        else:
            # 通用行动选项
            actions = [
                {
                    'action_type': 'buy',
                    'description': '增加风险资产敞口',
                    'size': np.random.uniform(0.05, 0.15)
                },
                {
                    'action_type': 'sell',
                    'description': '减少风险资产敞口', 
                    'size': np.random.uniform(0.05, 0.15)
                },
                {
                    'action_type': 'hold',
                    'description': '维持当前头寸',
                    'rationale': '等待更清晰的市场信号'
                }
            ]
        
        # 为每个行动添加唯一ID
        for i, action in enumerate(actions):
            action['action_id'] = f"action_{i+1}"
        
        return actions
    
    def _determine_market_regime(self, market_state: Dict) -> MarketRegime:
        """确定市场状态"""
        volatility = market_state.get('volatility', 0.15)
        trend = market_state.get('trend', 0.0)
        momentum = market_state.get('momentum', 0.0)
        
        if volatility > 0.25:
            if trend < -0.1:
                return MarketRegime.CRASH
            else:
                return MarketRegime.VOLATILE
        elif trend > 0.05:
            return MarketRegime.BULL
        elif trend < -0.05:
            return MarketRegime.BEAR
        elif abs(trend) < 0.02 and momentum > 0:
            return MarketRegime.RECOVERY
        else:
            return MarketRegime.SIDEWAYS
    
    def _generate_context_info(self, scenario_type: str, expert: ExpertProfile) -> Dict[str, str]:
        """生成上下文信息"""
        context_templates = {
            'portfolio_rebalancing': {
                'market_outlook': '市场进入调整期，波动性增加',
                'client_constraints': '客户风险容忍度为中等，关注长期收益',
                'regulatory_notes': '新的ESG披露要求生效',
                'time_horizon': '投资时间窗口为3-5年'
            },
            'risk_management': {
                'market_outlook': '地缘政治风险上升，市场不确定性增加',
                'risk_budget': '当前风险预算使用率75%',
                'stress_test_results': '压力测试显示下行风险较大',
                'correlation_warning': '资产间相关性异常上升'
            }
        }
        
        default_context = {
            'market_outlook': '市场情况复杂，需要谨慎评估',
            'time_pressure': '决策时间窗口有限',
            'data_quality': '市场数据质量良好',
            'external_factors': '考虑宏观经济环境影响'
        }
        
        return context_templates.get(scenario_type, default_context)
    
    def _calculate_difficulty_level(self, expert: ExpertProfile, scenario_type: str) -> str:
        """计算场景难度等级"""
        base_difficulty = self.scenario_templates[scenario_type]['complexity']
        
        # 根据专家级别调整难度
        if expert.expertise_level == ExpertiseLevel.JUNIOR:
            if base_difficulty == 'high':
                return 'expert'
            elif base_difficulty == 'medium':
                return 'high'
            else:
                return 'medium'
        elif expert.expertise_level == ExpertiseLevel.EXPERT:
            return 'expert'
        else:
            return base_difficulty

class MarketDataSimulator:
    """市场数据模拟器"""
    
    def generate_market_state(self, scenario_type: str) -> Dict[str, float]:
        """生成市场状态数据"""
        
        # 基础市场指标
        market_state = {
            'sp500_return': np.random.normal(0.001, 0.015),    # S&P 500日收益率
            'volatility': np.random.uniform(0.10, 0.30),       # 波动率
            'interest_rate': np.random.uniform(0.02, 0.06),    # 利率
            'credit_spread': np.random.uniform(0.005, 0.025),  # 信用利差
            'dollar_index': np.random.normal(0.0, 0.008),      # 美元指数变化
            'vix': np.random.uniform(15, 35),                  # VIX恐慌指数
            'yield_curve_slope': np.random.uniform(-0.5, 2.0), # 收益率曲线斜率
            'sector_rotation': np.random.uniform(-0.5, 0.5),   # 行业轮动指标
            'momentum': np.random.normal(0.0, 0.01),           # 动量指标
            'trend': np.random.normal(0.0, 0.02),              # 趋势指标
        }
        
        # 根据场景类型调整某些指标
        if scenario_type == 'risk_management':
            market_state['volatility'] *= 1.5  # 提高波动率
            market_state['vix'] *= 1.3          # 提高恐慌指数
        
        elif scenario_type == 'market_timing':
            market_state['momentum'] *= 2.0     # 增强动量信号
            market_state['trend'] *= 1.5        # 增强趋势信号
        
        return market_state

class ExpertFeedbackInterface:
    """专家交易员反馈收集接口"""
    
    def __init__(self, db_path: str = "expert_feedback.db"):
        self.database = FeedbackDatabase(db_path)
        self.scenario_generator = TradingScenarioGenerator()
        self.quality_controller = FeedbackQualityController()
        self.experts = {}  # 专家档案缓存
        self._load_existing_experts()
        
        logger.info("ExpertFeedbackInterface initialized successfully")
    
    def _load_existing_experts(self):
        """加载现有专家档案"""
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.execute("SELECT * FROM expert_profiles")
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                for row in rows:
                    data = dict(zip(columns, row))
                    # 反序列化JSON字段
                    data['specialization'] = json.loads(data['specialization'])
                    data['track_record'] = json.loads(data['track_record'])
                    data['bias_profile'] = json.loads(data['bias_profile'])
                    data['contact_info'] = json.loads(data['contact_info'])
                    
                    # 移除数据库专用字段
                    data.pop('created_at', None)
                    data.pop('updated_at', None)
                    
                    expert = ExpertProfile.from_dict(data)
                    self.experts[expert.expert_id] = expert
                    
            logger.info(f"Loaded {len(self.experts)} existing expert profiles")
        except Exception as e:
            logger.warning(f"Could not load existing experts: {e}")
    
    def register_expert(self, 
                       name: str,
                       expertise_level: ExpertiseLevel,
                       specialization: List[str],
                       contact_info: Dict[str, str] = None) -> ExpertProfile:
        """注册新专家"""
        
        expert_id = f"expert_{uuid.uuid4().hex[:8]}"
        
        expert = ExpertProfile(
            expert_id=expert_id,
            name=name,
            expertise_level=expertise_level,
            specialization=specialization,
            track_record={},
            reliability_score=0.8,  # 默认可靠性评分
            bias_profile={},
            preferred_timeframe="daily",
            contact_info=contact_info or {},
            active_since=time.time(),
            total_feedbacks=0
        )
        
        # 存储到数据库
        self.database.store_expert_profile(expert)
        
        # 缓存到内存
        self.experts[expert_id] = expert
        
        logger.info(f"Registered new expert: {name} ({expert_id})")
        return expert
    
    def collect_preference_feedback(self,
                                  expert_id: str,
                                  scenario_a: TradingScenario,
                                  scenario_b: TradingScenario,
                                  preferred_scenario: str,
                                  confidence_level: float,
                                  reasoning: str = None) -> FeedbackData:
        """收集偏好比较反馈"""
        
        if expert_id not in self.experts:
            raise ValueError(f"Expert {expert_id} not found")
        
        # 验证偏好选择
        if preferred_scenario not in [scenario_a.scenario_id, scenario_b.scenario_id]:
            raise ValueError("Preferred scenario must be one of the two scenarios")
        
        # 存储场景数据
        self.database.store_trading_scenario(scenario_a)
        self.database.store_trading_scenario(scenario_b)
        
        # 创建反馈数据
        feedback = FeedbackData(
            feedback_id=f"feedback_{uuid.uuid4().hex[:8]}",
            expert_id=expert_id,
            scenario_id=f"{scenario_a.scenario_id}_vs_{scenario_b.scenario_id}",
            feedback_type=FeedbackType.PREFERENCE_PAIR,
            content={
                'scenario_a_id': scenario_a.scenario_id,
                'scenario_b_id': scenario_b.scenario_id,
                'preferred_scenario_id': preferred_scenario,
                'comparison_aspects': self._analyze_scenario_differences(scenario_a, scenario_b)
            },
            confidence_level=confidence_level,
            reasoning=reasoning,
            timestamp=time.time(),
            processing_time=0.0,  # 将在处理完成后更新
            metadata={
                'interface_version': '1.0',
                'collection_method': 'direct_input',
                'scenario_complexity': scenario_a.metadata.get('difficulty_level', 'medium')
            }
        )
        
        # 质量控制
        start_time = time.time()
        if self.quality_controller.validate_feedback(feedback, self.experts[expert_id]):
            feedback.processing_time = time.time() - start_time
            self.database.store_feedback(feedback)
            
            # 更新专家可靠性评分
            self._update_expert_reliability(expert_id, feedback)
            
            logger.info(f"Collected preference feedback from {expert_id}")
            return feedback
        else:
            raise ValueError("Feedback quality validation failed")
    
    def collect_scalar_rating(self,
                            expert_id: str,
                            scenario: TradingScenario,
                            action_id: str,
                            ratings: Dict[str, float],
                            confidence_level: float,
                            reasoning: str = None) -> FeedbackData:
        """收集标量评分反馈"""
        
        if expert_id not in self.experts:
            raise ValueError(f"Expert {expert_id} not found")
        
        # 验证评分维度
        required_dimensions = ['overall_quality', 'risk_appropriateness', 'timing_quality', 
                             'expected_return', 'market_fit']
        
        for dim in required_dimensions:
            if dim not in ratings:
                ratings[dim] = 5.0  # 默认中性评分
            elif not 1.0 <= ratings[dim] <= 10.0:
                raise ValueError(f"Rating for {dim} must be between 1.0 and 10.0")
        
        # 查找对应的行动
        target_action = None
        for action in scenario.action_options:
            if action['action_id'] == action_id:
                target_action = action
                break
        
        if not target_action:
            raise ValueError(f"Action {action_id} not found in scenario")
        
        # 存储场景数据
        self.database.store_trading_scenario(scenario)
        
        # 创建反馈数据
        feedback = FeedbackData(
            feedback_id=f"feedback_{uuid.uuid4().hex[:8]}",
            expert_id=expert_id,
            scenario_id=scenario.scenario_id,
            feedback_type=FeedbackType.SCALAR_RATING,
            content={
                'action_id': action_id,
                'action_details': target_action,
                'ratings': ratings,
                'composite_score': np.mean(list(ratings.values()))
            },
            confidence_level=confidence_level,
            reasoning=reasoning,
            timestamp=time.time(),
            processing_time=0.0,
            metadata={
                'rating_dimensions': list(ratings.keys()),
                'scenario_type': scenario.metadata.get('scenario_type', 'unknown'),
                'market_regime': scenario.market_regime.value
            }
        )
        
        # 质量控制和存储
        start_time = time.time()
        if self.quality_controller.validate_feedback(feedback, self.experts[expert_id]):
            feedback.processing_time = time.time() - start_time
            self.database.store_feedback(feedback)
            
            # 更新专家统计
            self._update_expert_reliability(expert_id, feedback)
            
            logger.info(f"Collected scalar rating from {expert_id}")
            return feedback
        else:
            raise ValueError("Feedback quality validation failed")
    
    def generate_expert_scenarios(self,
                                expert_id: str,
                                num_scenarios: int = 5,
                                focus_area: str = None) -> List[TradingScenario]:
        """为专家生成个性化场景"""
        
        if expert_id not in self.experts:
            raise ValueError(f"Expert {expert_id} not found")
        
        expert = self.experts[expert_id]
        scenarios = self.scenario_generator.generate_scenarios(
            expert=expert,
            num_scenarios=num_scenarios,
            focus_area=focus_area
        )
        
        # 存储生成的场景
        for scenario in scenarios:
            self.database.store_trading_scenario(scenario)
        
        logger.info(f"Generated {len(scenarios)} scenarios for expert {expert_id}")
        return scenarios
    
    def get_expert_statistics(self, expert_id: str) -> Dict[str, Any]:
        """获取专家统计信息"""
        
        if expert_id not in self.experts:
            raise ValueError(f"Expert {expert_id} not found")
        
        expert = self.experts[expert_id]
        feedback_history = self.database.get_expert_feedback_history(expert_id)
        
        # 计算统计指标
        stats = {
            'basic_info': {
                'expert_id': expert_id,
                'name': expert.name,
                'expertise_level': expert.expertise_level.value,
                'specialization': expert.specialization,
                'active_since': expert.active_since,
                'reliability_score': expert.reliability_score
            },
            'feedback_stats': {
                'total_feedbacks': len(feedback_history),
                'feedback_types': {},
                'average_confidence': 0.0,
                'recent_activity': []
            },
            'performance_metrics': {
                'consistency_score': 0.0,
                'response_time': 0.0,
                'quality_score': 0.0
            }
        }
        
        if feedback_history:
            # 反馈类型分布
            for feedback in feedback_history:
                feedback_type = feedback.feedback_type.value
                stats['feedback_stats']['feedback_types'][feedback_type] = \
                    stats['feedback_stats']['feedback_types'].get(feedback_type, 0) + 1
            
            # 平均置信度
            stats['feedback_stats']['average_confidence'] = \
                np.mean([f.confidence_level for f in feedback_history])
            
            # 最近活动
            recent_feedback = feedback_history[:5]
            stats['feedback_stats']['recent_activity'] = [
                {
                    'timestamp': f.timestamp,
                    'type': f.feedback_type.value,
                    'confidence': f.confidence_level
                } for f in recent_feedback
            ]
            
            # 性能指标
            processing_times = [f.processing_time for f in feedback_history if f.processing_time > 0]
            if processing_times:
                stats['performance_metrics']['response_time'] = np.mean(processing_times)
        
        return stats
    
    def _analyze_scenario_differences(self, scenario_a: TradingScenario, scenario_b: TradingScenario) -> Dict[str, Any]:
        """分析两个场景的差异"""
        
        differences = {
            'market_regime': {
                'scenario_a': scenario_a.market_regime.value,
                'scenario_b': scenario_b.market_regime.value,
                'different': scenario_a.market_regime != scenario_b.market_regime
            },
            'action_complexity': {
                'scenario_a': len(scenario_a.action_options),
                'scenario_b': len(scenario_b.action_options),
                'different': len(scenario_a.action_options) != len(scenario_b.action_options)
            },
            'portfolio_risk': {
                'scenario_a': scenario_a.portfolio_state.get('risk_metrics', {}).get('var_95', 0),
                'scenario_b': scenario_b.portfolio_state.get('risk_metrics', {}).get('var_95', 0)
            }
        }
        
        return differences
    
    def _update_expert_reliability(self, expert_id: str, feedback: FeedbackData):
        """更新专家可靠性评分"""
        
        expert = self.experts[expert_id]
        
        # 基于反馈质量调整可靠性
        quality_indicators = {
            'confidence_level': feedback.confidence_level,
            'has_reasoning': 1.0 if feedback.reasoning else 0.0,
            'processing_time': 1.0 - min(feedback.processing_time / 300.0, 1.0)  # 5分钟内完成为满分
        }
        
        quality_score = np.mean(list(quality_indicators.values()))
        
        # 指数移动平均更新可靠性评分
        alpha = 0.1  # 学习率
        expert.reliability_score = (1 - alpha) * expert.reliability_score + alpha * quality_score
        
        # 更新专家档案
        self.database.store_expert_profile(expert)
        self.experts[expert_id] = expert

class FeedbackQualityController:
    """反馈质量控制器"""
    
    def __init__(self):
        self.consistency_threshold = 0.7
        self.confidence_threshold = 0.3
        self.time_threshold = 600.0  # 10分钟
    
    def validate_feedback(self, feedback: FeedbackData, expert: ExpertProfile) -> bool:
        """验证反馈质量"""
        
        checks = []
        
        # 置信度检查
        checks.append(feedback.confidence_level >= self.confidence_threshold)
        
        # 内容完整性检查
        checks.append(self._check_content_completeness(feedback))
        
        # 专家专长匹配检查
        checks.append(self._check_expertise_match(feedback, expert))
        
        # 逻辑一致性检查
        checks.append(self._check_logical_consistency(feedback))
        
        # 至少通过80%的检查
        return sum(checks) >= len(checks) * 0.8
    
    def _check_content_completeness(self, feedback: FeedbackData) -> bool:
        """检查内容完整性"""
        
        if not feedback.content:
            return False
        
        if feedback.feedback_type == FeedbackType.PREFERENCE_PAIR:
            required_keys = ['scenario_a_id', 'scenario_b_id', 'preferred_scenario_id']
            return all(key in feedback.content for key in required_keys)
        
        elif feedback.feedback_type == FeedbackType.SCALAR_RATING:
            return 'ratings' in feedback.content and len(feedback.content['ratings']) > 0
        
        return True
    
    def _check_expertise_match(self, feedback: FeedbackData, expert: ExpertProfile) -> bool:
        """检查专家专长匹配"""
        # 这里可以实现更复杂的匹配逻辑
        return True  # 暂时返回True
    
    def _check_logical_consistency(self, feedback: FeedbackData) -> bool:
        """检查逻辑一致性"""
        
        if feedback.feedback_type == FeedbackType.SCALAR_RATING:
            ratings = feedback.content.get('ratings', {})
            if not ratings:
                return False
            
            # 检查评分是否在合理范围内
            for dimension, rating in ratings.items():
                if not 1.0 <= rating <= 10.0:
                    return False
            
            # 检查评分的逻辑一致性（例如高风险低收益的组合应该得到较低评分）
            risk_rating = ratings.get('risk_appropriateness', 5.0)
            return_rating = ratings.get('expected_return', 5.0)
            
            # 简单的一致性检查：风险和收益应该有某种相关性
            if abs(risk_rating - return_rating) > 4.0:  # 差异过大可能存在逻辑问题
                return False
        
        return True

if __name__ == "__main__":
    # 示例使用
    interface = ExpertFeedbackInterface()
    
    # 注册专家
    expert = interface.register_expert(
        name="张三",
        expertise_level=ExpertiseLevel.SENIOR,
        specialization=["portfolio_management", "risk_management"],
        contact_info={"email": "zhangsan@example.com"}
    )
    
    # 生成场景
    scenarios = interface.generate_expert_scenarios(expert.expert_id, num_scenarios=2)
    
    # 收集偏好反馈
    feedback = interface.collect_preference_feedback(
        expert_id=expert.expert_id,
        scenario_a=scenarios[0],
        scenario_b=scenarios[1],
        preferred_scenario=scenarios[0].scenario_id,
        confidence_level=0.8,
        reasoning="场景A的风险控制更好，符合当前市场环境"
    )
    
    # 查看专家统计
    stats = interface.get_expert_statistics(expert.expert_id)
    print(f"Expert stats: {stats}")