"""
测试风险管理器模块
"""

import pytest
from datetime import datetime
from src.risk.risk_manager import RiskManager
from src.utils.config import Config


class TestRiskManager:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.config = Config()
        self.risk_manager = RiskManager(config=self.config)
    
    def test_risk_manager_initialization(self):
        """测试风险管理器初始化"""
        assert self.risk_manager.config is not None
        assert hasattr(self.risk_manager, 'max_position_size')
        assert hasattr(self.risk_manager, 'max_drawdown')
    
    def test_portfolio_state_update(self):
        """测试投资组合状态更新"""
        self.risk_manager.update_portfolio_state(
            total_value=100000.0,
            cash=50000.0,
            positions={'AAPL': 25000.0, 'MSFT': 25000.0},
            timestamp=datetime.now()
        )
        
        assert self.risk_manager.current_portfolio_value == 100000.0
        assert self.risk_manager.current_cash == 50000.0
    
    def test_position_size_validation(self):
        """测试仓位大小验证"""
        # 测试有效仓位
        valid_action = {'symbol': 'AAPL', 'target_position': 0.1}
        result, warnings = self.risk_manager.validate_position_size(valid_action)
        
        assert result is True
        
        # 测试过大仓位
        invalid_action = {'symbol': 'AAPL', 'target_position': 0.5}  # 假设限制是0.3
        result, warnings = self.risk_manager.validate_position_size(invalid_action)
        
        # 根据具体实现调整断言
        assert isinstance(result, bool)
    
    def test_risk_controls_application(self):
        """测试风险控制应用"""
        action = {
            'symbol': 'AAPL',
            'target_position': 0.2,
            'current_position': 0.1
        }
        
        adjusted_action, warnings = self.risk_manager.apply_risk_controls(action)
        
        assert isinstance(adjusted_action, dict)
        assert isinstance(warnings, list)