"""
向后兼容性模块 - Backward Compatibility Module
"""

from .legacy_adapter import LegacyRewardAdapter
from .context_converter import ContextConverter
from .api_bridge import APIBridge

__all__ = [
    'LegacyRewardAdapter',
    'ContextConverter', 
    'APIBridge'
]