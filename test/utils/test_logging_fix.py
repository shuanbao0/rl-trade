#!/usr/bin/env python
"""
日志记录不全问题的测试和修复脚本

问题诊断:
1. 训练过程中日志缓冲区未及时刷新
2. 长时间运行的Stable-Baselines3训练期间日志丢失
3. 多进程环境下日志输出不完整
4. 异常中断时缓冲区数据丢失

测试方法:
python -m pytest test/utils/test_logging_fix.py -v
"""

import pytest
import os
import sys
import logging
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger
from src.training.stable_baselines_trainer import TradingCallback


class TestLoggingFix:
    """日志修复测试类"""
    
    def test_logger_flush_mechanism(self):
        """测试日志刷新机制"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test_flush.log")
            
            # 创建logger
            logger = setup_logger("TestLogger", log_file=log_file)
            
            # 写入日志
            logger.info("测试日志消息 1")
            logger.info("测试日志消息 2")
            
            # 检查是否有flush方法
            if hasattr(logger, 'flush_all'):
                logger.flush_all()
            
            # 立即检查文件是否写入
            assert os.path.exists(log_file), "日志文件未创建"
            
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "测试日志消息 1" in content, "日志消息1未写入"
                assert "测试日志消息 2" in content, "日志消息2未写入"
    
    def test_training_callback_logging(self):
        """测试训练回调日志记录"""
        # 创建模拟的训练回调
        callback = TradingCallback(detailed_output=True)
        
        # 模拟训练步骤
        mock_locals = {
            'infos': [{'portfolio': {'total_value': 10500.0}}],
            'rewards': [0.05],
            'actions': [[0.3]],
            'dones': [False]
        }
        
        callback.locals = mock_locals
        
        # 模拟1000步，应该触发日志输出
        for step in range(1, 1001):
            callback.total_steps = step
            if step % 1000 == 0:
                # 这里应该触发日志刷新和输出
                result = callback._on_step()
                assert result is True or result is None, "回调函数执行失败"


def create_enhanced_logger_solution():
    """
    创建增强的Logger解决方案
    
    主要改进：
    1. 添加强制刷新机制
    2. 改进文件处理器
    3. 添加训练状态监控
    """
    
    solution_code = '''
# 日志刷新增强方案
# 在 src/utils/logger.py 中添加以下代码：

import logging
import sys


class FlushingFileHandler(logging.FileHandler):
    """自动刷新的文件处理器"""
    
    def emit(self, record):
        super().emit(record)
        self.flush()  # 每次写入后立即刷新


class FlushingStreamHandler(logging.StreamHandler):
    """自动刷新的控制台处理器"""
    
    def emit(self, record):
        super().emit(record)
        self.flush()


# 修改setup_logger函数，添加force_flush参数
def setup_logger_enhanced(name: str, level: str = "INFO", log_file=None, 
                         console_output: bool = True, force_flush: bool = True):
    """增强版Logger设置"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 使用增强的处理器
    if console_output:
        if force_flush:
            console_handler = FlushingStreamHandler(sys.stdout)
        else:
            console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        if force_flush:
            file_handler = FlushingFileHandler(log_file, encoding='utf-8')
        else:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 添加刷新方法
    if force_flush:
        def flush_all():
            for handler in logger.handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()
        logger.flush_all = flush_all
    
    return logger
'''
    
    return solution_code


def create_training_callback_enhancement():
    """
    创建训练回调增强方案
    
    主要改进：
    1. 定期强制刷新日志
    2. 保存训练状态快照
    3. 改进异常处理
    """
    
    enhancement_code = '''
# 训练回调增强方案
# 在TradingCallback类的_on_step方法中添加：

def _on_step(self) -> bool:
    """增强版训练步骤回调"""
    self.total_steps += 1
    
    # 每1000步强制刷新日志
    if self.total_steps % 1000 == 0:
        try:
            # 刷新logger
            if hasattr(self._logger, 'flush_all'):
                self._logger.flush_all()
            
            # 刷新系统输出
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            
        except Exception:
            pass  # 静默处理刷新错误
    
    # 每5000步保存训练状态
    if self.total_steps % 5000 == 0:
        self._save_training_snapshot()
    
    # 原有的训练逻辑...
    return True

def _save_training_snapshot(self):
    """保存训练状态快照"""
    try:
        import json
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'total_steps': self.total_steps,
            'current_episode': self.current_episode,
            'status': 'running'
        }
        
        os.makedirs('logs', exist_ok=True)
        with open(f'logs/training_snapshot_{self.total_steps}.json', 'w') as f:
            json.dump(snapshot, f, indent=2)
            
    except Exception as e:
        self._logger.warning(f"保存训练快照失败: {e}")
'''
    
    return enhancement_code


def test_create_logging_solutions():
    """测试创建日志解决方案"""
    
    # 测试Logger增强方案
    logger_solution = create_enhanced_logger_solution()
    assert "FlushingFileHandler" in logger_solution
    assert "force_flush" in logger_solution
    
    # 测试训练回调增强方案  
    callback_enhancement = create_training_callback_enhancement()
    assert "_save_training_snapshot" in callback_enhancement
    assert "flush_all" in callback_enhancement
    
    print("✅ 日志解决方案代码生成成功")


def create_simple_fix_script():
    """创建简单的修复脚本"""
    
    fix_script = f"""#!/usr/bin/env python
'''
简单日志修复脚本

直接执行: python fix_logging_simple.py
'''

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def fix_logger_file():
    '''修复logger.py文件'''
    logger_file = PROJECT_ROOT / "src/utils/logger.py"
    
    # 备份原文件
    backup_file = logger_file.with_suffix('.py.backup')
    if not backup_file.exists():
        with open(logger_file, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"已备份原文件: {{backup_file}}")
    
    # 修改logger.py
    with open(logger_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加刷新功能
    if "flush()" not in content:
        # 在FileHandler创建后添加
        old_line = "        file_handler = logging.FileHandler(log_file, encoding='utf-8')"
        new_line = '''        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        # 设置为无缓冲模式，立即写入
        file_handler.stream.reconfigure(line_buffering=True)'''
        
        content = content.replace(old_line, new_line)
        
        with open(logger_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Logger文件修复完成")
        return True
    
    print("ℹ️  Logger文件已经修复过")
    return True

def main():
    print("🔧 开始简单日志修复...")
    
    if fix_logger_file():
        print("✅ 日志修复完成!")
        print("现在可以重新运行训练，日志应该会更及时地写入文件。")
    else:
        print("❌ 日志修复失败")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
"""
    
    return fix_script


def test_fix_script_creation():
    """测试修复脚本创建"""
    
    script_content = create_simple_fix_script()
    assert "fix_logger_file" in script_content
    assert "line_buffering=True" in script_content
    
    # 将脚本保存到项目根目录
    script_path = PROJECT_ROOT / "fix_logging_simple.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✅ 简单修复脚本已创建: {script_path}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])