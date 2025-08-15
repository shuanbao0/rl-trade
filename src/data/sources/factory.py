"""
数据源工厂和注册机制
"""

from typing import Type, Dict, Optional, Any, List, Union, Tuple
import importlib
import inspect
import yaml
import json
import os
from pathlib import Path

from .base import AbstractDataSource, DataSource


class DataSourceRegistry:
    """数据源注册表"""
    
    _sources: Dict[DataSource, Type[AbstractDataSource]] = {}
    _source_configs: Dict[DataSource, Dict[str, Any]] = {}
    
    @classmethod
    def register(
        cls, 
        source: Union[str, DataSource], 
        source_class: Type[AbstractDataSource],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        注册数据源
        
        Args:
            source: 数据源枚举或字符串名称
            source_class: 数据源类
            config: 默认配置
        """
        # 允许真正的类或测试中的Mock对象
        if not (inspect.isclass(source_class) or hasattr(source_class, '__call__')):
            raise TypeError(f"source_class must be a class or callable, got {type(source_class)}")
        
        # 转换为DataSource枚举
        if isinstance(source, str):
            try:
                source = DataSource.from_string(source)
            except ValueError as e:
                print(f"[WARNING] Failed to register data source: {e}")
                return
        
        cls._sources[source] = source_class
        cls._source_configs[source] = config or {}
        
        print(f"[OK] Registered data source: {source.value}")
    
    @classmethod
    def unregister(cls, source: Union[str, DataSource]):
        """注销数据源"""
        if isinstance(source, str):
            source = DataSource.from_string(source)
        
        if source in cls._sources:
            del cls._sources[source]
            del cls._source_configs[source]
            print(f"[OK] Unregistered data source: {source.value}")
    
    @classmethod
    def get(cls, source: Union[str, DataSource]) -> Optional[Type[AbstractDataSource]]:
        """
        获取数据源类
        
        Args:
            source: 数据源枚举或字符串名称
            
        Returns:
            数据源类或None
        """
        if isinstance(source, str):
            try:
                source = DataSource.from_string(source)
            except ValueError:
                return None
        return cls._sources.get(source)
    
    @classmethod
    def get_config(cls, source: Union[str, DataSource]) -> Dict[str, Any]:
        """
        获取数据源默认配置
        
        Args:
            source: 数据源枚举或字符串名称
            
        Returns:
            默认配置字典
        """
        if isinstance(source, str):
            try:
                source = DataSource.from_string(source)
            except ValueError:
                return {}
        return cls._source_configs.get(source, {}).copy()
    
    @classmethod
    def list_sources(cls) -> List[DataSource]:
        """
        列出所有注册的数据源
        
        Returns:
            数据源枚举列表
        """
        return list(cls._sources.keys())
    
    @classmethod
    def is_registered(cls, source: Union[str, DataSource]) -> bool:
        """
        检查数据源是否已注册
        
        Args:
            source: 数据源枚举或字符串名称
            
        Returns:
            是否已注册
        """
        if isinstance(source, str):
            try:
                source = DataSource.from_string(source)
            except ValueError:
                return False
        return source in cls._sources
    
    @classmethod
    def get_source_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        获取数据源信息
        
        Args:
            name: 数据源名称
            
        Returns:
            数据源信息字典
        """
        source_class = cls.get(name)
        if source_class is None:
            return None
            
        return {
            'name': name,
            'class': source_class.__name__,
            'module': source_class.__module__,
            'doc': source_class.__doc__,
            'config': cls.get_config(name)
        }
    
    @classmethod
    def clear(cls):
        """清空所有注册"""
        cls._sources.clear()
        cls._source_configs.clear()
    
    @classmethod
    def auto_discover(cls, package_path: str = "src.data.sources"):
        """
        自动发现和注册数据源
        
        Args:
            package_path: 要扫描的包路径
        """
        try:
            package = importlib.import_module(package_path)
            package_dir = Path(package.__file__).parent
            
            for py_file in package_dir.glob("*_source.py"):
                module_name = py_file.stem
                try:
                    module = importlib.import_module(f"{package_path}.{module_name}")
                    
                    # 查找AbstractDataSource的子类
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (obj != AbstractDataSource and 
                            issubclass(obj, AbstractDataSource) and 
                            obj.__module__ == module.__name__):
                            
                            source_name = name.lower().replace('datasource', '').replace('source', '')
                            if not cls.is_registered(source_name):
                                cls.register(source_name, obj)
                                
                except ImportError as e:
                    print(f"Warning: Could not import {module_name}: {e}")
                    
        except ImportError as e:
            print(f"Warning: Could not auto-discover sources: {e}")


class ConfigLoader:
    """配置加载器"""
    
    @staticmethod
    def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        从YAML文件加载配置
        
        Args:
            file_path: YAML文件路径
            
        Returns:
            配置字典
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 处理环境变量替换
        return ConfigLoader._resolve_env_vars(config)
    
    @staticmethod  
    def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        从JSON文件加载配置
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            配置字典
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        return ConfigLoader._resolve_env_vars(config)
    
    @staticmethod
    def _resolve_env_vars(config: Any) -> Any:
        """
        递归解析环境变量
        
        Args:
            config: 配置对象
            
        Returns:
            解析后的配置对象
        """
        if isinstance(config, dict):
            return {k: ConfigLoader._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [ConfigLoader._resolve_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            # 环境变量格式: ${VAR_NAME} 或 ${VAR_NAME:default_value}
            env_var = config[2:-1]
            if ':' in env_var:
                var_name, default_value = env_var.split(':', 1)
                return os.getenv(var_name, default_value)
            else:
                return os.getenv(env_var, config)
        else:
            return config


class DataSourceFactory:
    """数据源工厂"""
    
    @staticmethod
    def create_data_source(
        source: Union[str, DataSource],
        config: Optional[Dict[str, Any]] = None
    ) -> AbstractDataSource:
        """
        创建数据源实例
        
        Args:
            source: 数据源枚举或字符串类型
            config: 配置参数
            
        Returns:
            数据源实例
            
        Raises:
            ValueError: 未知的数据源类型
            TypeError: 配置参数类型错误
        """
        # 转换为DataSource枚举
        if isinstance(source, str):
            try:
                source = DataSource.from_string(source)
            except ValueError as e:
                available = [s.value for s in DataSourceRegistry.list_sources()]
                raise ValueError(
                    f"Unknown data source: '{source}'. "
                    f"Available sources: {', '.join(available)}"
                ) from e
        
        source_class = DataSourceRegistry.get(source)
        if not source_class:
            available = [s.value for s in DataSourceRegistry.list_sources()]
            raise ValueError(
                f"Data source not registered: '{source.value}'. "
                f"Available sources: {', '.join(available)}"
            )
        
        # 合并默认配置和用户配置
        default_config = DataSourceRegistry.get_config(source)
        final_config = default_config.copy()
        
        if config is not None:
            if not isinstance(config, dict):
                raise TypeError(f"config must be dict, got {type(config)}")
            final_config.update(config)
        
        # 添加数据源名称到配置中
        final_config['name'] = source.value
        final_config['source_enum'] = source
        
        try:
            return source_class(final_config)
        except Exception as e:
            raise RuntimeError(f"Failed to create {source.value} data source: {e}") from e
    
    @staticmethod
    def create_from_config_file(
        config_path: Union[str, Path],
        source_type: Optional[str] = None
    ) -> AbstractDataSource:
        """
        从配置文件创建数据源
        
        Args:
            config_path: 配置文件路径
            source_type: 数据源类型（如果配置文件中未指定）
            
        Returns:
            数据源实例
        """
        config_path = Path(config_path)
        
        # 根据文件扩展名选择加载器
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = ConfigLoader.load_yaml(config_path)
        elif config_path.suffix.lower() == '.json':
            config = ConfigLoader.load_json(config_path)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # 确定数据源类型
        if source_type is None:
            source_type = config.get('type') or config.get('source_type')
            if source_type is None:
                raise ValueError(
                    "source_type not specified and not found in config file. "
                    "Please specify 'type' or 'source_type' in config or pass source_type parameter"
                )
        
        return DataSourceFactory.create_data_source(source_type, config)
    
    @staticmethod
    def create_from_env(
        source_type: str,
        env_prefix: str = "DATA_SOURCE"
    ) -> AbstractDataSource:
        """
        从环境变量创建数据源
        
        Args:
            source_type: 数据源类型
            env_prefix: 环境变量前缀
            
        Returns:
            数据源实例
        """
        config = {}
        prefix = f"{env_prefix}_{source_type.upper()}_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                
                # 尝试转换数据类型
                if value.lower() in ('true', 'false'):
                    config[config_key] = value.lower() == 'true'
                elif value.isdigit():
                    config[config_key] = int(value)
                elif value.replace('.', '', 1).isdigit():
                    config[config_key] = float(value)
                else:
                    config[config_key] = value
        
        return DataSourceFactory.create_data_source(source_type, config)
    
    @staticmethod
    def create_multiple(
        sources_config: Dict[str, Dict[str, Any]]
    ) -> Dict[str, AbstractDataSource]:
        """
        批量创建多个数据源
        
        Args:
            sources_config: 数据源配置字典，格式为 {source_name: config}
            
        Returns:
            数据源实例字典
        """
        sources = {}
        errors = {}
        
        for source_name, config in sources_config.items():
            try:
                sources[source_name] = DataSourceFactory.create_data_source(
                    source_name, config
                )
            except Exception as e:
                errors[source_name] = str(e)
        
        if errors:
            error_msg = "Failed to create some data sources:\n"
            for name, error in errors.items():
                error_msg += f"  {name}: {error}\n"
            print(f"Warning: {error_msg}")
        
        return sources
    
    @staticmethod
    def get_default_source() -> Optional[AbstractDataSource]:
        """
        获取默认数据源
        
        Returns:
            默认数据源实例或None
        """
        # 尝试按优先级创建默认数据源
        preferred_sources = ['yfinance', 'truefx', 'oanda']
        
        for source_type in preferred_sources:
            if DataSourceRegistry.is_registered(source_type):
                try:
                    return DataSourceFactory.create_data_source(source_type)
                except Exception:
                    continue
        
        # 如果没有找到首选源，尝试第一个可用的
        available = DataSourceRegistry.list_sources()
        if available:
            try:
                return DataSourceFactory.create_data_source(available[0])
            except Exception:
                pass
        
        return None
    
    @staticmethod
    def validate_config(
        source_type: str,
        config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        验证配置是否有效
        
        Args:
            source_type: 数据源类型
            config: 配置字典
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查数据源是否已注册
        if not DataSourceRegistry.is_registered(source_type):
            errors.append(f"Data source '{source_type}' is not registered")
            return False, errors
        
        # 尝试创建实例来验证配置
        try:
            source = DataSourceFactory.create_data_source(source_type, config)
            # 清理测试实例
            try:
                source.disconnect()
            except:
                pass
        except Exception as e:
            errors.append(f"Config validation failed: {e}")
        
        return len(errors) == 0, errors


# 自动注册已知的数据源
def auto_register_sources():
    """自动注册内置数据源"""
    try:
        # 尝试注册YFinance
        try:
            from .yfinance_source import YFinanceDataSource
            DataSourceRegistry.register('yfinance', YFinanceDataSource)
        except ImportError:
            pass
        
        # 尝试注册其他数据源
        source_modules = [
            ('truefx', 'truefx_source', 'TrueFXDataSource'),
            ('oanda', 'oanda_source', 'OandaDataSource'),
            ('dukascopy', 'dukascopy_source', 'DukascopyDataSource'),
            ('histdata', 'histdata_source', 'HistDataDataSource'),
            ('fxminute', 'fxminute_source', 'FXMinuteDataSource'),
            ('fxcm', 'fxcm_source', 'FXCMDataSource'),
        ]
        
        for source_name, module_name, class_name in source_modules:
            try:
                module = importlib.import_module(f".{module_name}", __package__)
                source_class = getattr(module, class_name)
                DataSourceRegistry.register(source_name, source_class)
            except (ImportError, AttributeError):
                pass
                
    except Exception as e:
        print(f"Warning: Auto-registration failed: {e}")


# 在模块加载时自动注册
auto_register_sources()