"""
基础搜索类

提供搜索功能的抽象基类和通用功能
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import time

from model.get_models import get_llm_model, get_embeddings_model
from config.neo4jdb import get_db_manager
from search_new.config import get_search_config
from search_new.utils.cache_manager import CacheManager, MemoryCacheBackend, DiskCacheBackend

class BaseSearch(ABC):
    """
    搜索基类
    
    提供所有搜索实现的通用功能，包括：
    - 模型管理
    - 数据库连接
    - 缓存管理
    - 性能监控
    - 错误处理
    """
    
    def __init__(self, 
                 llm=None, 
                 embeddings=None, 
                 cache_dir: Optional[str] = None,
                 enable_cache: bool = True):
        """
        初始化基础搜索类
        
        参数:
            llm: 大语言模型实例，如果为None则自动获取
            embeddings: 嵌入模型实例，如果为None则自动获取
            cache_dir: 缓存目录，如果为None则使用默认配置
            enable_cache: 是否启用缓存
        """
        # 获取配置
        self.config = get_search_config()
        
        # 初始化模型
        self.llm = llm or get_llm_model()
        self.embeddings = embeddings or get_embeddings_model()
        
        # 初始化数据库连接
        self._setup_database()
        
        # 初始化缓存
        if enable_cache:
            self._setup_cache(cache_dir)
        else:
            self.cache_manager = None
        
        # 性能监控
        self.performance_metrics = {
            "query_time": 0.0,
            "llm_time": 0.0,
            "db_time": 0.0,
            "cache_time": 0.0,
            "total_time": 0.0
        }
        
        # 错误统计
        self.error_stats = {
            "query_errors": 0,
            "llm_errors": 0,
            "db_errors": 0,
            "cache_errors": 0
        }
    
    def _setup_database(self):
        """设置数据库连接"""
        try:
            self.db_manager = get_db_manager()
            self.graph = self.db_manager.get_graph()
            self.driver = self.db_manager.get_driver()
            
        except Exception as e:
            print(f"数据库连接设置失败: {e}")
            raise
    
    def _setup_cache(self, cache_dir: Optional[str] = None):
        """设置缓存管理器"""
        try:
            # 使用配置中的缓存目录或传入的目录
            if cache_dir is None:
                cache_dir = self.config.cache.base_cache_dir
            
            # 创建缓存后端
            memory_backend = None
            disk_backend = None
            
            if self.config.cache.memory_cache_enabled:
                memory_backend = MemoryCacheBackend(
                    max_size=self.config.cache.max_cache_size,
                    default_ttl=self.config.cache.cache_ttl
                )
            
            if self.config.cache.disk_cache_enabled:
                disk_backend = DiskCacheBackend(
                    cache_dir=cache_dir,
                    default_ttl=self.config.cache.cache_ttl
                )
            
            # 创建缓存管理器
            self.cache_manager = CacheManager(
                memory_backend=memory_backend,
                disk_backend=disk_backend,
                use_memory=self.config.cache.memory_cache_enabled,
                use_disk=self.config.cache.disk_cache_enabled
            )
            
        except Exception as e:
            print(f"缓存设置失败: {e}")
            self.cache_manager = None
    
    def _get_cache_key(self, query: str, **kwargs) -> str:
        """
        生成缓存键
        
        参数:
            query: 查询字符串
            **kwargs: 其他参数
            
        返回:
            str: 缓存键
        """
        # 基础键包含类名和查询
        base_key = f"{self.__class__.__name__}:{query}"
        
        # 添加其他参数
        if kwargs:
            params_str = "|".join(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            base_key = f"{base_key}|{params_str}"
        
        return base_key
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """从缓存获取结果"""
        if not self.cache_manager:
            return None
            
        try:
            start_time = time.time()
            result = self.cache_manager.get(cache_key)
            self.performance_metrics["cache_time"] += time.time() - start_time
            
            if result is not None:
                print(f"缓存命中: {cache_key}")
            
            return result
            
        except Exception as e:
            print(f"缓存读取失败: {e}")
            self.error_stats["cache_errors"] += 1
            return None
    
    def _set_to_cache(self, cache_key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存"""
        if not self.cache_manager:
            return False
            
        try:
            start_time = time.time()
            success = self.cache_manager.set(cache_key, value, ttl)
            self.performance_metrics["cache_time"] += time.time() - start_time
            
            if success:
                print(f"缓存设置成功: {cache_key}")
            
            return success
            
        except Exception as e:
            print(f"缓存设置失败: {e}")
            self.error_stats["cache_errors"] += 1
            return False
    
    def _execute_db_query(self, cypher: str, params: Optional[Dict] = None) -> Any:
        """
        执行数据库查询
        
        参数:
            cypher: Cypher查询语句
            params: 查询参数
            
        返回:
            查询结果
        """
        try:
            start_time = time.time()
            result = self.db_manager.execute_query(cypher, params or {})
            self.performance_metrics["db_time"] += time.time() - start_time
            
            return result
            
        except Exception as e:
            print(f"数据库查询失败: {e}")
            self.error_stats["db_errors"] += 1
            raise
    
    def _call_llm(self, messages, **kwargs) -> str:
        """
        调用大语言模型
        
        参数:
            messages: 消息列表
            **kwargs: 其他参数
            
        返回:
            str: 模型响应
        """
        try:
            start_time = time.time()
            response = self.llm.invoke(messages, **kwargs)
            self.performance_metrics["llm_time"] += time.time() - start_time
            
            # 提取响应内容
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            print(f"LLM调用失败: {e}")
            self.error_stats["llm_errors"] += 1
            raise
    
    def _reset_metrics(self):
        """重置性能指标"""
        for key in self.performance_metrics:
            self.performance_metrics[key] = 0.0
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        return self.performance_metrics.copy()
    
    def get_error_stats(self) -> Dict[str, int]:
        """获取错误统计"""
        return self.error_stats.copy()
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> str:
        """
        执行搜索
        
        参数:
            query: 搜索查询
            **kwargs: 其他参数
            
        返回:
            str: 搜索结果
        """
        pass
    
    def close(self):
        """关闭资源连接"""
        try:
            # 清理缓存
            if self.cache_manager:
                self.cache_manager.clear()
            
            # 关闭数据库连接
            if hasattr(self, 'graph') and hasattr(self.graph, 'close'):
                self.graph.close()
                
        except Exception as e:
            print(f"资源关闭失败: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
