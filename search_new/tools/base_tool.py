"""
搜索工具基类

为各种搜索工具提供统一的接口和通用功能
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import time

from langchain_core.tools import BaseTool

from model.get_models import get_llm_model, get_embeddings_model
from config.neo4jdb import get_db_manager
from search_new.config import get_search_config
from search_new.utils.cache_manager import CacheManager, MemoryCacheBackend, DiskCacheBackend
from search_new.utils.vector_utils import VectorUtils


class BaseSearchTool(ABC):
    """搜索工具基础类，为各种搜索实现提供通用功能"""
    
    def __init__(self, 
                 cache_dir: Optional[str] = None,
                 enable_cache: bool = True,
                 llm=None,
                 embeddings=None):
        """
        初始化搜索工具
        
        参数:
            cache_dir: 缓存目录，用于存储搜索结果
            enable_cache: 是否启用缓存
            llm: 大语言模型实例
            embeddings: 嵌入模型实例
        """
        # 获取配置
        self.config = get_search_config()
        
        # 初始化模型
        self.llm = llm or get_llm_model()
        self.embeddings = embeddings or get_embeddings_model()

        # 初始化数据库连接
        self._setup_database()

        # 初始化缓存管理器
        if enable_cache:
            self._setup_cache(cache_dir)
        else:
            self.cache_manager = None
        
        # 性能监控指标
        self.performance_metrics = {
            "query_time": 0.0,      # 数据库查询时间
            "llm_time": 0.0,        # 大语言模型处理时间
            "cache_time": 0.0,      # 缓存操作时间
            "keyword_time": 0.0,    # 关键词提取时间
            "total_time": 0.0       # 总处理时间
        }
        
        # 错误统计
        self.error_stats = {
            "query_errors": 0,
            "llm_errors": 0,
            "cache_errors": 0,
            "keyword_errors": 0
        }
        
        # 设置处理链
        self._setup_chains()

        print(f"{self.__class__.__name__} 初始化完成")
    
    def _setup_cache(self, cache_dir: Optional[str] = None):
        """设置缓存管理器"""
        try:
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

    def _setup_database(self):
        """设置数据库连接"""
        try:
            # 使用项目统一的数据库管理器
            db_manager = get_db_manager()
            self.graph = db_manager.get_graph()
            self.driver = db_manager.get_driver()

        except Exception as e:
            print(f"数据库连接设置失败: {e}")
            self.graph = None
            self.driver = None

    def db_query(self, cypher: str, params: Dict[str, Any] = None):
        """
        执行Cypher查询

        参数:
            cypher: Cypher查询语句
            params: 查询参数

        返回:
            查询结果
        """
        if params is None:
            params = {}

        try:
            # 使用数据库管理器执行查询
            return get_db_manager().execute_query(cypher, params)
        except Exception as e:
            print(f"数据库查询失败: {e}")
            self.error_stats["query_errors"] += 1
            return None

    @abstractmethod
    def _setup_chains(self):
        """
        设置处理链，子类必须实现
        用于配置各种LLM处理链和提示模板
        """
        pass
    
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
    
    def _call_llm(self, messages, **kwargs) -> str:
        """
        调用大语言模型
        
        参数:
            messages: 消息列表或提示模板
            **kwargs: 其他参数
            
        返回:
            str: 模型响应
        """
        try:
            start_time = time.time()
            
            # 如果是提示模板，先调用invoke
            if hasattr(messages, 'invoke'):
                response = messages.invoke(kwargs)
            else:
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
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 关键词字典，包含低级和高级关键词
        """
        pass
    
    @abstractmethod
    def search(self, query: Union[str, Dict[str, Any]]) -> str:
        """
        执行搜索
        
        参数:
            query: 查询内容，可以是字符串或包含更多信息的字典
            
        返回:
            str: 搜索结果
        """
        pass
    
    def search_with_details(self, query: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行搜索并返回详细信息
        
        参数:
            query: 查询内容
            
        返回:
            Dict: 包含搜索结果和详细信息的字典
        """
        start_time = time.time()
        
        try:
            # 执行搜索
            result = self.search(query)
            
            return {
                "result": result,
                "performance": self.get_performance_metrics(),
                "error_stats": self.get_error_stats(),
                "total_time": time.time() - start_time,
                "tool_name": self.__class__.__name__
            }
            
        except Exception as e:
            print(f"详细搜索失败: {e}")
            return {
                "result": f"搜索失败: {str(e)}",
                "performance": self.get_performance_metrics(),
                "error_stats": self.get_error_stats(),
                "error": str(e),
                "total_time": time.time() - start_time,
                "tool_name": self.__class__.__name__
            }
    
    def get_tool(self) -> BaseTool:
        """
        获取LangChain兼容的工具实例
        
        返回:
            BaseTool: 搜索工具
        """
        # 创建动态工具类
        class DynamicSearchTool(BaseTool):
            name: str = f"{self.__class__.__name__.lower()}"
            description: str = f"{self.__class__.__name__} - 高级搜索工具，用于在知识库中查找信息"
            
            def _run(self_tool, query: Any) -> str:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")
        
        return DynamicSearchTool()
    
    def vector_search(self, query: str, candidates: List[Dict], 
                     embedding_field: str = "embedding",
                     top_k: int = 5) -> List[Dict]:
        """
        基于向量相似度的搜索方法
        
        参数:
            query: 搜索查询
            candidates: 候选项列表
            embedding_field: 嵌入向量字段名
            top_k: 返回的最大结果数
            
        返回:
            List[Dict]: 按相似度排序的候选项列表
        """
        try:
            # 生成查询的嵌入向量
            query_embedding = self.embeddings.embed_query(query)
            
            # 使用向量工具进行排序
            return VectorUtils.rank_by_similarity(
                query_embedding, 
                candidates, 
                embedding_field, 
                top_k=top_k
            )
            
        except Exception as e:
            print(f"向量搜索失败: {e}")
            return candidates[:top_k] if top_k else candidates
    
    def close(self):
        """关闭资源连接"""
        try:
            # 清理缓存
            if self.cache_manager:
                self.cache_manager.clear()
                
        except Exception as e:
            print(f"资源关闭失败: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保资源被正确释放"""
        self.close()
