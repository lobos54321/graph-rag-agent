"""
搜索配置管理模块

统一管理搜索相关的配置参数，提供配置验证和默认值管理
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import os


@dataclass
class LocalSearchConfig:
    """本地搜索配置"""
    # 向量检索参数
    top_entities: int = 10
    top_chunks: int = 10
    top_communities: int = 2
    top_outside_rels: int = 10
    top_inside_rels: int = 10
    
    # 索引配置
    index_name: str = "vector"
    response_type: str = "多个段落"
    
    # 检索查询模板
    retrieval_query: str = """
    WITH node AS chunk, score AS similarity
    CALL {
        WITH chunk
        MATCH (chunk)-[:PART_OF]->(d:__Document__)
        RETURN d.id AS document_id, d.title AS document_title
    }
    CALL {
        WITH chunk
        OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:__Entity__)
        WITH e ORDER BY e.rank DESC LIMIT $topChunks
        RETURN collect(e{.id, .description, .rank}) AS entities
    }
    CALL {
        WITH chunk
        OPTIONAL MATCH (chunk)-[:PART_OF]->(d:__Document__)
        OPTIONAL MATCH (d)-[:IN_COMMUNITY]->(c:__Community__)
        WITH c ORDER BY c.rank DESC LIMIT $topCommunities
        RETURN collect(c{.id, .summary, .rank}) AS communities
    }
    CALL {
        WITH chunk
        OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:__Entity__)
        OPTIONAL MATCH (e)-[r:RELATED]->(e2:__Entity__)
        WHERE NOT (chunk)-[:HAS_ENTITY]->(e2)
        WITH r ORDER BY r.rank DESC LIMIT $topOutsideRels
        RETURN collect(r{.description, .rank, source: e.id, target: e2.id}) AS outside_rels
    }
    CALL {
        WITH chunk
        OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:__Entity__)
        OPTIONAL MATCH (e)-[r:RELATED]->(e2:__Entity__)
        WHERE (chunk)-[:HAS_ENTITY]->(e2)
        WITH r ORDER BY r.rank DESC LIMIT $topInsideRels
        RETURN collect(r{.description, .rank, source: e.id, target: e2.id}) AS inside_rels
    }
    RETURN chunk.text AS text,
           similarity,
           document_id,
           document_title,
           entities,
           communities,
           outside_rels,
           inside_rels
    ORDER BY similarity DESC
    """


@dataclass
class GlobalSearchConfig:
    """全局搜索配置"""
    # 社区层级配置
    default_level: int = 2
    response_type: str = "多个段落"
    
    # 批处理配置
    batch_size: int = 10
    max_communities: int = 100


@dataclass
class CacheConfig:
    """缓存配置"""
    # 缓存目录
    base_cache_dir: str = "./cache"
    local_search_cache_dir: str = "./cache/local_search"
    global_search_cache_dir: str = "./cache/global_search"
    deep_research_cache_dir: str = "./cache/deep_research"
    
    # 缓存策略
    max_cache_size: int = 200
    cache_ttl: int = 3600  # 1小时
    
    # 内存缓存配置
    memory_cache_enabled: bool = True
    disk_cache_enabled: bool = True


@dataclass
class PerformanceConfig:
    """性能配置"""
    # 超时配置
    query_timeout: int = 30
    llm_timeout: int = 60
    
    # 并发配置
    max_concurrent_queries: int = 5
    max_workers: int = 4
    
    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class ReasoningConfig:
    """推理配置"""
    # 迭代配置
    max_iterations: int = 5
    max_search_limit: int = 10
    
    # 思考引擎配置
    thinking_depth: int = 3
    exploration_width: int = 3
    max_exploration_steps: int = 5
    
    # 证据链配置
    max_evidence_items: int = 50
    evidence_relevance_threshold: float = 0.7


@dataclass
class SearchConfig:
    """搜索模块总配置"""
    local_search: LocalSearchConfig = field(default_factory=LocalSearchConfig)
    global_search: GlobalSearchConfig = field(default_factory=GlobalSearchConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    
    # 调试配置
    debug_mode: bool = False
    verbose_logging: bool = False
    
    @classmethod
    def from_env(cls) -> 'SearchConfig':
        """从环境变量创建配置"""
        config = cls()
        
        # 从环境变量读取配置
        if os.getenv('SEARCH_DEBUG_MODE'):
            config.debug_mode = os.getenv('SEARCH_DEBUG_MODE').lower() == 'true'
            
        if os.getenv('SEARCH_VERBOSE_LOGGING'):
            config.verbose_logging = os.getenv('SEARCH_VERBOSE_LOGGING').lower() == 'true'
            
        if os.getenv('SEARCH_MAX_ITERATIONS'):
            config.reasoning.max_iterations = int(os.getenv('SEARCH_MAX_ITERATIONS'))
            
        if os.getenv('SEARCH_CACHE_SIZE'):
            config.cache.max_cache_size = int(os.getenv('SEARCH_CACHE_SIZE'))
            
        return config
    
    def validate(self) -> bool:
        """验证配置有效性"""
        try:
            # 验证数值范围
            assert self.local_search.top_entities > 0, "top_entities must be positive"
            assert self.global_search.default_level >= 0, "default_level must be non-negative"
            assert self.cache.max_cache_size > 0, "max_cache_size must be positive"
            assert self.performance.max_concurrent_queries > 0, "max_concurrent_queries must be positive"
            assert self.reasoning.max_iterations > 0, "max_iterations must be positive"
            
            # 验证目录路径
            cache_dirs = [
                self.cache.base_cache_dir,
                self.cache.local_search_cache_dir,
                self.cache.global_search_cache_dir,
                self.cache.deep_research_cache_dir
            ]
            
            for cache_dir in cache_dirs:
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir, exist_ok=True)
            
            return True
            
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'local_search': {
                'top_entities': self.local_search.top_entities,
                'top_chunks': self.local_search.top_chunks,
                'top_communities': self.local_search.top_communities,
                'top_outside_rels': self.local_search.top_outside_rels,
                'top_inside_rels': self.local_search.top_inside_rels,
                'index_name': self.local_search.index_name,
                'response_type': self.local_search.response_type
            },
            'global_search': {
                'default_level': self.global_search.default_level,
                'response_type': self.global_search.response_type,
                'batch_size': self.global_search.batch_size,
                'max_communities': self.global_search.max_communities
            },
            'cache': {
                'max_cache_size': self.cache.max_cache_size,
                'cache_ttl': self.cache.cache_ttl,
                'memory_cache_enabled': self.cache.memory_cache_enabled,
                'disk_cache_enabled': self.cache.disk_cache_enabled
            },
            'performance': {
                'query_timeout': self.performance.query_timeout,
                'llm_timeout': self.performance.llm_timeout,
                'max_concurrent_queries': self.performance.max_concurrent_queries,
                'max_workers': self.performance.max_workers
            },
            'reasoning': {
                'max_iterations': self.reasoning.max_iterations,
                'max_search_limit': self.reasoning.max_search_limit,
                'thinking_depth': self.reasoning.thinking_depth,
                'exploration_width': self.reasoning.exploration_width
            },
            'debug_mode': self.debug_mode,
            'verbose_logging': self.verbose_logging
        }


# 全局配置实例
_global_config: Optional[SearchConfig] = None


def get_search_config() -> SearchConfig:
    """获取全局搜索配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = SearchConfig.from_env()
        if not _global_config.validate():
            raise ValueError("搜索配置验证失败")
    return _global_config


def set_search_config(config: SearchConfig):
    """设置全局搜索配置"""
    global _global_config
    if not config.validate():
        raise ValueError("搜索配置验证失败")
    _global_config = config


def reset_search_config():
    """重置搜索配置"""
    global _global_config
    _global_config = None
