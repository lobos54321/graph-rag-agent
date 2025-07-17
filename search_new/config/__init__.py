"""
搜索配置模块
"""

from config.settings import (
    LOCAL_SEARCH_CONFIG,
    GLOBAL_SEARCH_CONFIG,
    SEARCH_CACHE_CONFIG,
    REASONING_CONFIG,
)


class SearchConfig:
    """搜索配置类，基于项目统一配置"""

    def __init__(self):
        self.local_search = LocalSearchConfig()
        self.global_search = GlobalSearchConfig()
        self.cache = CacheConfig()
        self.reasoning = ReasoningConfig()


class LocalSearchConfig:
    """本地搜索配置"""

    def __init__(self):
        config = LOCAL_SEARCH_CONFIG
        self.top_entities = config["top_entities"]
        self.top_chunks = config["top_chunks"]
        self.top_communities = config["top_communities"]
        self.top_outside_rels = config["top_outside_rels"]
        self.top_inside_rels = config["top_inside_rels"]
        self.index_name = config["index_name"]
        self.response_type = config["response_type"]
        self.retrieval_query = config["retrieval_query"]


class GlobalSearchConfig:
    """全局搜索配置"""

    def __init__(self):
        config = GLOBAL_SEARCH_CONFIG
        self.default_level = config["default_level"]
        self.response_type = config["response_type"]
        self.batch_size = config["batch_size"]
        self.max_communities = config["max_communities"]


class CacheConfig:
    """缓存配置"""

    def __init__(self):
        config = SEARCH_CACHE_CONFIG
        self.base_cache_dir = config["base_cache_dir"]
        self.local_search_cache_dir = config["local_search_cache_dir"]
        self.global_search_cache_dir = config["global_search_cache_dir"]
        self.deep_research_cache_dir = config["deep_research_cache_dir"]
        self.max_cache_size = config["max_cache_size"]
        self.cache_ttl = config["cache_ttl"]
        self.memory_cache_enabled = config["memory_cache_enabled"]
        self.disk_cache_enabled = config["disk_cache_enabled"]


class ReasoningConfig:
    """推理配置"""

    def __init__(self):
        config = REASONING_CONFIG
        self.max_iterations = config["max_iterations"]
        self.max_search_limit = config["max_search_limit"]
        self.thinking_depth = config["thinking_depth"]
        self.exploration_width = config["exploration_width"]
        self.max_exploration_steps = config["max_exploration_steps"]
        self.max_evidence_items = config["max_evidence_items"]
        self.evidence_relevance_threshold = config["evidence_relevance_threshold"]


# 全局配置实例
_global_config = None


def get_search_config() -> SearchConfig:
    """获取全局搜索配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = SearchConfig()
    return _global_config


def get_reasoning_config() -> ReasoningConfig:
    """获取推理配置实例"""
    return get_search_config().reasoning


__all__ = [
    "SearchConfig",
    "LocalSearchConfig",
    "GlobalSearchConfig",
    "CacheConfig",
    "ReasoningConfig",
    "get_search_config",
    "get_reasoning_config"
]