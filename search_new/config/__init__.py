"""
搜索配置管理模块

提供统一的配置管理接口，包括搜索配置和推理配置
"""

from .search_config import (
    LocalSearchConfig,
    GlobalSearchConfig,
    CacheConfig,
    PerformanceConfig,
    SearchConfig,
    get_search_config,
    set_search_config,
    reset_search_config
)

from .reasoning_config import (
    ThinkingConfig,
    ExplorationConfig,
    EvidenceConfig,
    CommunityConfig,
    ValidationConfig,
    PromptConfig,
    ReasoningConfig,
    get_reasoning_config,
    set_reasoning_config
)

__all__ = [
    # 搜索配置
    "LocalSearchConfig",
    "GlobalSearchConfig",
    "CacheConfig",
    "PerformanceConfig",
    "SearchConfig",
    "get_search_config",
    "set_search_config",
    "reset_search_config",

    # 推理配置
    "ThinkingConfig",
    "ExplorationConfig",
    "EvidenceConfig",
    "CommunityConfig",
    "ValidationConfig",
    "PromptConfig",
    "ReasoningConfig",
    "get_reasoning_config",
    "set_reasoning_config"
]