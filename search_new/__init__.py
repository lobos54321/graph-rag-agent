"""
搜索模块重构版本

提供高质量、可维护的搜索功能实现
"""

# 核心搜索类
from .core import BaseSearch, LocalSearch, GlobalSearch

# 搜索工具类
from .tools import (
    BaseSearchTool,
    LocalSearchTool,
    GlobalSearchTool,
    HybridSearchTool,
    NaiveSearchTool,
    DeepResearchTool,
    DeeperResearchTool
)

# 配置管理
from .config import (
    SearchConfig,
    ReasoningConfig,
    get_search_config,
    get_reasoning_config
)

# 推理组件
from .reasoning import (
    ThinkingEngine,
    QueryGenerator,
    EvidenceTracker,
    DualPathSearcher,
    ChainedExploration,
    AnswerValidator,
    ComplexityEstimator
)

# 流式处理
from .streaming import (
    StreamProcessor,
    AsyncIteratorCallbackHandler,
    AsyncStreamManager,
    get_stream_processor,
    get_async_stream_manager
)

# 工具函数
from .utils import (
    VectorUtils,
    PerformanceMonitor,
    get_performance_monitor
)

__all__ = [
    # 核心搜索类
    "BaseSearch",
    "LocalSearch",
    "GlobalSearch",

    # 搜索工具类
    "BaseSearchTool",
    "LocalSearchTool",
    "GlobalSearchTool",
    "HybridSearchTool",
    "NaiveSearchTool",
    "DeepResearchTool",
    "DeeperResearchTool",

    # 配置管理
    "SearchConfig",
    "ReasoningConfig",
    "get_search_config",
    "get_reasoning_config",

    # 推理组件
    "ThinkingEngine",
    "QueryGenerator",
    "EvidenceTracker",
    "DualPathSearcher",
    "ChainedExploration",
    "AnswerValidator",
    "ComplexityEstimator",

    # 流式处理
    "StreamProcessor",
    "AsyncIteratorCallbackHandler",
    "AsyncStreamManager",
    "get_stream_processor",
    "get_async_stream_manager",

    # 工具函数
    "VectorUtils",
    "PerformanceMonitor",
    "get_performance_monitor"
]

# 版本信息
__version__ = "2.0.0"
__author__ = "GraphRAG Agent Team"
__description__ = "重构后的搜索模块，提供高质量、可维护的搜索功能"