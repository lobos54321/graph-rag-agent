"""
推理模块

提供完整的推理功能，包括思考引擎、搜索组件、验证组件等
"""

# 核心推理组件
from .core import (
    ThinkingEngine,
    ThinkingStep,
    ThinkingSession,
    QueryGenerator,
    QueryContext,
    EvidenceTracker,
    Evidence,
    ReasoningStep,
    EvidenceChain
)

# 推理搜索组件
from .search import (
    DualPathSearcher,
    ChainedExploration,
    ExplorationNode,
    ExplorationPath
)

# 验证组件
from .validation import (
    AnswerValidator,
    ValidationResult,
    AnswerQuality,
    ComplexityEstimator,
    ComplexityMetrics,
    ComplexityLevel
)

# 工具函数
from .utils import (
    clean_text,
    extract_queries_from_text,
    extract_keywords,
    extract_entities,
    PromptTemplates,
    get_prompt_templates,
    get_prompt,
    KnowledgeGraphBuilder,
    KGEntity,
    KGRelation,
    KnowledgeGraph
)

__all__ = [
    # 核心推理组件
    "ThinkingEngine",
    "ThinkingStep",
    "ThinkingSession",
    "QueryGenerator",
    "QueryContext",
    "EvidenceTracker",
    "Evidence",
    "ReasoningStep",
    "EvidenceChain",

    # 推理搜索组件
    "DualPathSearcher",
    "ChainedExploration",
    "ExplorationNode",
    "ExplorationPath",

    # 验证组件
    "AnswerValidator",
    "ValidationResult",
    "AnswerQuality",
    "ComplexityEstimator",
    "ComplexityMetrics",
    "ComplexityLevel",

    # 工具函数
    "clean_text",
    "extract_queries_from_text",
    "extract_keywords",
    "extract_entities",
    "PromptTemplates",
    "get_prompt_templates",
    "get_prompt",
    "KnowledgeGraphBuilder",
    "KGEntity",
    "KGRelation",
    "KnowledgeGraph"
]