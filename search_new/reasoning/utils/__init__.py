"""
推理工具模块

提供推理过程中使用的各种工具函数
"""

from .nlp_utils import (
    clean_text,
    extract_queries_from_text,
    split_sentences,
    is_query_like,
    extract_keywords,
    extract_entities,
    calculate_text_similarity,
    summarize_text,
    detect_language
)

from .prompts import (
    PromptTemplates,
    get_prompt_templates,
    get_prompt
)

from .kg_builder import (
    KnowledgeGraphBuilder,
    KGEntity,
    KGRelation,
    KnowledgeGraph
)

__all__ = [
    # NLP工具
    "clean_text",
    "extract_queries_from_text",
    "split_sentences",
    "is_query_like",
    "extract_keywords",
    "extract_entities",
    "calculate_text_similarity",
    "summarize_text",
    "detect_language",

    # 提示模板
    "PromptTemplates",
    "get_prompt_templates",
    "get_prompt",

    # 知识图谱构建
    "KnowledgeGraphBuilder",
    "KGEntity",
    "KGRelation",
    "KnowledgeGraph"
]