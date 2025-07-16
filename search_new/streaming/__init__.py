"""
流式处理模块

提供异步流式输出功能，支持实时响应和渐进式结果展示
"""

from .stream_processor import (
    StreamProcessor,
    StreamChunk,
    StreamStatus,
    get_stream_processor,
    reset_stream_processor
)
from .async_handler import (
    AsyncIteratorCallbackHandler,
    AsyncStreamManager,
    get_async_stream_manager,
    reset_async_stream_manager
)
from .stream_utils import (
    chunk_text,
    merge_chunks,
    format_stream_output,
    simulate_typing_effect,
    simulate_typing_effect_sync
)

__all__ = [
    # 流式处理器
    "StreamProcessor",
    "StreamChunk",
    "StreamStatus",
    "get_stream_processor",
    "reset_stream_processor",

    # 异步处理器
    "AsyncIteratorCallbackHandler",
    "AsyncStreamManager",
    "get_async_stream_manager",
    "reset_async_stream_manager",

    # 流式工具
    "chunk_text",
    "merge_chunks",
    "format_stream_output",
    "simulate_typing_effect",
    "simulate_typing_effect_sync"
]
