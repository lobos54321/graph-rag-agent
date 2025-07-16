"""
异步处理器

提供异步回调处理和流式管理功能
"""

import asyncio
import queue
import threading
import time
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator, Callable
from dataclasses import dataclass, field

from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .stream_processor import StreamChunk, StreamStatus

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """流式事件数据类"""
    event_type: str  # token, start, end, error
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class AsyncIteratorCallbackHandler(AsyncCallbackHandler):
    """
    异步迭代器回调处理器
    
    用于处理LLM的流式输出，将token转换为异步迭代器
    """
    
    def __init__(self):
        """初始化异步回调处理器"""
        self.queue = asyncio.Queue()
        self.done = False
        self.error = None
        
        # 统计信息
        self.token_count = 0
        self.start_time = None
        self.end_time = None
        
        logger.debug("异步迭代器回调处理器初始化完成")
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """LLM开始时的回调"""
        self.start_time = time.time()
        self.token_count = 0
        self.done = False
        self.error = None
        
        await self.queue.put(StreamEvent(
            event_type="start",
            metadata={"prompts_count": len(prompts)}
        ))
        
        logger.debug("LLM开始生成")
    
    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """新token生成时的回调"""
        self.token_count += 1
        
        await self.queue.put(StreamEvent(
            event_type="token",
            content=token,
            metadata={"token_count": self.token_count}
        ))
    
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM结束时的回调"""
        self.end_time = time.time()
        self.done = True
        
        duration = self.end_time - self.start_time if self.start_time else 0
        
        await self.queue.put(StreamEvent(
            event_type="end",
            metadata={
                "token_count": self.token_count,
                "duration": duration,
                "tokens_per_second": self.token_count / duration if duration > 0 else 0
            }
        ))
        
        logger.debug(f"LLM生成完成，token数: {self.token_count}, 耗时: {duration:.2f}s")
    
    async def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """LLM错误时的回调"""
        self.error = error
        self.done = True
        
        await self.queue.put(StreamEvent(
            event_type="error",
            content=str(error),
            metadata={"error_type": type(error).__name__}
        ))
        
        logger.error(f"LLM生成错误: {error}")
    
    async def aiter(self) -> AsyncGenerator[StreamEvent, None]:
        """异步迭代器"""
        while not self.done or not self.queue.empty():
            try:
                # 等待事件，设置超时避免无限等待
                event = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                yield event
                
                # 如果是结束或错误事件，标记完成
                if event.event_type in ["end", "error"]:
                    break
                    
            except asyncio.TimeoutError:
                # 超时检查是否真的完成了
                if self.done:
                    break
                continue
            except Exception as e:
                logger.error(f"异步迭代器错误: {e}")
                break
    
    def reset(self):
        """重置处理器状态"""
        self.queue = asyncio.Queue()
        self.done = False
        self.error = None
        self.token_count = 0
        self.start_time = None
        self.end_time = None


class AsyncStreamManager:
    """
    异步流式管理器
    
    管理多个异步流式会话，提供统一的接口
    """
    
    def __init__(self, max_concurrent_streams: int = 10):
        """
        初始化异步流式管理器
        
        参数:
            max_concurrent_streams: 最大并发流数
        """
        self.max_concurrent_streams = max_concurrent_streams
        
        # 会话管理
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_handlers: Dict[str, AsyncIteratorCallbackHandler] = {}
        
        # 统计信息
        self.total_sessions = 0
        self.completed_sessions = 0
        self.failed_sessions = 0
        
        logger.info(f"异步流式管理器初始化完成，最大并发: {max_concurrent_streams}")
    
    async def create_stream_session(self, session_id: Optional[str] = None) -> str:
        """
        创建流式会话
        
        参数:
            session_id: 会话ID，如果为None则自动生成
            
        返回:
            str: 会话ID
        """
        try:
            # 检查并发限制
            if len(self.active_sessions) >= self.max_concurrent_streams:
                raise Exception("达到最大并发流式会话限制")
            
            if session_id is None:
                session_id = f"async_stream_{int(time.time() * 1000)}"
            
            # 创建回调处理器
            handler = AsyncIteratorCallbackHandler()
            
            # 注册会话
            self.active_sessions[session_id] = {
                "status": "created",
                "created_at": time.time(),
                "handler": handler
            }
            self.session_handlers[session_id] = handler
            
            self.total_sessions += 1
            
            logger.debug(f"创建流式会话: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"创建流式会话失败: {e}")
            raise
    
    async def start_stream(self, session_id: str) -> AsyncGenerator[StreamChunk, None]:
        """
        开始流式输出
        
        参数:
            session_id: 会话ID
            
        返回:
            AsyncGenerator[StreamChunk, None]: 流式数据块生成器
        """
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"会话不存在: {session_id}")
            
            session = self.active_sessions[session_id]
            handler = self.session_handlers[session_id]
            
            session["status"] = "streaming"
            session["started_at"] = time.time()
            
            # 发送开始状态
            yield StreamChunk(
                content="",
                chunk_type="status",
                metadata={"status": "streaming", "session_id": session_id}
            )
            
            # 异步迭代处理器输出
            async for event in handler.aiter():
                if event.event_type == "token":
                    yield StreamChunk(
                        content=event.content,
                        chunk_type="text",
                        metadata={
                            "session_id": session_id,
                            "token_count": event.metadata.get("token_count", 0)
                        }
                    )
                elif event.event_type == "end":
                    session["status"] = "completed"
                    session["completed_at"] = time.time()
                    self.completed_sessions += 1
                    
                    yield StreamChunk(
                        content="",
                        chunk_type="status",
                        metadata={
                            "status": "completed",
                            "session_id": session_id,
                            **event.metadata
                        },
                        is_final=True
                    )
                    break
                elif event.event_type == "error":
                    session["status"] = "failed"
                    session["error"] = event.content
                    self.failed_sessions += 1
                    
                    yield StreamChunk(
                        content="",
                        chunk_type="error",
                        metadata={
                            "error": event.content,
                            "session_id": session_id
                        }
                    )
                    break
            
        except Exception as e:
            logger.error(f"流式输出失败: {e}")
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["status"] = "failed"
                self.active_sessions[session_id]["error"] = str(e)
                self.failed_sessions += 1
            
            yield StreamChunk(
                content="",
                chunk_type="error",
                metadata={"error": str(e), "session_id": session_id}
            )
        finally:
            # 清理会话
            await self.cleanup_session(session_id)
    
    async def cleanup_session(self, session_id: str):
        """
        清理会话
        
        参数:
            session_id: 会话ID
        """
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            if session_id in self.session_handlers:
                del self.session_handlers[session_id]
            
            logger.debug(f"清理流式会话: {session_id}")
            
        except Exception as e:
            logger.error(f"清理会话失败: {e}")
    
    def get_session_handler(self, session_id: str) -> Optional[AsyncIteratorCallbackHandler]:
        """
        获取会话的回调处理器
        
        参数:
            session_id: 会话ID
            
        返回:
            AsyncIteratorCallbackHandler: 回调处理器
        """
        return self.session_handlers.get(session_id)
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话状态
        
        参数:
            session_id: 会话ID
            
        返回:
            Dict: 会话状态信息
        """
        session = self.active_sessions.get(session_id)
        if session:
            # 复制会话信息，排除handler对象
            status = {k: v for k, v in session.items() if k != "handler"}
            return status
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        返回:
            Dict: 统计信息
        """
        return {
            "total_sessions": self.total_sessions,
            "active_sessions": len(self.active_sessions),
            "completed_sessions": self.completed_sessions,
            "failed_sessions": self.failed_sessions,
            "success_rate": self.completed_sessions / self.total_sessions if self.total_sessions > 0 else 0
        }
    
    async def close(self):
        """关闭流式管理器"""
        try:
            # 清理所有活跃会话
            session_ids = list(self.active_sessions.keys())
            for session_id in session_ids:
                await self.cleanup_session(session_id)
            
            logger.info("异步流式管理器已关闭")
            
        except Exception as e:
            logger.error(f"异步流式管理器关闭失败: {e}")


# 全局异步流式管理器实例
_global_async_manager: Optional[AsyncStreamManager] = None


def get_async_stream_manager() -> AsyncStreamManager:
    """获取全局异步流式管理器实例"""
    global _global_async_manager
    if _global_async_manager is None:
        _global_async_manager = AsyncStreamManager()
    return _global_async_manager


async def reset_async_stream_manager():
    """重置全局异步流式管理器"""
    global _global_async_manager
    if _global_async_manager:
        await _global_async_manager.close()
    _global_async_manager = None
