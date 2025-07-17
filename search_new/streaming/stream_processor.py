"""
流式处理器

核心流式处理功能，支持文本分块、状态管理和异步输出
"""

from typing import AsyncGenerator, Generator, List, Dict, Any, Optional
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum

class StreamStatus(Enum):
    """流式状态枚举"""
    STARTING = "starting"
    PROCESSING = "processing"
    STREAMING = "streaming"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class StreamChunk:
    """流式数据块"""
    content: str
    chunk_type: str = "text"  # text, status, metadata, error
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_final: bool = False


class StreamProcessor:
    """
    流式处理器：管理流式输出的核心组件
    
    主要功能：
    1. 文本分块和流式输出
    2. 状态管理和进度跟踪
    3. 异步处理和并发控制
    4. 错误处理和恢复
    """
    
    def __init__(self, chunk_size: int = 50, delay: float = 0.02):
        """
        初始化流式处理器
        
        参数:
            chunk_size: 分块大小
            delay: 输出延迟（秒）
        """
        self.chunk_size = chunk_size
        self.delay = delay
        
        # 状态管理
        self.status = StreamStatus.STARTING
        self.current_session_id: Optional[str] = None
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        
        # 配置
        self.max_concurrent_streams = 10
        self.stream_timeout = 300  # 5分钟
        
        print(f"流式处理器初始化完成，分块大小: {chunk_size}")
    
    async def stream_text(self, text: str, session_id: Optional[str] = None) -> AsyncGenerator[StreamChunk, None]:
        """
        流式输出文本
        
        参数:
            text: 要输出的文本
            session_id: 会话ID
            
        返回:
            AsyncGenerator[StreamChunk, None]: 流式数据块生成器
        """
        try:
            if session_id is None:
                session_id = f"stream_{int(time.time() * 1000)}"
            
            self.current_session_id = session_id
            self.status = StreamStatus.PROCESSING
            
            # 注册流式会话
            self._register_stream(session_id, {"text": text, "start_time": time.time()})
            
            # 发送开始状态
            yield StreamChunk(
                content="",
                chunk_type="status",
                metadata={"status": "starting", "session_id": session_id}
            )
            
            self.status = StreamStatus.STREAMING
            
            # 分块输出文本
            chunks = self._chunk_text(text)
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks):
                # 检查会话是否被取消
                if not self._is_stream_active(session_id):
                    yield StreamChunk(
                        content="",
                        chunk_type="status",
                        metadata={"status": "cancelled", "session_id": session_id}
                    )
                    return
                
                # 输出文本块
                yield StreamChunk(
                    content=chunk,
                    chunk_type="text",
                    metadata={
                        "progress": (i + 1) / total_chunks,
                        "chunk_index": i,
                        "total_chunks": total_chunks,
                        "session_id": session_id
                    }
                )
                
                # 延迟以模拟打字效果
                if i < total_chunks - 1:  # 最后一块不延迟
                    await asyncio.sleep(self.delay)
            
            # 发送完成状态
            self.status = StreamStatus.COMPLETED
            yield StreamChunk(
                content="",
                chunk_type="status",
                metadata={"status": "completed", "session_id": session_id},
                is_final=True
            )
            
            # 注销流式会话
            self._unregister_stream(session_id)
            
        except asyncio.CancelledError:
            self.status = StreamStatus.CANCELLED
            yield StreamChunk(
                content="",
                chunk_type="status",
                metadata={"status": "cancelled", "session_id": session_id}
            )
            self._unregister_stream(session_id)
            
        except Exception as e:
            self.status = StreamStatus.ERROR
            print(f"流式处理失败: {e}")
            yield StreamChunk(
                content="",
                chunk_type="error",
                metadata={"error": str(e), "session_id": session_id}
            )
            self._unregister_stream(session_id)
    
    def stream_text_sync(self, text: str, session_id: Optional[str] = None) -> Generator[StreamChunk, None, None]:
        """
        同步版本的流式输出
        
        参数:
            text: 要输出的文本
            session_id: 会话ID
            
        返回:
            Generator[StreamChunk, None, None]: 流式数据块生成器
        """
        try:
            if session_id is None:
                session_id = f"stream_sync_{int(time.time() * 1000)}"
            
            self.current_session_id = session_id
            self.status = StreamStatus.PROCESSING
            
            # 注册流式会话
            self._register_stream(session_id, {"text": text, "start_time": time.time()})
            
            # 发送开始状态
            yield StreamChunk(
                content="",
                chunk_type="status",
                metadata={"status": "starting", "session_id": session_id}
            )
            
            self.status = StreamStatus.STREAMING
            
            # 分块输出文本
            chunks = self._chunk_text(text)
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks):
                # 检查会话是否被取消
                if not self._is_stream_active(session_id):
                    yield StreamChunk(
                        content="",
                        chunk_type="status",
                        metadata={"status": "cancelled", "session_id": session_id}
                    )
                    return
                
                # 输出文本块
                yield StreamChunk(
                    content=chunk,
                    chunk_type="text",
                    metadata={
                        "progress": (i + 1) / total_chunks,
                        "chunk_index": i,
                        "total_chunks": total_chunks,
                        "session_id": session_id
                    }
                )
                
                # 同步延迟
                if i < total_chunks - 1:
                    time.sleep(self.delay)
            
            # 发送完成状态
            self.status = StreamStatus.COMPLETED
            yield StreamChunk(
                content="",
                chunk_type="status",
                metadata={"status": "completed", "session_id": session_id},
                is_final=True
            )
            
            # 注销流式会话
            self._unregister_stream(session_id)
            
        except Exception as e:
            self.status = StreamStatus.ERROR
            print(f"同步流式处理失败: {e}")
            yield StreamChunk(
                content="",
                chunk_type="error",
                metadata={"error": str(e), "session_id": session_id}
            )
            self._unregister_stream(session_id)
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        将文本分块
        
        参数:
            text: 输入文本
            
        返回:
            List[str]: 文本块列表
        """
        if not text:
            return []
        
        try:
            chunks = []
            
            # 按句子分割
            import re
            sentences = re.split(r'([.!?。！？]\s*)', text)
            
            current_chunk = ""
            for i in range(len(sentences)):
                sentence = sentences[i]
                
                # 如果当前块加上新句子超过限制，输出当前块
                if len(current_chunk + sentence) > self.chunk_size and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    current_chunk += sentence
            
            # 添加最后一块
            if current_chunk:
                chunks.append(current_chunk)
            
            # 如果没有句子分割符，按字符分块
            if len(chunks) <= 1 and len(text) > self.chunk_size:
                chunks = []
                for i in range(0, len(text), self.chunk_size):
                    chunks.append(text[i:i + self.chunk_size])
            
            return chunks if chunks else [text]
            
        except Exception as e:
            print(f"文本分块失败: {e}")
            return [text]
    
    def _register_stream(self, session_id: str, metadata: Dict[str, Any]):
        """注册流式会话"""
        try:
            # 检查并发限制
            if len(self.active_streams) >= self.max_concurrent_streams:
                # 清理超时的会话
                self._cleanup_expired_streams()
                
                if len(self.active_streams) >= self.max_concurrent_streams:
                    raise Exception("达到最大并发流式会话限制")
            
            self.active_streams[session_id] = {
                **metadata,
                "status": "active",
                "registered_at": time.time()
            }
            
            print(f"注册流式会话: {session_id}")
            
        except Exception as e:
            print(f"注册流式会话失败: {e}")
    
    def _unregister_stream(self, session_id: str):
        """注销流式会话"""
        try:
            if session_id in self.active_streams:
                del self.active_streams[session_id]
                print(f"注销流式会话: {session_id}")
                
        except Exception as e:
            print(f"注销流式会话失败: {e}")
    
    def _is_stream_active(self, session_id: str) -> bool:
        """检查流式会话是否活跃"""
        return session_id in self.active_streams and \
               self.active_streams[session_id].get("status") == "active"
    
    def _cleanup_expired_streams(self):
        """清理过期的流式会话"""
        try:
            current_time = time.time()
            expired_sessions = []
            
            for session_id, metadata in self.active_streams.items():
                if current_time - metadata.get("registered_at", 0) > self.stream_timeout:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                self._unregister_stream(session_id)
                print(f"清理过期流式会话: {session_id}")
                
        except Exception as e:
            print(f"清理过期会话失败: {e}")
    
    def cancel_stream(self, session_id: str) -> bool:
        """
        取消流式会话
        
        参数:
            session_id: 会话ID
            
        返回:
            bool: 是否成功取消
        """
        try:
            if session_id in self.active_streams:
                self.active_streams[session_id]["status"] = "cancelled"
                print(f"取消流式会话: {session_id}")
                return True
            return False
            
        except Exception as e:
            print(f"取消流式会话失败: {e}")
            return False
    
    def get_stream_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取流式会话状态
        
        参数:
            session_id: 会话ID
            
        返回:
            Dict: 会话状态信息
        """
        try:
            if session_id in self.active_streams:
                return self.active_streams[session_id].copy()
            return None
            
        except Exception as e:
            print(f"获取流式状态失败: {e}")
            return None
    
    def get_active_streams(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有活跃的流式会话
        
        返回:
            Dict: 活跃会话字典
        """
        return self.active_streams.copy()
    
    def close(self):
        """关闭流式处理器"""
        try:
            # 取消所有活跃会话
            for session_id in list(self.active_streams.keys()):
                self.cancel_stream(session_id)
            
            self.active_streams.clear()
            self.status = StreamStatus.COMPLETED
            
            print("流式处理器已关闭")
            
        except Exception as e:
            print(f"流式处理器关闭失败: {e}")


# 全局流式处理器实例
_global_stream_processor: Optional[StreamProcessor] = None


def get_stream_processor() -> StreamProcessor:
    """获取全局流式处理器实例"""
    global _global_stream_processor
    if _global_stream_processor is None:
        _global_stream_processor = StreamProcessor()
    return _global_stream_processor


def reset_stream_processor():
    """重置全局流式处理器"""
    global _global_stream_processor
    if _global_stream_processor:
        _global_stream_processor.close()
    _global_stream_processor = None
