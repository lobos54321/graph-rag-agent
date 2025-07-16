"""
流式工具函数

提供流式处理相关的工具函数
"""

import asyncio
import time
import re
from typing import List, Dict, Any, Optional, AsyncGenerator, Generator, Union

from .stream_processor import StreamChunk


def chunk_text(text: str, chunk_size: int = 50, method: str = "sentence") -> List[str]:
    """
    将文本分块
    
    参数:
        text: 输入文本
        chunk_size: 分块大小
        method: 分块方法 (sentence, word, char)
        
    返回:
        List[str]: 文本块列表
    """
    if not text:
        return []
    
    try:
        if method == "sentence":
            return _chunk_by_sentence(text, chunk_size)
        elif method == "word":
            return _chunk_by_word(text, chunk_size)
        elif method == "char":
            return _chunk_by_char(text, chunk_size)
        else:
            return _chunk_by_sentence(text, chunk_size)
            
    except Exception as e:
        # 如果分块失败，返回原文本
        return [text]


def _chunk_by_sentence(text: str, chunk_size: int) -> List[str]:
    """按句子分块"""
    # 按句子分割
    sentences = re.split(r'([.!?。！？]\s*)', text)
    
    chunks = []
    current_chunk = ""
    
    for i in range(len(sentences)):
        sentence = sentences[i]
        
        # 如果当前块加上新句子超过限制，输出当前块
        if len(current_chunk + sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence
    
    # 添加最后一块
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]


def _chunk_by_word(text: str, chunk_size: int) -> List[str]:
    """按单词分块"""
    words = text.split()
    chunks = []
    
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks if chunks else [text]


def _chunk_by_char(text: str, chunk_size: int) -> List[str]:
    """按字符分块"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


def merge_chunks(chunks: List[StreamChunk]) -> str:
    """
    合并流式数据块
    
    参数:
        chunks: 流式数据块列表
        
    返回:
        str: 合并后的文本
    """
    try:
        text_chunks = []
        for chunk in chunks:
            if chunk.chunk_type == "text" and chunk.content:
                text_chunks.append(chunk.content)
        
        return "".join(text_chunks)
        
    except Exception as e:
        return ""


def format_stream_output(chunks: List[StreamChunk], include_metadata: bool = False) -> Dict[str, Any]:
    """
    格式化流式输出
    
    参数:
        chunks: 流式数据块列表
        include_metadata: 是否包含元数据
        
    返回:
        Dict: 格式化后的输出
    """
    try:
        result = {
            "content": merge_chunks(chunks),
            "chunk_count": len(chunks),
            "text_chunks": len([c for c in chunks if c.chunk_type == "text"]),
            "status_chunks": len([c for c in chunks if c.chunk_type == "status"]),
            "error_chunks": len([c for c in chunks if c.chunk_type == "error"])
        }
        
        if include_metadata:
            result["chunks"] = [
                {
                    "content": chunk.content,
                    "type": chunk.chunk_type,
                    "timestamp": chunk.timestamp,
                    "metadata": chunk.metadata,
                    "is_final": chunk.is_final
                }
                for chunk in chunks
            ]
        
        # 提取状态信息
        status_chunks = [c for c in chunks if c.chunk_type == "status"]
        if status_chunks:
            last_status = status_chunks[-1]
            result["final_status"] = last_status.metadata.get("status", "unknown")
        
        # 提取错误信息
        error_chunks = [c for c in chunks if c.chunk_type == "error"]
        if error_chunks:
            result["errors"] = [c.metadata.get("error", c.content) for c in error_chunks]
        
        return result
        
    except Exception as e:
        return {"content": "", "error": str(e)}


async def simulate_typing_effect(text: str, delay: float = 0.02, 
                                chunk_size: int = 1) -> AsyncGenerator[str, None]:
    """
    模拟打字效果
    
    参数:
        text: 要输出的文本
        delay: 每个字符的延迟
        chunk_size: 每次输出的字符数
        
    返回:
        AsyncGenerator[str, None]: 字符流生成器
    """
    try:
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            yield chunk
            
            if i + chunk_size < len(text):  # 最后一块不延迟
                await asyncio.sleep(delay)
                
    except Exception as e:
        yield text  # 如果出错，直接返回完整文本


def simulate_typing_effect_sync(text: str, delay: float = 0.02, 
                               chunk_size: int = 1) -> Generator[str, None, None]:
    """
    同步版本的打字效果
    
    参数:
        text: 要输出的文本
        delay: 每个字符的延迟
        chunk_size: 每次输出的字符数
        
    返回:
        Generator[str, None, None]: 字符流生成器
    """
    try:
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            yield chunk
            
            if i + chunk_size < len(text):  # 最后一块不延迟
                time.sleep(delay)
                
    except Exception as e:
        yield text  # 如果出错，直接返回完整文本


def estimate_stream_duration(text: str, delay: float = 0.02, 
                            chunk_size: int = 50) -> float:
    """
    估算流式输出持续时间
    
    参数:
        text: 文本内容
        delay: 延迟时间
        chunk_size: 分块大小
        
    返回:
        float: 估算的持续时间（秒）
    """
    try:
        if not text:
            return 0.0
        
        chunks = chunk_text(text, chunk_size)
        # 总延迟 = (块数 - 1) * 延迟时间
        return max(0, len(chunks) - 1) * delay
        
    except Exception as e:
        return 0.0


def create_progress_indicator(current: int, total: int, width: int = 20) -> str:
    """
    创建进度指示器
    
    参数:
        current: 当前进度
        total: 总数
        width: 进度条宽度
        
    返回:
        str: 进度条字符串
    """
    try:
        if total <= 0:
            return "[" + "?" * width + "]"
        
        progress = min(current / total, 1.0)
        filled = int(progress * width)
        bar = "█" * filled + "░" * (width - filled)
        percentage = int(progress * 100)
        
        return f"[{bar}] {percentage}%"
        
    except Exception as e:
        return "[" + "?" * width + "]"


def filter_stream_chunks(chunks: List[StreamChunk], 
                        chunk_types: Optional[List[str]] = None,
                        min_content_length: int = 0) -> List[StreamChunk]:
    """
    过滤流式数据块
    
    参数:
        chunks: 流式数据块列表
        chunk_types: 要保留的块类型列表
        min_content_length: 最小内容长度
        
    返回:
        List[StreamChunk]: 过滤后的数据块列表
    """
    try:
        filtered_chunks = []
        
        for chunk in chunks:
            # 类型过滤
            if chunk_types and chunk.chunk_type not in chunk_types:
                continue
            
            # 长度过滤
            if len(chunk.content) < min_content_length:
                continue
            
            filtered_chunks.append(chunk)
        
        return filtered_chunks
        
    except Exception as e:
        return chunks


def calculate_stream_stats(chunks: List[StreamChunk]) -> Dict[str, Any]:
    """
    计算流式统计信息
    
    参数:
        chunks: 流式数据块列表
        
    返回:
        Dict: 统计信息
    """
    try:
        if not chunks:
            return {"total_chunks": 0}
        
        # 基础统计
        total_chunks = len(chunks)
        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        status_chunks = [c for c in chunks if c.chunk_type == "status"]
        error_chunks = [c for c in chunks if c.chunk_type == "error"]
        
        # 时间统计
        timestamps = [c.timestamp for c in chunks if c.timestamp]
        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
        
        # 内容统计
        total_content_length = sum(len(c.content) for c in text_chunks)
        avg_chunk_size = total_content_length / len(text_chunks) if text_chunks else 0
        
        return {
            "total_chunks": total_chunks,
            "text_chunks": len(text_chunks),
            "status_chunks": len(status_chunks),
            "error_chunks": len(error_chunks),
            "total_content_length": total_content_length,
            "avg_chunk_size": avg_chunk_size,
            "duration": duration,
            "chunks_per_second": total_chunks / duration if duration > 0 else 0,
            "has_errors": len(error_chunks) > 0,
            "is_completed": any(c.is_final for c in chunks)
        }
        
    except Exception as e:
        return {"total_chunks": len(chunks), "error": str(e)}
