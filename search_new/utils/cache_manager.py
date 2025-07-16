"""
缓存管理器

提供统一的缓存管理功能，支持内存缓存和磁盘缓存
"""

import os
import json
import time
import hashlib
import pickle
from typing import Any, Optional, Dict, List
from abc import ABC, abstractmethod
import threading
import logging

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """缓存后端抽象基类"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """清空缓存"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        pass


class MemoryCacheBackend(CacheBackend):
    """内存缓存后端"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def _is_expired(self, key: str) -> bool:
        """检查缓存项是否过期"""
        if key not in self._cache:
            return True
            
        item = self._cache[key]
        if item.get('ttl') is None:
            return False
            
        return time.time() > item['created_at'] + item['ttl']
    
    def _evict_if_needed(self):
        """如果需要，执行缓存淘汰"""
        if len(self._cache) < self.max_size:
            return
            
        # 删除过期项
        expired_keys = [key for key in self._cache if self._is_expired(key)]
        for key in expired_keys:
            del self._cache[key]
            self._access_times.pop(key, None)
        
        # 如果仍然超过限制，使用LRU淘汰
        if len(self._cache) >= self.max_size:
            # 按访问时间排序，删除最久未访问的项
            sorted_keys = sorted(self._access_times.items(), key=lambda x: x[1])
            keys_to_remove = sorted_keys[:len(self._cache) - self.max_size + 1]
            
            for key, _ in keys_to_remove:
                self._cache.pop(key, None)
                self._access_times.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key not in self._cache or self._is_expired(key):
                return None
                
            # 更新访问时间
            self._access_times[key] = time.time()
            return self._cache[key]['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        with self._lock:
            try:
                self._evict_if_needed()
                
                self._cache[key] = {
                    'value': value,
                    'created_at': time.time(),
                    'ttl': ttl or self.default_ttl
                }
                self._access_times[key] = time.time()
                return True
                
            except Exception as e:
                logger.error(f"设置内存缓存失败: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_times.pop(key, None)
                return True
            return False
    
    def clear(self) -> bool:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            return True
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            return key in self._cache and not self._is_expired(key)


class DiskCacheBackend(CacheBackend):
    """磁盘缓存后端"""
    
    def __init__(self, cache_dir: str, default_ttl: int = 3600):
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        self._lock = threading.RLock()
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_file_path(self, key: str) -> str:
        """获取缓存文件路径"""
        # 使用MD5哈希避免文件名过长或包含特殊字符
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    def _is_expired(self, file_path: str, ttl: int) -> bool:
        """检查缓存文件是否过期"""
        if not os.path.exists(file_path):
            return True
            
        file_mtime = os.path.getmtime(file_path)
        return time.time() > file_mtime + ttl
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            file_path = self._get_file_path(key)
            
            try:
                if not os.path.exists(file_path):
                    return None
                
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # 检查是否过期
                ttl = data.get('ttl', self.default_ttl)
                if self._is_expired(file_path, ttl):
                    os.remove(file_path)
                    return None
                
                return data['value']
                
            except Exception as e:
                logger.error(f"读取磁盘缓存失败: {e}")
                # 删除损坏的缓存文件
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        with self._lock:
            file_path = self._get_file_path(key)
            
            try:
                data = {
                    'value': value,
                    'ttl': ttl or self.default_ttl,
                    'created_at': time.time()
                }
                
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                
                return True
                
            except Exception as e:
                logger.error(f"写入磁盘缓存失败: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        with self._lock:
            file_path = self._get_file_path(key)
            
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    return True
                return False
                
            except Exception as e:
                logger.error(f"删除磁盘缓存失败: {e}")
                return False
    
    def clear(self) -> bool:
        """清空缓存"""
        with self._lock:
            try:
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.cache'):
                        file_path = os.path.join(self.cache_dir, filename)
                        os.remove(file_path)
                return True
                
            except Exception as e:
                logger.error(f"清空磁盘缓存失败: {e}")
                return False
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            file_path = self._get_file_path(key)
            
            if not os.path.exists(file_path):
                return False
            
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                ttl = data.get('ttl', self.default_ttl)
                return not self._is_expired(file_path, ttl)
                
            except:
                return False


class CacheManager:
    """统一缓存管理器"""
    
    def __init__(self, 
                 memory_backend: Optional[MemoryCacheBackend] = None,
                 disk_backend: Optional[DiskCacheBackend] = None,
                 use_memory: bool = True,
                 use_disk: bool = True):
        
        self.use_memory = use_memory and memory_backend is not None
        self.use_disk = use_disk and disk_backend is not None
        
        self.memory_backend = memory_backend if self.use_memory else None
        self.disk_backend = disk_backend if self.use_disk else None
        
        if not self.use_memory and not self.use_disk:
            raise ValueError("至少需要启用一种缓存后端")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值，优先从内存缓存获取"""
        # 先尝试内存缓存
        if self.use_memory:
            value = self.memory_backend.get(key)
            if value is not None:
                return value
        
        # 再尝试磁盘缓存
        if self.use_disk:
            value = self.disk_backend.get(key)
            if value is not None:
                # 将磁盘缓存的值加载到内存缓存
                if self.use_memory:
                    self.memory_backend.set(key, value)
                return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值，同时写入内存和磁盘缓存"""
        success = True
        
        if self.use_memory:
            success &= self.memory_backend.set(key, value, ttl)
        
        if self.use_disk:
            success &= self.disk_backend.set(key, value, ttl)
        
        return success
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        success = True
        
        if self.use_memory:
            success &= self.memory_backend.delete(key)
        
        if self.use_disk:
            success &= self.disk_backend.delete(key)
        
        return success
    
    def clear(self) -> bool:
        """清空所有缓存"""
        success = True
        
        if self.use_memory:
            success &= self.memory_backend.clear()
        
        if self.use_disk:
            success &= self.disk_backend.clear()
        
        return success
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        if self.use_memory and self.memory_backend.exists(key):
            return True
        
        if self.use_disk and self.disk_backend.exists(key):
            return True
        
        return False
