"""
性能监控模块

提供搜索性能监控和统计功能
"""

import time
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """性能指标数据类"""
    name: str
    value: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchStats:
    """搜索统计数据类"""
    total_searches: int = 0
    successful_searches: int = 0
    failed_searches: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # 时间统计
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    
    # 错误统计
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, max_history_size: int = 1000):
        """
        初始化性能监控器
        
        参数:
            max_history_size: 最大历史记录数量
        """
        self.max_history_size = max_history_size
        self._lock = threading.RLock()
        
        # 性能指标历史
        self._metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        
        # 搜索统计
        self._search_stats = SearchStats()
        
        # 实时监控数据
        self._current_metrics: Dict[str, float] = {}
        
        # 监控开关
        self._monitoring_enabled = True
        
        logger.info("性能监控器初始化完成")
    
    def enable_monitoring(self):
        """启用监控"""
        with self._lock:
            self._monitoring_enabled = True
            logger.info("性能监控已启用")
    
    def disable_monitoring(self):
        """禁用监控"""
        with self._lock:
            self._monitoring_enabled = False
            logger.info("性能监控已禁用")
    
    def record_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """
        记录性能指标
        
        参数:
            name: 指标名称
            value: 指标值
            metadata: 元数据
        """
        if not self._monitoring_enabled:
            return
        
        with self._lock:
            try:
                metric = PerformanceMetric(
                    name=name,
                    value=value,
                    timestamp=time.time(),
                    metadata=metadata or {}
                )
                
                # 添加到历史记录
                self._metrics_history[name].append(metric)
                
                # 更新当前指标
                self._current_metrics[name] = value
                
            except Exception as e:
                logger.error(f"记录性能指标失败: {e}")
    
    def record_search_start(self, query: str, tool_name: str) -> str:
        """
        记录搜索开始
        
        参数:
            query: 搜索查询
            tool_name: 工具名称
            
        返回:
            str: 搜索会话ID
        """
        if not self._monitoring_enabled:
            return ""
        
        session_id = f"{tool_name}_{int(time.time() * 1000)}"
        
        with self._lock:
            self._search_stats.total_searches += 1
            
            # 记录搜索开始指标
            self.record_metric(
                f"search_start_{tool_name}",
                time.time(),
                {
                    "session_id": session_id,
                    "query": query[:100],  # 只记录前100个字符
                    "tool_name": tool_name
                }
            )
        
        return session_id
    
    def record_search_end(self, session_id: str, success: bool, duration: float, 
                         error: Optional[str] = None):
        """
        记录搜索结束
        
        参数:
            session_id: 搜索会话ID
            success: 是否成功
            duration: 持续时间
            error: 错误信息
        """
        if not self._monitoring_enabled:
            return
        
        with self._lock:
            try:
                # 更新搜索统计
                if success:
                    self._search_stats.successful_searches += 1
                else:
                    self._search_stats.failed_searches += 1
                    if error:
                        self._search_stats.error_counts[error] += 1
                
                # 更新时间统计
                self._search_stats.total_time += duration
                self._search_stats.avg_time = (
                    self._search_stats.total_time / self._search_stats.total_searches
                )
                self._search_stats.min_time = min(self._search_stats.min_time, duration)
                self._search_stats.max_time = max(self._search_stats.max_time, duration)
                
                # 记录搜索结束指标
                self.record_metric(
                    "search_duration",
                    duration,
                    {
                        "session_id": session_id,
                        "success": success,
                        "error": error
                    }
                )
                
            except Exception as e:
                logger.error(f"记录搜索结束失败: {e}")
    
    def record_cache_hit(self, tool_name: str):
        """记录缓存命中"""
        if not self._monitoring_enabled:
            return
        
        with self._lock:
            self._search_stats.cache_hits += 1
            self.record_metric(f"cache_hit_{tool_name}", 1)
    
    def record_cache_miss(self, tool_name: str):
        """记录缓存未命中"""
        if not self._monitoring_enabled:
            return
        
        with self._lock:
            self._search_stats.cache_misses += 1
            self.record_metric(f"cache_miss_{tool_name}", 1)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """获取当前指标"""
        with self._lock:
            return self._current_metrics.copy()
    
    def get_search_stats(self) -> SearchStats:
        """获取搜索统计"""
        with self._lock:
            return SearchStats(
                total_searches=self._search_stats.total_searches,
                successful_searches=self._search_stats.successful_searches,
                failed_searches=self._search_stats.failed_searches,
                cache_hits=self._search_stats.cache_hits,
                cache_misses=self._search_stats.cache_misses,
                total_time=self._search_stats.total_time,
                avg_time=self._search_stats.avg_time,
                min_time=self._search_stats.min_time,
                max_time=self._search_stats.max_time,
                error_counts=dict(self._search_stats.error_counts)
            )
    
    def get_metric_history(self, metric_name: str, limit: Optional[int] = None) -> List[PerformanceMetric]:
        """
        获取指标历史
        
        参数:
            metric_name: 指标名称
            limit: 限制返回数量
            
        返回:
            List[PerformanceMetric]: 指标历史列表
        """
        with self._lock:
            history = list(self._metrics_history.get(metric_name, []))
            if limit:
                history = history[-limit:]
            return history
    
    def get_metric_statistics(self, metric_name: str) -> Dict[str, float]:
        """
        获取指标统计信息
        
        参数:
            metric_name: 指标名称
            
        返回:
            Dict[str, float]: 统计信息
        """
        with self._lock:
            history = self._metrics_history.get(metric_name, [])
            if not history:
                return {}
            
            values = [metric.value for metric in history]
            
            try:
                return {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0.0
                }
            except Exception as e:
                logger.error(f"计算统计信息失败: {e}")
                return {}
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        获取性能报告
        
        返回:
            Dict[str, Any]: 性能报告
        """
        with self._lock:
            try:
                search_stats = self.get_search_stats()
                
                # 计算成功率
                success_rate = 0.0
                if search_stats.total_searches > 0:
                    success_rate = search_stats.successful_searches / search_stats.total_searches
                
                # 计算缓存命中率
                cache_hit_rate = 0.0
                total_cache_requests = search_stats.cache_hits + search_stats.cache_misses
                if total_cache_requests > 0:
                    cache_hit_rate = search_stats.cache_hits / total_cache_requests
                
                # 获取关键指标统计
                duration_stats = self.get_metric_statistics("search_duration")
                
                return {
                    "search_statistics": {
                        "total_searches": search_stats.total_searches,
                        "successful_searches": search_stats.successful_searches,
                        "failed_searches": search_stats.failed_searches,
                        "success_rate": success_rate
                    },
                    "cache_statistics": {
                        "cache_hits": search_stats.cache_hits,
                        "cache_misses": search_stats.cache_misses,
                        "cache_hit_rate": cache_hit_rate
                    },
                    "performance_statistics": {
                        "avg_duration": search_stats.avg_time,
                        "min_duration": search_stats.min_time if search_stats.min_time != float('inf') else 0,
                        "max_duration": search_stats.max_time,
                        "duration_stats": duration_stats
                    },
                    "error_statistics": dict(search_stats.error_counts),
                    "current_metrics": self.get_current_metrics(),
                    "report_timestamp": time.time()
                }
                
            except Exception as e:
                logger.error(f"生成性能报告失败: {e}")
                return {"error": str(e)}
    
    def reset_statistics(self):
        """重置统计信息"""
        with self._lock:
            self._search_stats = SearchStats()
            self._metrics_history.clear()
            self._current_metrics.clear()
            logger.info("性能统计信息已重置")
    
    def export_metrics(self, metric_names: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """
        导出指标数据
        
        参数:
            metric_names: 要导出的指标名称列表，None表示导出所有指标
            
        返回:
            Dict[str, List[Dict]]: 导出的指标数据
        """
        with self._lock:
            exported_data = {}
            
            metrics_to_export = metric_names or list(self._metrics_history.keys())
            
            for metric_name in metrics_to_export:
                if metric_name in self._metrics_history:
                    history = self._metrics_history[metric_name]
                    exported_data[metric_name] = [
                        {
                            "value": metric.value,
                            "timestamp": metric.timestamp,
                            "metadata": metric.metadata
                        }
                        for metric in history
                    ]
            
            return exported_data


# 全局性能监控器实例
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器实例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def reset_performance_monitor():
    """重置全局性能监控器"""
    global _global_monitor
    _global_monitor = None
