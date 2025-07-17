"""
双路径搜索器

支持同时使用多种方式搜索知识库
"""

from typing import List, Dict, Any, Callable
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from search_new.config import get_reasoning_config


class DualPathSearcher:
    """
    双路径搜索器：支持多种搜索策略的并行执行
    
    主要功能：
    1. 知识库搜索
    2. 知识图谱搜索
    3. 并行搜索执行
    4. 结果融合和排序
    """
    
    def __init__(self, kb_retrieve_func: Callable, kg_retrieve_func: Callable, 
                 kb_name: str = "knowledge_base"):
        """
        初始化双路径搜索器
        
        参数:
            kb_retrieve_func: 知识库检索函数
            kg_retrieve_func: 知识图谱检索函数
            kb_name: 知识库名称
        """
        self.kb_retrieve_func = kb_retrieve_func
        self.kg_retrieve_func = kg_retrieve_func
        self.kb_name = kb_name
        
        self.config = get_reasoning_config()
        
        # 搜索配置
        self.enable_parallel = True
        self.max_workers = 2
        self.search_timeout = 30
        
        # 结果权重
        self.kb_weight = 0.6
        self.kg_weight = 0.4
        
        # 搜索历史
        self.search_history: List[Dict[str, Any]] = []
        
        print(f"双路径搜索器初始化完成，知识库: {kb_name}")
    
    def search(self, query: str, search_type: str = "both") -> List[str]:
        """
        执行搜索
        
        参数:
            query: 搜索查询
            search_type: 搜索类型 (kb, kg, both)
            
        返回:
            List[str]: 搜索结果列表
        """
        start_time = time.time()
        
        try:
            print(f"开始双路径搜索: {query[:50]}... (类型: {search_type})")
            
            results = []
            
            if search_type == "kb":
                results = self._search_kb_only(query)
            elif search_type == "kg":
                results = self._search_kg_only(query)
            else:  # both
                if self.enable_parallel:
                    results = self._search_parallel(query)
                else:
                    results = self._search_sequential(query)
            
            # 记录搜索历史
            self._record_search_history(query, search_type, results, time.time() - start_time)
            
            print(f"双路径搜索完成，结果数: {len(results)}")
            return results
            
        except Exception as e:
            print(f"双路径搜索失败: {e}")
            return []
    
    def _search_kb_only(self, query: str) -> List[str]:
        """仅搜索知识库"""
        try:
            return self.kb_retrieve_func(query)
        except Exception as e:
            print(f"知识库搜索失败: {e}")
            return []
    
    def _search_kg_only(self, query: str) -> List[str]:
        """仅搜索知识图谱"""
        try:
            return self.kg_retrieve_func(query)
        except Exception as e:
            print(f"知识图谱搜索失败: {e}")
            return []
    
    def _search_parallel(self, query: str) -> List[str]:
        """并行搜索"""
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交搜索任务
                future_to_type = {
                    executor.submit(self.kb_retrieve_func, query): "kb",
                    executor.submit(self.kg_retrieve_func, query): "kg"
                }
                
                results = {"kb": [], "kg": []}
                
                # 收集结果
                for future in as_completed(future_to_type, timeout=self.search_timeout):
                    search_type = future_to_type[future]
                    try:
                        result = future.result()
                        if result:
                            results[search_type] = result
                    except Exception as e:
                        print(f"{search_type}搜索失败: {e}")
                
                # 融合结果
                return self._merge_results(results["kb"], results["kg"])
                
        except Exception as e:
            print(f"并行搜索失败: {e}")
            # 降级到顺序搜索
            return self._search_sequential(query)
    
    def _search_sequential(self, query: str) -> List[str]:
        """顺序搜索"""
        try:
            kb_results = self._search_kb_only(query)
            kg_results = self._search_kg_only(query)
            
            return self._merge_results(kb_results, kg_results)
            
        except Exception as e:
            print(f"顺序搜索失败: {e}")
            return []
    
    def _merge_results(self, kb_results: List[str], kg_results: List[str]) -> List[str]:
        """
        融合搜索结果
        
        参数:
            kb_results: 知识库结果
            kg_results: 知识图谱结果
            
        返回:
            List[str]: 融合后的结果
        """
        try:
            merged_results = []
            
            # 加权融合
            kb_weighted = [(result, self.kb_weight) for result in kb_results]
            kg_weighted = [(result, self.kg_weight) for result in kg_results]
            
            # 合并并按权重排序
            all_results = kb_weighted + kg_weighted
            
            # 去重并保持顺序
            seen = set()
            for result, weight in all_results:
                if result not in seen:
                    merged_results.append(result)
                    seen.add(result)
            
            return merged_results
            
        except Exception as e:
            print(f"结果融合失败: {e}")
            # 简单合并
            return kb_results + kg_results
    
    def search_with_context(self, query: str, context: str, 
                           search_type: str = "both") -> List[str]:
        """
        带上下文的搜索
        
        参数:
            query: 搜索查询
            context: 上下文信息
            search_type: 搜索类型
            
        返回:
            List[str]: 搜索结果
        """
        try:
            # 构建增强查询
            enhanced_query = f"{query}\n上下文: {context}"
            
            return self.search(enhanced_query, search_type)
            
        except Exception as e:
            print(f"上下文搜索失败: {e}")
            return self.search(query, search_type)
    
    def batch_search(self, queries: List[str], search_type: str = "both") -> Dict[str, List[str]]:
        """
        批量搜索
        
        参数:
            queries: 查询列表
            search_type: 搜索类型
            
        返回:
            Dict[str, List[str]]: 查询到结果的映射
        """
        try:
            results = {}
            
            if self.enable_parallel:
                # 并行批量搜索
                with ThreadPoolExecutor(max_workers=min(len(queries), self.max_workers)) as executor:
                    future_to_query = {
                        executor.submit(self.search, query, search_type): query
                        for query in queries
                    }
                    
                    for future in as_completed(future_to_query, timeout=self.search_timeout * len(queries)):
                        query = future_to_query[future]
                        try:
                            result = future.result()
                            results[query] = result
                        except Exception as e:
                            print(f"批量搜索查询失败 '{query}': {e}")
                            results[query] = []
            else:
                # 顺序批量搜索
                for query in queries:
                    results[query] = self.search(query, search_type)
            
            return results
            
        except Exception as e:
            print(f"批量搜索失败: {e}")
            return {query: [] for query in queries}
    
    def search_with_filters(self, query: str, filters: Dict[str, Any], 
                           search_type: str = "both") -> List[str]:
        """
        带过滤条件的搜索
        
        参数:
            query: 搜索查询
            filters: 过滤条件
            search_type: 搜索类型
            
        返回:
            List[str]: 过滤后的搜索结果
        """
        try:
            # 执行基础搜索
            results = self.search(query, search_type)
            
            # 应用过滤条件
            filtered_results = self._apply_filters(results, filters)
            
            return filtered_results
            
        except Exception as e:
            print(f"过滤搜索失败: {e}")
            return []
    
    def _apply_filters(self, results: List[str], filters: Dict[str, Any]) -> List[str]:
        """
        应用过滤条件
        
        参数:
            results: 原始结果
            filters: 过滤条件
            
        返回:
            List[str]: 过滤后的结果
        """
        try:
            filtered_results = []
            
            for result in results:
                if self._match_filters(result, filters):
                    filtered_results.append(result)
            
            return filtered_results
            
        except Exception as e:
            print(f"应用过滤条件失败: {e}")
            return results
    
    def _match_filters(self, result: str, filters: Dict[str, Any]) -> bool:
        """
        检查结果是否匹配过滤条件
        
        参数:
            result: 搜索结果
            filters: 过滤条件
            
        返回:
            bool: 是否匹配
        """
        try:
            # 长度过滤
            if "min_length" in filters and len(result) < filters["min_length"]:
                return False
            
            if "max_length" in filters and len(result) > filters["max_length"]:
                return False
            
            # 关键词过滤
            if "required_keywords" in filters:
                for keyword in filters["required_keywords"]:
                    if keyword.lower() not in result.lower():
                        return False
            
            if "excluded_keywords" in filters:
                for keyword in filters["excluded_keywords"]:
                    if keyword.lower() in result.lower():
                        return False
            
            return True
            
        except Exception as e:
            print(f"过滤条件匹配失败: {e}")
            return True
    
    def _record_search_history(self, query: str, search_type: str, 
                              results: List[str], duration: float):
        """记录搜索历史"""
        try:
            history_entry = {
                "query": query,
                "search_type": search_type,
                "results_count": len(results),
                "duration": duration,
                "timestamp": time.time()
            }
            
            self.search_history.append(history_entry)
            
            # 限制历史记录数量
            if len(self.search_history) > 100:
                self.search_history = self.search_history[-100:]
                
        except Exception as e:
            print(f"记录搜索历史失败: {e}")
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """
        获取搜索统计信息
        
        返回:
            Dict[str, Any]: 统计信息
        """
        try:
            if not self.search_history:
                return {"total_searches": 0}
            
            total_searches = len(self.search_history)
            total_duration = sum(entry["duration"] for entry in self.search_history)
            avg_duration = total_duration / total_searches
            
            search_types = {}
            for entry in self.search_history:
                search_type = entry["search_type"]
                search_types[search_type] = search_types.get(search_type, 0) + 1
            
            return {
                "total_searches": total_searches,
                "avg_duration": avg_duration,
                "total_duration": total_duration,
                "search_types": search_types,
                "kb_name": self.kb_name
            }
            
        except Exception as e:
            print(f"获取搜索统计失败: {e}")
            return {"error": str(e)}
    
    def clear_history(self):
        """清空搜索历史"""
        self.search_history.clear()
        print("搜索历史已清空")
    
    def close(self):
        """关闭搜索器"""
        self.clear_history()
        print("双路径搜索器已关闭")
