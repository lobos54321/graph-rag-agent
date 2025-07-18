"""
全局搜索工具

基于Map-Reduce模式的跨社区查询工具
"""

from typing import List, Dict, Any, Union
import time
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from search_new.tools.base_tool import BaseSearchTool
from search_new.core.global_search import GlobalSearch


class GlobalSearchTool(BaseSearchTool):
    """全局搜索工具，基于Map-Reduce模式实现跨社区查询"""
    
    def __init__(self):
        """初始化全局搜索工具"""
        # 先初始化基类以获取config
        super().__init__()
        # 然后设置特定的缓存目录
        if hasattr(self, 'cache_manager') and self.cache_manager:
            self.cache_manager.cache_dir = self.config.cache.global_search_cache_dir
        
        # 创建全局搜索器
        self.global_searcher = GlobalSearch(self.llm)
        
        # 默认搜索层级
        self.default_level = self.config.global_search.default_level
        
        print(f"全局搜索工具初始化完成，默认层级: {self.default_level}")

    def _setup_chains(self):
        """设置处理链"""
        try:
            # 设置关键词提取链
            self._setup_keyword_chain()

        except Exception as e:
            print(f"处理链设置失败: {e}")
            raise
    
    def _setup_keyword_chain(self):
        """设置关键词提取链"""
        keyword_prompt = ChatPromptTemplate.from_template("""
        请从以下查询中提取关键词，分为低级关键词（具体实体、人名、地名等）和高级关键词（概念、主题等）。
        全局搜索更关注高级概念和主题。

        查询: {query}

        请以JSON格式返回结果：
        {{
            "low_level": ["关键词1", "关键词2"],
            "high_level": ["概念1", "概念2"]
        }}
        """)
        
        self.keyword_chain = keyword_prompt | self.llm | StrOutputParser()
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 分类关键词字典
        """
        # 检查缓存
        cache_key = f"global_keywords:{query}"
        cached_keywords = self._get_from_cache(cache_key)
        if cached_keywords:
            return cached_keywords
        
        try:
            start_time = time.time()
            
            # 调用LLM提取关键词
            result = self.keyword_chain.invoke({"query": query})
            
            # 记录处理时间
            self.performance_metrics["keyword_time"] += time.time() - start_time
            
            # 解析JSON结果
            try:
                keywords = json.loads(result)
            except json.JSONDecodeError:
                print(f"关键词提取结果解析失败: {result}")
                keywords = {"low_level": [], "high_level": []}

            # 确保包含必要的键
            if not isinstance(keywords, dict):
                keywords = {}
            if "low_level" not in keywords:
                keywords["low_level"] = []
            if "high_level" not in keywords:
                keywords["high_level"] = []

            # 缓存结果
            self._set_to_cache(cache_key, keywords)

            return keywords

        except Exception as e:
            print(f"关键词提取失败: {e}")
            self.error_stats["keyword_errors"] += 1
            return {"low_level": [], "high_level": []}
    
    def _parse_query_input(self, query_input: Union[str, Dict[str, Any]]) -> tuple:
        """
        解析查询输入
        
        参数:
            query_input: 查询输入
            
        返回:
            tuple: (query, level, keywords)
        """
        if isinstance(query_input, dict):
            query = query_input.get("query", str(query_input))
            level = query_input.get("level", self.default_level)
            keywords = query_input.get("keywords", [])
        else:
            query = str(query_input)
            level = self.default_level
            keywords = []
        
        return query, level, keywords
    
    def search(self, query_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行全局搜索

        参数:
            query_input: 查询输入，可以是字符串或字典

        返回:
            Dict[str, Any]: 包含answer和sources的搜索结果
        """
        overall_start = time.time()
        self._reset_metrics()
        
        try:
            # 解析输入
            query, level, keywords = self._parse_query_input(query_input)
            
            # 生成缓存键
            cache_key = self._get_cache_key(query, level=level, keywords=keywords)
            
            # 检查缓存
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                print(f"全局搜索缓存命中: {query[:50]}...")
                # 如果缓存的是字符串，转换为字典格式
                if isinstance(cached_result, str):
                    return {
                        "answer": cached_result,
                        "sources": [],
                        "metadata": {"from_cache": True}
                    }
                return cached_result

            print(f"开始全局搜索: {query[:100]}..., 层级: {level}")

            # 使用全局搜索器执行搜索
            search_start = time.time()
            result = self.global_searcher.search(query, level)
            self.performance_metrics["query_time"] = time.time() - search_start

            # 构建返回结果
            if not result or result.strip() == "":
                search_result = {
                    "answer": "未找到相关信息",
                    "sources": [],
                    "metadata": {
                        "level": level,
                        "query_time": self.performance_metrics["query_time"]
                    }
                }
            else:
                search_result = {
                    "answer": result,
                    "sources": [],  # 全局搜索通常基于社区摘要，sources为空
                    "metadata": {
                        "level": level,
                        "query_time": self.performance_metrics["query_time"]
                    }
                }

            # 缓存结果
            self._set_to_cache(cache_key, search_result)

            # 记录性能指标
            self.performance_metrics["total_time"] = time.time() - overall_start

            print(f"全局搜索完成，耗时: {self.performance_metrics['total_time']:.2f}s")
            return search_result

        except Exception as e:
            print(f"全局搜索失败: {e}")
            self.error_stats["query_errors"] += 1
            self.performance_metrics["total_time"] = time.time() - overall_start

            return {
                "answer": f"搜索过程中出现问题: {str(e)}",
                "sources": [],
                "metadata": {"error": True}
            }
    
    def search_with_details(self, query_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行全局搜索并返回详细信息
        
        参数:
            query_input: 查询输入
            
        返回:
            Dict: 包含搜索结果和详细信息的字典
        """
        start_time = time.time()
        
        try:
            # 解析输入
            query, level, keywords = self._parse_query_input(query_input)
            
            # 执行搜索
            result = self.search(query_input)
            
            # 提取关键词
            extracted_keywords = self.extract_keywords(query)
            
            # 获取详细搜索信息
            detailed_result = self.global_searcher.search_with_details(query, level)
            
            return {
                "result": result,
                "query": query,
                "level": level,
                "keywords": keywords,
                "extracted_keywords": extracted_keywords,
                "communities_count": detailed_result.get("communities_count", 0),
                "performance": self.get_performance_metrics(),
                "error_stats": self.get_error_stats(),
                "total_time": time.time() - start_time,
                "tool_name": "GlobalSearchTool"
            }
            
        except Exception as e:
            print(f"详细全局搜索失败: {e}")
            return {
                "result": f"搜索失败: {str(e)}",
                "query": str(query_input),
                "error": str(e),
                "performance": self.get_performance_metrics(),
                "error_stats": self.get_error_stats(),
                "total_time": time.time() - start_time,
                "tool_name": "GlobalSearchTool"
            }
    
    def search_multiple_levels(self, query: str, levels: List[int]) -> Dict[int, str]:
        """
        在多个层级执行搜索
        
        参数:
            query: 查询字符串
            levels: 层级列表
            
        返回:
            Dict[int, str]: 每个层级的搜索结果
        """
        results = {}
        
        for level in levels:
            try:
                print(f"在层级 {level} 执行搜索")
                result = self.search({"query": query, "level": level})
                results[level] = result

            except Exception as e:
                print(f"层级 {level} 搜索失败: {e}")
                results[level] = f"搜索失败: {str(e)}"
        
        return results
    
    def get_community_summary(self, level: int) -> Dict[str, Any]:
        """
        获取指定层级的社区摘要信息
        
        参数:
            level: 社区层级
            
        返回:
            Dict: 社区摘要信息
        """
        try:
            # 使用全局搜索器获取社区数据
            communities = self.global_searcher._get_community_data(level)
            
            return {
                "level": level,
                "communities_count": len(communities),
                "communities": [
                    {
                        "id": community.get("communityId", ""),
                        "content_length": len(community.get("full_content", ""))
                    } for community in communities[:10]  # 只返回前10个社区的摘要
                ]
            }
            
        except Exception as e:
            print(f"获取社区摘要失败: {e}")
            return {
                "level": level,
                "communities_count": 0,
                "communities": [],
                "error": str(e)
            }
    
    def close(self):
        """关闭资源"""
        try:
            # 调用父类方法
            super().close()
            
            # 关闭全局搜索器
            if hasattr(self, 'global_searcher'):
                self.global_searcher.close()
                
        except Exception as e:
            print(f"全局搜索工具关闭失败: {e}")
