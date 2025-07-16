"""
混合搜索工具

结合本地搜索和全局搜索的混合策略，参考lightrag的双级检索思想
"""

from typing import List, Dict, Any, Union, Optional
import time
import json
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from search_new.tools.base_tool import BaseSearchTool
from search_new.tools.local_tool import LocalSearchTool
from search_new.tools.global_tool import GlobalSearchTool

logger = logging.getLogger(__name__)


class HybridSearchTool(BaseSearchTool):
    """混合搜索工具，结合本地搜索和全局搜索的优势"""
    
    def __init__(self):
        """初始化混合搜索工具"""
        super().__init__(cache_dir=f"{self.config.cache.base_cache_dir}/hybrid_search")
        
        # 创建子搜索工具
        self.local_tool = LocalSearchTool()
        self.global_tool = GlobalSearchTool()
        
        # 混合搜索配置
        self.local_weight = 0.6  # 本地搜索权重
        self.global_weight = 0.4  # 全局搜索权重
        self.enable_parallel = True  # 是否并行执行搜索
        
        logger.info("混合搜索工具初始化完成")
    
    def _setup_chains(self):
        """设置处理链"""
        try:
            # 设置关键词提取链
            self._setup_keyword_chain()
            
            # 设置结果融合链
            self._setup_fusion_chain()
            
            # 设置查询分类链
            self._setup_classification_chain()
            
        except Exception as e:
            logger.error(f"处理链设置失败: {e}")
            raise
    
    def _setup_keyword_chain(self):
        """设置关键词提取链"""
        keyword_prompt = ChatPromptTemplate.from_template("""
        请从以下查询中提取关键词，分为低级关键词（具体实体、人名、地名等）和高级关键词（概念、主题等）。
        混合搜索需要同时考虑具体实体和抽象概念。

        查询: {query}

        请以JSON格式返回结果：
        {{
            "low_level": ["关键词1", "关键词2"],
            "high_level": ["概念1", "概念2"]
        }}
        """)
        
        self.keyword_chain = keyword_prompt | self.llm | StrOutputParser()
    
    def _setup_fusion_chain(self):
        """设置结果融合链"""
        fusion_prompt = ChatPromptTemplate.from_template("""
        请将以下本地搜索和全局搜索的结果融合成一个完整、连贯的答案。

        用户问题: {query}

        本地搜索结果（侧重具体细节）:
        {local_result}

        全局搜索结果（侧重整体概念）:
        {global_result}

        请融合这两个结果，生成一个既包含具体细节又有整体视角的完整答案。
        请按以下格式输出：
        1. 使用三级标题(###)标记主题
        2. 主要内容用清晰的段落展示
        3. 最后必须用"#### 引用数据"标记引用部分，列出用到的数据来源
        """)
        
        self.fusion_chain = fusion_prompt | self.llm | StrOutputParser()
    
    def _setup_classification_chain(self):
        """设置查询分类链"""
        classification_prompt = ChatPromptTemplate.from_template("""
        请分析以下查询的类型，判断更适合使用本地搜索还是全局搜索，或者两者结合。

        查询: {query}

        请以JSON格式返回结果：
        {{
            "search_strategy": "local|global|hybrid",
            "confidence": 0.8,
            "reasoning": "选择理由"
        }}

        判断标准：
        - local: 查询涉及具体实体、详细信息、特定事实
        - global: 查询涉及概念性问题、需要整体视角、跨领域分析
        - hybrid: 查询既需要具体细节又需要整体理解
        """)
        
        self.classification_chain = classification_prompt | self.llm | StrOutputParser()
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 分类关键词字典
        """
        # 检查缓存
        cache_key = f"hybrid_keywords:{query}"
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
                logger.warning(f"关键词提取结果解析失败: {result}")
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
            logger.error(f"关键词提取失败: {e}")
            self.error_stats["keyword_errors"] += 1
            return {"low_level": [], "high_level": []}
    
    def _classify_query(self, query: str) -> Dict[str, Any]:
        """
        分类查询类型
        
        参数:
            query: 查询字符串
            
        返回:
            Dict: 分类结果
        """
        try:
            result = self.classification_chain.invoke({"query": query})
            
            try:
                classification = json.loads(result)
            except json.JSONDecodeError:
                logger.warning(f"查询分类结果解析失败: {result}")
                classification = {
                    "search_strategy": "hybrid",
                    "confidence": 0.5,
                    "reasoning": "解析失败，使用默认混合策略"
                }
            
            return classification
            
        except Exception as e:
            logger.error(f"查询分类失败: {e}")
            return {
                "search_strategy": "hybrid",
                "confidence": 0.5,
                "reasoning": f"分类失败: {str(e)}"
            }
    
    def _execute_local_search(self, query: str) -> str:
        """执行本地搜索"""
        try:
            return self.local_tool.search(query)
        except Exception as e:
            logger.error(f"本地搜索执行失败: {e}")
            return f"本地搜索失败: {str(e)}"
    
    def _execute_global_search(self, query: str) -> str:
        """执行全局搜索"""
        try:
            return self.global_tool.search(query)
        except Exception as e:
            logger.error(f"全局搜索执行失败: {e}")
            return f"全局搜索失败: {str(e)}"
    
    def _fuse_results(self, query: str, local_result: str, global_result: str) -> str:
        """
        融合搜索结果
        
        参数:
            query: 用户查询
            local_result: 本地搜索结果
            global_result: 全局搜索结果
            
        返回:
            str: 融合后的结果
        """
        try:
            # 检查结果有效性
            if not local_result or "失败" in local_result:
                return global_result
            if not global_result or "失败" in global_result:
                return local_result
            
            # 使用融合链生成最终结果
            fused_result = self.fusion_chain.invoke({
                "query": query,
                "local_result": local_result,
                "global_result": global_result
            })
            
            return fused_result
            
        except Exception as e:
            logger.error(f"结果融合失败: {e}")
            # 如果融合失败，返回较长的结果
            if len(local_result) > len(global_result):
                return local_result
            else:
                return global_result
    
    def search(self, query_input: Union[str, Dict[str, Any]]) -> str:
        """
        执行混合搜索
        
        参数:
            query_input: 查询输入，可以是字符串或字典
            
        返回:
            str: 搜索结果
        """
        overall_start = time.time()
        self._reset_metrics()
        
        try:
            # 解析查询
            if isinstance(query_input, dict):
                query = query_input.get("query", str(query_input))
                strategy = query_input.get("strategy", None)
            else:
                query = str(query_input)
                strategy = None
            
            # 生成缓存键
            cache_key = self._get_cache_key(query, strategy=strategy)
            
            # 检查缓存
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.info(f"混合搜索缓存命中: {query[:50]}...")
                return cached_result
            
            logger.info(f"开始混合搜索: {query[:100]}...")
            
            # 如果没有指定策略，则自动分类
            if strategy is None:
                classification = self._classify_query(query)
                strategy = classification["search_strategy"]
                logger.info(f"查询分类结果: {strategy} (置信度: {classification['confidence']})")
            
            # 根据策略执行搜索
            search_start = time.time()
            
            if strategy == "local":
                result = self._execute_local_search(query)
            elif strategy == "global":
                result = self._execute_global_search(query)
            else:  # hybrid
                # 并行执行本地和全局搜索
                local_result = self._execute_local_search(query)
                global_result = self._execute_global_search(query)
                
                # 融合结果
                result = self._fuse_results(query, local_result, global_result)
            
            self.performance_metrics["query_time"] = time.time() - search_start
            
            # 缓存结果
            self._set_to_cache(cache_key, result)
            
            # 记录性能指标
            self.performance_metrics["total_time"] = time.time() - overall_start
            
            if not result or result.strip() == "":
                return "未找到相关信息"
            
            logger.info(f"混合搜索完成，耗时: {self.performance_metrics['total_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            self.error_stats["query_errors"] += 1
            self.performance_metrics["total_time"] = time.time() - overall_start
            
            return f"搜索过程中出现问题: {str(e)}"
    
    def search_with_details(self, query_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行混合搜索并返回详细信息
        
        参数:
            query_input: 查询输入
            
        返回:
            Dict: 包含搜索结果和详细信息的字典
        """
        start_time = time.time()
        
        try:
            # 解析查询
            if isinstance(query_input, dict):
                query = query_input.get("query", str(query_input))
            else:
                query = str(query_input)
            
            # 执行搜索
            result = self.search(query_input)
            
            # 获取分类信息
            classification = self._classify_query(query)
            
            # 提取关键词
            keywords = self.extract_keywords(query)
            
            return {
                "result": result,
                "query": query,
                "classification": classification,
                "keywords": keywords,
                "local_weight": self.local_weight,
                "global_weight": self.global_weight,
                "performance": self.get_performance_metrics(),
                "error_stats": self.get_error_stats(),
                "total_time": time.time() - start_time,
                "tool_name": "HybridSearchTool"
            }
            
        except Exception as e:
            logger.error(f"详细混合搜索失败: {e}")
            return {
                "result": f"搜索失败: {str(e)}",
                "query": str(query_input),
                "error": str(e),
                "performance": self.get_performance_metrics(),
                "error_stats": self.get_error_stats(),
                "total_time": time.time() - start_time,
                "tool_name": "HybridSearchTool"
            }
    
    def close(self):
        """关闭资源"""
        try:
            # 调用父类方法
            super().close()
            
            # 关闭子工具
            if hasattr(self, 'local_tool'):
                self.local_tool.close()
            if hasattr(self, 'global_tool'):
                self.global_tool.close()
                
        except Exception as e:
            logger.error(f"混合搜索工具关闭失败: {e}")
