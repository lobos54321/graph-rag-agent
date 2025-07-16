"""
深度研究工具

支持多轮推理和搜索的高级搜索工具
"""

from typing import Dict, List, Any, Optional, Union
import time
import logging
import traceback

from langchain_core.tools import BaseTool

from search_new.tools.base_tool import BaseSearchTool
from search_new.tools.hybrid_tool import HybridSearchTool
from search_new.tools.local_tool import LocalSearchTool
from search_new.tools.global_tool import GlobalSearchTool
from search_new.reasoning import (
    ThinkingEngine,
    QueryGenerator,
    EvidenceTracker,
    DualPathSearcher,
    AnswerValidator
)

logger = logging.getLogger(__name__)


class DeepResearchTool(BaseSearchTool):
    """
    深度研究工具：通过多轮推理和搜索解决复杂问题
    
    主要功能：
    1. 多轮思考-搜索-推理循环
    2. 子查询生成和执行
    3. 证据收集和跟踪
    4. 答案验证和优化
    """
    
    def __init__(self):
        """初始化深度研究工具"""
        super().__init__(cache_dir=f"{self.config.cache.base_cache_dir}/deep_research")
        
        # 初始化子工具
        self.hybrid_tool = HybridSearchTool()
        self.local_tool = LocalSearchTool()
        self.global_tool = GlobalSearchTool()
        
        # 初始化推理组件
        self.thinking_engine = ThinkingEngine(self.llm)
        self.query_generator = QueryGenerator(self.llm)
        self.evidence_tracker = EvidenceTracker()
        self.answer_validator = AnswerValidator(self.llm)
        
        # 初始化双路径搜索器
        self.dual_searcher = DualPathSearcher(
            kb_retrieve_func=self._kb_retrieve,
            kg_retrieve_func=self._kg_retrieve,
            kb_name="knowledge_base"
        )
        
        # 深度研究配置
        self.max_iterations = 5
        self.max_sub_queries = 8
        self.evidence_threshold = 0.7
        
        # 执行状态
        self.execution_logs = []
        self.all_retrieved_info = []
        self.current_query_context = {}
        
        logger.info("深度研究工具初始化完成")
    
    def _setup_chains(self):
        """设置处理链"""
        # 深度研究工具的链由子组件管理
        pass
    
    def _kb_retrieve(self, query: str) -> List[str]:
        """知识库检索函数"""
        try:
            result = self.local_tool.search(query)
            return [result] if result else []
        except Exception as e:
            logger.error(f"知识库检索失败: {e}")
            return []
    
    def _kg_retrieve(self, query: str) -> List[str]:
        """知识图谱检索函数"""
        try:
            result = self.global_tool.search(query)
            return [result] if result else []
        except Exception as e:
            logger.error(f"知识图谱检索失败: {e}")
            return []
    
    def _log(self, message: str):
        """记录执行日志"""
        self.execution_logs.append(message)
        logger.debug(message)
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        提取关键词
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 关键词字典
        """
        try:
            return self.hybrid_tool.extract_keywords(query)
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            return {"low_level": [], "high_level": []}
    
    def search(self, query_input: Union[str, Dict[str, Any]]) -> str:
        """
        执行深度研究搜索
        
        参数:
            query_input: 查询输入
            
        返回:
            str: 研究结果
        """
        overall_start = time.time()
        self._reset_metrics()
        
        try:
            # 解析查询
            if isinstance(query_input, dict):
                query = query_input.get("query", str(query_input))
            else:
                query = str(query_input)
            
            # 生成缓存键
            cache_key = self._get_cache_key(query)
            
            # 检查缓存
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.info(f"深度研究缓存命中: {query[:50]}...")
                return cached_result
            
            logger.info(f"开始深度研究: {query[:100]}...")
            
            # 执行深度研究流程
            result = self._execute_deep_research(query)
            
            # 缓存结果
            self._set_to_cache(cache_key, result)
            
            # 记录性能指标
            self.performance_metrics["total_time"] = time.time() - overall_start
            
            logger.info(f"深度研究完成，耗时: {self.performance_metrics['total_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"深度研究失败: {e}")
            self.error_stats["query_errors"] += 1
            self.performance_metrics["total_time"] = time.time() - overall_start
            
            return f"深度研究过程中出现问题: {str(e)}"
    
    def _execute_deep_research(self, query: str) -> str:
        """执行深度研究流程"""
        try:
            # 清空执行状态
            self.execution_logs = []
            self.all_retrieved_info = []
            
            # 创建证据链
            evidence_chain_id = self.evidence_tracker.create_evidence_chain(query)
            
            # 初始化思考引擎
            thinking_session_id = self.thinking_engine.create_session(query)
            
            self._log(f"开始深度研究: {query}")
            
            # 生成初始子查询
            from search_new.reasoning.core.query_generator import QueryContext
            query_context = QueryContext(
                original_query=query,
                current_context="初始查询分析"
            )
            
            initial_sub_queries = self.query_generator.generate_sub_queries(query_context)
            self._log(f"生成了 {len(initial_sub_queries)} 个初始子查询")
            
            # 执行多轮研究
            iteration = 0
            think_content = ""
            
            while iteration < self.max_iterations:
                iteration += 1
                self._log(f"\n=== 第 {iteration} 轮研究 ===")
                
                # 生成下一步查询
                next_query_result = self.thinking_engine.generate_next_query()
                
                if next_query_result["status"] == "answer_ready":
                    self._log("思考引擎认为可以生成答案")
                    break
                elif next_query_result["status"] == "has_query":
                    # 执行查询
                    queries = next_query_result["queries"]
                    self._log(f"执行 {len(queries)} 个查询")
                    
                    for sub_query in queries[:3]:  # 限制查询数量
                        search_results = self._execute_sub_query(sub_query)
                        
                        # 添加证据
                        for result in search_results:
                            self.evidence_tracker.add_evidence(
                                source_id=f"search_{iteration}",
                                content=result,
                                source_type="search_result",
                                relevance_score=0.8,
                                confidence_score=0.8
                            )
                        
                        # 更新思考引擎
                        self.thinking_engine.add_executed_query(
                            sub_query, 
                            "\n".join(search_results)
                        )
                
                # 更新思考内容
                think_content += next_query_result.get("content", "")
                
                # 检查是否有足够证据
                evidence_summary = self.evidence_tracker.get_evidence_chain_summary()
                if evidence_summary.get("evidence_count", 0) >= 5:
                    self._log("收集到足够证据，准备生成答案")
                    break
            
            # 生成最终答案
            final_answer = self._generate_final_answer(query, think_content)
            
            # 验证答案
            validation_result = self.answer_validator.validate_answer(
                query, 
                final_answer,
                [info.get("content", "") for info in self.all_retrieved_info]
            )
            
            if not validation_result.is_valid:
                self._log(f"答案验证失败: {validation_result.issues}")
                # 可以选择重新生成或改进答案
            
            return final_answer
            
        except Exception as e:
            logger.error(f"深度研究执行失败: {e}")
            return f"深度研究过程中出现错误: {str(e)}"
    
    def _execute_sub_query(self, sub_query: str) -> List[str]:
        """执行子查询"""
        try:
            self._log(f"执行子查询: {sub_query}")
            
            # 使用双路径搜索
            results = self.dual_searcher.search(sub_query)
            
            # 记录检索信息
            for result in results:
                self.all_retrieved_info.append({
                    "query": sub_query,
                    "content": result,
                    "timestamp": time.time()
                })
            
            return results
            
        except Exception as e:
            logger.error(f"子查询执行失败: {e}")
            return []
    
    def _generate_final_answer(self, query: str, think_content: str) -> str:
        """生成最终答案"""
        try:
            # 收集所有检索到的信息
            all_content = []
            for info in self.all_retrieved_info:
                all_content.append(info["content"])
            
            retrieved_content = "\n\n".join(all_content)
            
            # 使用混合工具生成答案
            answer_query = f"""
            基于以下思考过程和检索信息，回答问题：{query}

            思考过程：
            {think_content}

            检索信息：
            {retrieved_content}

            请生成一个完整、准确的答案。
            """
            
            final_answer = self.hybrid_tool.search(answer_query)
            
            return final_answer
            
        except Exception as e:
            logger.error(f"最终答案生成失败: {e}")
            return "抱歉，无法生成满意的答案。"
    
    def thinking(self, query: str) -> Dict[str, Any]:
        """
        执行思考过程并返回详细信息
        
        参数:
            query: 查询字符串
            
        返回:
            Dict: 思考过程详情
        """
        try:
            # 执行深度研究
            result = self.search(query)
            
            # 获取思考历史
            thinking_history = self.thinking_engine.get_thinking_history()
            
            # 获取证据链摘要
            evidence_summary = self.evidence_tracker.get_evidence_chain_summary()
            
            # 获取推理轨迹
            reasoning_trace = self.evidence_tracker.get_reasoning_trace()
            
            return {
                "query": query,
                "final_answer": result,
                "thinking_history": thinking_history,
                "evidence_summary": evidence_summary,
                "reasoning_trace": reasoning_trace,
                "execution_logs": self.execution_logs,
                "retrieved_info": self.all_retrieved_info,
                "performance": self.get_performance_metrics()
            }
            
        except Exception as e:
            logger.error(f"思考过程执行失败: {e}")
            return {
                "query": query,
                "error": str(e),
                "execution_logs": self.execution_logs
            }
    
    def get_tool(self) -> BaseTool:
        """获取LangChain兼容的工具"""
        class DeepResearchRetrievalTool(BaseTool):
            name: str = "deep_research"
            description: str = "深度研究工具：通过多轮推理和搜索解决复杂问题，尤其适用于需要深入分析的查询。"
            
            def _run(self_tool, query: Any) -> str:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")
        
        return DeepResearchRetrievalTool()
    
    def get_thinking_tool(self) -> BaseTool:
        """获取思考过程可见的工具版本"""
        class DeepThinkingTool(BaseTool):
            name: str = "deep_thinking"
            description: str = "深度思考工具：显示完整思考过程的深度研究，适用于需要查看推理步骤的情况。"
            
            def _run(self_tool, query: Any) -> Dict:
                # 解析输入
                if isinstance(query, dict) and "query" in query:
                    tk_query = query["query"]
                else:
                    tk_query = str(query)
                
                # 执行思考过程
                return self.thinking(tk_query)
            
            def _arun(self_tool, query: Any) -> Dict:
                raise NotImplementedError("异步执行未实现")
        
        return DeepThinkingTool()
    
    def close(self):
        """关闭深度研究工具"""
        try:
            # 调用父类方法
            super().close()
            
            # 关闭子工具
            if hasattr(self, 'hybrid_tool'):
                self.hybrid_tool.close()
            if hasattr(self, 'local_tool'):
                self.local_tool.close()
            if hasattr(self, 'global_tool'):
                self.global_tool.close()
            
            # 关闭推理组件
            if hasattr(self, 'thinking_engine'):
                self.thinking_engine.close()
            if hasattr(self, 'query_generator'):
                self.query_generator.close()
            if hasattr(self, 'evidence_tracker'):
                self.evidence_tracker.close()
            if hasattr(self, 'answer_validator'):
                self.answer_validator.close()
            if hasattr(self, 'dual_searcher'):
                self.dual_searcher.close()
                
        except Exception as e:
            logger.error(f"深度研究工具关闭失败: {e}")
