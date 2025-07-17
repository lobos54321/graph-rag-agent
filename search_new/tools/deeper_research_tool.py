"""
增强深度研究工具

在深度研究基础上添加社区感知和知识图谱功能
"""

from typing import Dict, List, Any, Union, AsyncGenerator
import time
import asyncio

from langchain_core.tools import BaseTool

from search_new.tools.deep_research_tool import DeepResearchTool
from search_new.reasoning import (
    ChainedExploration,
    ComplexityEstimator,
    KnowledgeGraphBuilder
)


class DeeperResearchTool(DeepResearchTool):
    """
    增强深度研究工具：添加社区感知和知识图谱分析
    
    主要功能：
    1. 继承深度研究的所有功能
    2. 社区感知搜索增强
    3. 动态知识图谱构建
    4. 链式探索搜索
    5. 复杂度自适应策略
    """
    
    def __init__(self):
        """初始化增强深度研究工具"""
        super().__init__()
        
        # 初始化增强组件
        self.complexity_estimator = ComplexityEstimator()
        self.knowledge_builder = KnowledgeGraphBuilder(self.llm)
        self.chain_explorer = ChainedExploration(
            graph_query_func=self._graph_query,
            max_steps=self.config.exploration.max_exploration_steps
        )
        
        # 增强配置
        self.enable_community_aware = True
        self.enable_kg_building = True
        self.enable_chain_exploration = True
        self.complexity_threshold = 0.7
        
        # 社区感知缓存
        self._community_cache = {}
        self._kg_cache = {}
        
        print("增强深度研究工具初始化完成")
    
    def _graph_query(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """图数据库查询函数"""
        try:
            # 这里应该连接到实际的图数据库
            # 为了演示，返回模拟结果
            return []
        except Exception as e:
            print(f"图查询失败: {e}")
            return []
    
    def search(self, query_input: Union[str, Dict[str, Any]]) -> str:
        """
        执行增强深度研究搜索
        
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
            cache_key = self._get_cache_key(query, enhanced=True)
            
            # 检查缓存
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                print(f"增强深度研究缓存命中: {query[:50]}...")
                return cached_result
            
            print(f"开始增强深度研究: {query[:100]}...")
            
            # 评估查询复杂度
            complexity = self.complexity_estimator.estimate_complexity(query)
            self._log(f"查询复杂度: {complexity.complexity_level.value} ({complexity.overall_complexity:.2f})")
            
            # 根据复杂度选择策略
            if complexity.overall_complexity >= self.complexity_threshold:
                result = self._execute_enhanced_research(query, complexity)
            else:
                # 对于简单查询，使用基础深度研究
                result = super().search(query)
            
            # 缓存结果
            self._set_to_cache(cache_key, result)
            
            # 记录性能指标
            self.performance_metrics["total_time"] = time.time() - overall_start
            
            print(f"增强深度研究完成，耗时: {self.performance_metrics['total_time']:.2f}s")
            return result
            
        except Exception as e:
            print(f"增强深度研究失败: {e}")
            self.error_stats["query_errors"] += 1
            self.performance_metrics["total_time"] = time.time() - overall_start
            
            return f"增强深度研究过程中出现问题: {str(e)}"
    
    def _execute_enhanced_research(self, query: str, complexity) -> str:
        """执行增强研究流程"""
        try:
            self._log("启动增强研究模式")
            
            # 清空执行状态
            self.execution_logs = []
            self.all_retrieved_info = []
            
            # 创建证据链和思考会话
            evidence_chain_id = self.evidence_tracker.create_evidence_chain(query)
            thinking_session_id = self.thinking_engine.create_session(query)
            
            # 提取关键词
            keywords = self.extract_keywords(query)
            
            # 社区感知搜索增强
            if self.enable_community_aware:
                community_context = self._enhance_search_with_community(query, keywords)
                self._log(f"社区感知增强完成")
            
            # 动态知识图谱构建
            if self.enable_kg_building:
                kg_graph_id = self._build_dynamic_knowledge_graph(query)
                self._log(f"动态知识图谱构建完成: {kg_graph_id}")
            
            # 链式探索搜索
            if self.enable_chain_exploration and keywords.get("high_level"):
                exploration_path_id = self._execute_chain_exploration(query, keywords)
                self._log(f"链式探索完成: {exploration_path_id}")
            
            # 执行基础深度研究流程
            base_result = self._execute_deep_research(query)
            
            # 增强答案生成
            enhanced_result = self._enhance_final_answer(query, base_result, complexity)
            
            return enhanced_result
            
        except Exception as e:
            print(f"增强研究执行失败: {e}")
            # 降级到基础深度研究
            return super()._execute_deep_research(query)
    
    def _enhance_search_with_community(self, query: str, keywords: Dict[str, List[str]]) -> Dict[str, Any]:
        """社区感知搜索增强"""
        try:
            cache_key = f"community:{query}"
            if cache_key in self._community_cache:
                return self._community_cache[cache_key]
            
            # 模拟社区感知分析
            community_context = {
                "search_strategy": {
                    "focus_entities": keywords.get("high_level", [])[:3],
                    "follow_up_queries": [
                        f"详细解释{entity}" for entity in keywords.get("high_level", [])[:2]
                    ],
                    "community_relevance": 0.8
                },
                "enhanced_keywords": keywords,
                "community_insights": f"基于社区分析，查询涉及{len(keywords.get('high_level', []))}个高级概念"
            }
            
            self._community_cache[cache_key] = community_context
            return community_context
            
        except Exception as e:
            print(f"社区感知增强失败: {e}")
            return {"search_strategy": {}, "enhanced_keywords": keywords}
    
    def _build_dynamic_knowledge_graph(self, query: str) -> str:
        """构建动态知识图谱"""
        try:
            cache_key = f"kg:{query}"
            if cache_key in self._kg_cache:
                return self._kg_cache[cache_key]
            
            # 从查询文本构建知识图谱
            kg_graph_id = self.knowledge_builder.build_knowledge_graph_from_text(
                query, 
                source=f"query_{int(time.time())}"
            )
            
            if kg_graph_id:
                # 获取图谱摘要
                kg_summary = self.knowledge_builder.get_knowledge_graph_summary(kg_graph_id)
                self._log(f"构建知识图谱: {kg_summary.get('entities_count', 0)} 个实体, "
                         f"{kg_summary.get('relations_count', 0)} 个关系")
                
                self._kg_cache[cache_key] = kg_graph_id
            
            return kg_graph_id
            
        except Exception as e:
            print(f"知识图谱构建失败: {e}")
            return ""
    
    def _execute_chain_exploration(self, query: str, keywords: Dict[str, List[str]]) -> str:
        """执行链式探索"""
        try:
            # 使用高级关键词作为种子实体
            seed_entities = keywords.get("high_level", [])[:3]
            
            if not seed_entities:
                return ""
            
            # 开始探索
            exploration_path_id = self.chain_explorer.start_exploration(query, seed_entities)
            
            # 执行几步探索
            for step in range(3):
                result = self.chain_explorer.explore_next_step(exploration_path_id)
                if result["status"] in ["completed", "error"]:
                    break
                
                self._log(f"探索步骤 {step + 1}: {result.get('new_nodes_count', 0)} 个新节点")
            
            # 获取探索摘要
            exploration_summary = self.chain_explorer.get_exploration_summary(exploration_path_id)
            self._log(f"链式探索完成: {exploration_summary.get('total_nodes', 0)} 个节点")
            
            return exploration_path_id
            
        except Exception as e:
            print(f"链式探索失败: {e}")
            return ""
    
    def _enhance_final_answer(self, query: str, base_result: str, complexity) -> str:
        """增强最终答案"""
        try:
            # 如果基础结果已经很好，直接返回
            if len(base_result) > 500 and "抱歉" not in base_result:
                return base_result
            
            # 收集增强信息
            enhancement_info = []
            
            # 添加复杂度分析
            enhancement_info.append(f"查询复杂度分析: {complexity.complexity_level.value}")
            
            # 添加社区洞察
            if hasattr(self, '_community_cache') and self._community_cache:
                enhancement_info.append("已应用社区感知增强")
            
            # 添加知识图谱信息
            if hasattr(self, '_kg_cache') and self._kg_cache:
                enhancement_info.append("已构建动态知识图谱")
            
            # 生成增强答案
            enhanced_query = f"""
            基于以下增强分析，改进回答：

            原始问题: {query}
            基础回答: {base_result}
            增强信息: {'; '.join(enhancement_info)}

            请生成一个更全面、更深入的答案。
            """
            
            enhanced_result = self.hybrid_tool.search(enhanced_query)
            
            return enhanced_result if enhanced_result else base_result
            
        except Exception as e:
            print(f"答案增强失败: {e}")
            return base_result
    
    async def search_stream(self, query: str) -> AsyncGenerator[str, None]:
        """
        流式执行增强深度研究
        
        参数:
            query: 查询字符串
            
        返回:
            AsyncGenerator[str, None]: 流式结果生成器
        """
        try:
            yield "**开始增强深度分析**...\n\n"
            
            # 评估复杂度
            complexity = self.complexity_estimator.estimate_complexity(query)
            yield f"**复杂度评估**: {complexity.complexity_level.value}\n"
            
            if complexity.overall_complexity >= self.complexity_threshold:
                yield "**激活增强研究模式**...\n\n"
                
                # 提取关键词
                keywords = self.extract_keywords(query)
                yield f"**关键词提取**: {len(keywords.get('high_level', []))} 个高级概念\n"
                
                # 社区感知
                if self.enable_community_aware:
                    yield "**社区感知分析**...\n"
                    community_context = self._enhance_search_with_community(query, keywords)
                    yield "✓ 社区分析完成\n"
                
                # 知识图谱构建
                if self.enable_kg_building:
                    yield "**构建知识图谱**...\n"
                    kg_graph_id = self._build_dynamic_knowledge_graph(query)
                    yield "✓ 知识图谱构建完成\n"
                
                # 链式探索
                if self.enable_chain_exploration and keywords.get("high_level"):
                    yield "**链式探索搜索**...\n"
                    exploration_path_id = self._execute_chain_exploration(query, keywords)
                    yield "✓ 链式探索完成\n\n"
            
            yield "**生成最终答案**...\n\n"
            
            # 执行搜索
            result = await asyncio.get_event_loop().run_in_executor(None, self.search, query)
            
            # 分块返回结果
            import re
            sentences = re.split(r'([.!?。！？]\s*)', result)
            buffer = ""
            
            for i in range(len(sentences)):
                buffer += sentences[i]
                if i % 2 == 1 or len(buffer) >= 60:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.02)
            
            if buffer:
                yield buffer
                
        except Exception as e:
            yield f"**增强深度研究失败**: {str(e)}"
    
    def get_tool(self) -> BaseTool:
        """获取LangChain兼容的工具"""
        class DeeperResearchRetrievalTool(BaseTool):
            name: str = "deeper_research"
            description: str = "增强版深度研究工具：通过社区感知和知识图谱分析，结合多轮推理和搜索解决复杂问题。"
            
            def _run(self_tool, query: Any) -> str:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")
        
        return DeeperResearchRetrievalTool()
    
    def get_stream_tool(self) -> BaseTool:
        """获取流式工具"""
        class DeeperResearchStreamTool(BaseTool):
            name: str = "deeper_research_stream"
            description: str = "增强版深度研究流式工具：支持流式输出的深度研究。"
            
            def _run(self_tool, query: Any) -> str:
                # 同步版本，返回完整结果
                return self.search(query)
            
            async def _arun(self_tool, query: Any) -> str:
                # 异步版本，收集所有流式输出
                result_parts = []
                async for chunk in self.search_stream(str(query)):
                    result_parts.append(chunk)
                return "".join(result_parts)
        
        return DeeperResearchStreamTool()
    
    def close(self):
        """关闭增强深度研究工具"""
        try:
            # 调用父类方法
            super().close()
            
            # 关闭增强组件
            if hasattr(self, 'complexity_estimator'):
                self.complexity_estimator.close()
            if hasattr(self, 'knowledge_builder'):
                self.knowledge_builder.close()
            if hasattr(self, 'chain_explorer'):
                self.chain_explorer.close()
            
            # 清空缓存
            self._community_cache.clear()
            self._kg_cache.clear()
                
        except Exception as e:
            print(f"增强深度研究工具关闭失败: {e}")
