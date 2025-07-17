"""
查询生成器

生成子查询和跟进查询，支持多种查询策略
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from search_new.config import get_reasoning_config
from search_new.reasoning.utils.nlp_utils import extract_queries_from_text, clean_text


@dataclass
class QueryContext:
    """查询上下文数据类"""
    original_query: str
    current_context: str
    previous_results: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    query_type: str = "general"  # general, followup, sub_query, clarification


class QueryGenerator:
    """
    查询生成器：生成各种类型的搜索查询
    
    主要功能：
    1. 生成子查询
    2. 生成跟进查询
    3. 生成澄清查询
    4. 查询优化和重写
    """
    
    def __init__(self, llm, sub_query_prompt: Optional[str] = None, 
                 followup_prompt: Optional[str] = None):
        """
        初始化查询生成器
        
        参数:
            llm: 大语言模型实例
            sub_query_prompt: 子查询提示模板
            followup_prompt: 跟进查询提示模板
        """
        self.llm = llm
        self.config = get_reasoning_config()
        
        # 设置提示模板
        self.sub_query_prompt = sub_query_prompt or self._get_default_sub_query_prompt()
        self.followup_prompt = followup_prompt or self._get_default_followup_prompt()
        
        # 创建处理链
        self._setup_chains()
        
        # 查询历史
        self.query_history: List[QueryContext] = []
        
        print("查询生成器初始化完成")
    
    def _get_default_sub_query_prompt(self) -> str:
        """获取默认子查询提示模板"""
        return """
        请将以下复杂问题分解为多个简单的子问题，每个子问题都应该是独立可搜索的。

        原始问题: {original_query}
        当前上下文: {context}

        请生成3-5个子查询，每个查询应该：
        1. 针对问题的一个特定方面
        2. 可以独立搜索和回答
        3. 有助于回答原始问题

        请以列表形式输出子查询：
        1. 子查询1
        2. 子查询2
        3. 子查询3
        ...
        """
    
    def _get_default_followup_prompt(self) -> str:
        """获取默认跟进查询提示模板"""
        return """
        基于以下信息，生成跟进查询以获取更多相关信息。

        原始问题: {original_query}
        已获得的信息: {previous_results}
        当前上下文: {context}

        请生成2-3个跟进查询，用于：
        1. 澄清模糊的信息
        2. 获取缺失的细节
        3. 验证已有信息
        4. 探索相关主题

        请以列表形式输出跟进查询：
        1. 跟进查询1
        2. 跟进查询2
        3. 跟进查询3
        """
    
    def _setup_chains(self):
        """设置处理链"""
        try:
            # 子查询生成链
            sub_query_template = ChatPromptTemplate.from_template(self.sub_query_prompt)
            self.sub_query_chain = sub_query_template | self.llm | StrOutputParser()
            
            # 跟进查询生成链
            followup_template = ChatPromptTemplate.from_template(self.followup_prompt)
            self.followup_chain = followup_template | self.llm | StrOutputParser()
            
            # 查询优化链
            optimization_template = ChatPromptTemplate.from_template("""
            请优化以下搜索查询，使其更加精确和有效：

            原始查询: {query}
            优化目标: {optimization_goal}
            上下文: {context}

            请提供优化后的查询，要求：
            1. 更加具体和精确
            2. 包含关键词
            3. 适合搜索引擎
            4. 保持原意不变

            优化后的查询：
            """)
            self.optimization_chain = optimization_template | self.llm | StrOutputParser()
            
        except Exception as e:
            print(f"处理链设置失败: {e}")
            raise
    
    def generate_sub_queries(self, query_context: QueryContext) -> List[str]:
        """
        生成子查询
        
        参数:
            query_context: 查询上下文
            
        返回:
            List[str]: 子查询列表
        """
        try:
            print(f"生成子查询: {query_context.original_query[:50]}...")
            
            # 调用子查询生成链
            result = self.sub_query_chain.invoke({
                "original_query": query_context.original_query,
                "context": query_context.current_context
            })
            
            # 提取查询
            queries = self._extract_queries_from_result(result)
            
            # 记录到历史
            context = QueryContext(
                original_query=query_context.original_query,
                current_context=query_context.current_context,
                query_type="sub_query"
            )
            self.query_history.append(context)
            
            print(f"生成了 {len(queries)} 个子查询")
            return queries
            
        except Exception as e:
            print(f"生成子查询失败: {e}")
            return []
    
    def generate_followup_queries(self, query_context: QueryContext) -> List[str]:
        """
        生成跟进查询
        
        参数:
            query_context: 查询上下文
            
        返回:
            List[str]: 跟进查询列表
        """
        try:
            print(f"生成跟进查询: {query_context.original_query[:50]}...")
            
            # 准备之前的结果
            previous_results_text = "\n".join(query_context.previous_results)
            
            # 调用跟进查询生成链
            result = self.followup_chain.invoke({
                "original_query": query_context.original_query,
                "previous_results": previous_results_text,
                "context": query_context.current_context
            })
            
            # 提取查询
            queries = self._extract_queries_from_result(result)
            
            # 记录到历史
            context = QueryContext(
                original_query=query_context.original_query,
                current_context=query_context.current_context,
                previous_results=query_context.previous_results,
                query_type="followup"
            )
            self.query_history.append(context)
            
            print(f"生成了 {len(queries)} 个跟进查询")
            return queries
            
        except Exception as e:
            print(f"生成跟进查询失败: {e}")
            return []
    
    def optimize_query(self, query: str, optimization_goal: str = "precision", 
                      context: str = "") -> str:
        """
        优化查询
        
        参数:
            query: 原始查询
            optimization_goal: 优化目标 (precision, recall, clarity)
            context: 上下文信息
            
        返回:
            str: 优化后的查询
        """
        try:
            print(f"优化查询: {query[:50]}...")
            
            # 调用优化链
            result = self.optimization_chain.invoke({
                "query": query,
                "optimization_goal": optimization_goal,
                "context": context
            })
            
            # 清理结果
            optimized_query = clean_text(result)
            
            if optimized_query and len(optimized_query) > 3:
                print("查询优化成功")
                return optimized_query
            else:
                print("查询优化结果无效，返回原查询")
                return query
                
        except Exception as e:
            print(f"查询优化失败: {e}")
            return query
    
    def generate_clarification_queries(self, query: str, ambiguous_terms: List[str]) -> List[str]:
        """
        生成澄清查询
        
        参数:
            query: 原始查询
            ambiguous_terms: 模糊术语列表
            
        返回:
            List[str]: 澄清查询列表
        """
        try:
            clarification_queries = []
            
            for term in ambiguous_terms:
                # 为每个模糊术语生成澄清查询
                clarification_query = f"什么是{term}？{term}的定义和含义是什么？"
                clarification_queries.append(clarification_query)
                
                # 生成上下文相关的澄清查询
                context_query = f"在'{query}'的上下文中，{term}指的是什么？"
                clarification_queries.append(context_query)
            
            print(f"生成了 {len(clarification_queries)} 个澄清查询")
            return clarification_queries
            
        except Exception as e:
            print(f"生成澄清查询失败: {e}")
            return []
    
    def _extract_queries_from_result(self, result: str) -> List[str]:
        """
        从LLM结果中提取查询
        
        参数:
            result: LLM生成的结果
            
        返回:
            List[str]: 提取的查询列表
        """
        try:
            # 使用NLP工具提取查询
            queries = extract_queries_from_text(result)
            
            # 清理和验证查询
            cleaned_queries = []
            for query in queries:
                cleaned = clean_text(query)
                if cleaned and len(cleaned) > 3:
                    cleaned_queries.append(cleaned)
            
            return cleaned_queries
            
        except Exception as e:
            print(f"提取查询失败: {e}")
            return []
    
    def get_query_suggestions(self, partial_query: str, context: str = "") -> List[str]:
        """
        获取查询建议
        
        参数:
            partial_query: 部分查询
            context: 上下文
            
        返回:
            List[str]: 查询建议列表
        """
        try:
            # 基于部分查询生成完整查询建议
            suggestions_prompt = f"""
            基于以下部分查询，生成3-5个完整的查询建议：

            部分查询: {partial_query}
            上下文: {context}

            请生成相关的完整查询建议：
            1. 建议1
            2. 建议2
            3. 建议3
            ...
            """
            
            template = ChatPromptTemplate.from_template(suggestions_prompt)
            chain = template | self.llm | StrOutputParser()
            
            result = chain.invoke({
                "partial_query": partial_query,
                "context": context
            })
            
            suggestions = self._extract_queries_from_result(result)
            
            print(f"生成了 {len(suggestions)} 个查询建议")
            return suggestions
            
        except Exception as e:
            print(f"获取查询建议失败: {e}")
            return []
    
    def get_query_history(self) -> List[Dict[str, Any]]:
        """
        获取查询历史
        
        返回:
            List[Dict]: 查询历史
        """
        return [
            {
                "original_query": ctx.original_query,
                "query_type": ctx.query_type,
                "keywords": ctx.keywords,
                "results_count": len(ctx.previous_results)
            }
            for ctx in self.query_history
        ]
    
    def clear_history(self):
        """清空查询历史"""
        self.query_history.clear()
        print("查询历史已清空")
    
    def close(self):
        """关闭查询生成器"""
        self.clear_history()
        print("查询生成器已关闭")
