"""
本地搜索工具

基于向量检索实现社区内部的精确查询
"""

from typing import List, Dict, Any, Union, Optional
import time
import json
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.tools.retriever import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser

from config.prompt import LC_SYSTEM_PROMPT, contextualize_q_system_prompt
from config.settings import lc_description
from search_new.tools.base_tool import BaseSearchTool
from search_new.core.local_search import LocalSearch


class LocalSearchTool(BaseSearchTool):
    """本地搜索工具，基于向量检索实现社区内部的精确查询"""
    
    def __init__(self):
        """初始化本地搜索工具"""
        # 设置标志，阻止基类调用_setup_chains
        self._skip_setup_chains = True

        # 先初始化基类以获取config
        super().__init__()

        # 然后设置特定的缓存目录
        if hasattr(self, 'cache_manager') and self.cache_manager:
            self.cache_manager.cache_dir = self.config.cache.local_search_cache_dir

        # 设置聊天历史，用于连续对话
        self.chat_history = []

        # 创建本地搜索器
        self.local_searcher = LocalSearch(self.llm, self.embeddings)

        # 创建检索器
        try:
            self.retriever = self.local_searcher.as_retriever()
        except Exception as e:
            print(f"创建检索器失败: {e}")
            self.retriever = None

        # 现在可以安全地设置处理链
        if self.retriever:
            self._setup_chains()

        print("本地搜索工具初始化完成")
    
    def _setup_chains(self):
        """设置处理链"""
        try:
            # 设置关键词提取链
            self._setup_keyword_chain()
            
            # 设置RAG链
            self._setup_rag_chain()
            
        except Exception as e:
            print(f"处理链设置失败: {e}")
            raise
    
    def _setup_keyword_chain(self):
        """设置关键词提取链"""
        keyword_prompt = ChatPromptTemplate.from_template("""
        请从以下查询中提取关键词，分为低级关键词（具体实体、人名、地名等）和高级关键词（概念、主题等）。

        查询: {query}

        请以JSON格式返回结果：
        {{
            "low_level": ["关键词1", "关键词2"],
            "high_level": ["概念1", "概念2"]
        }}
        """)
        
        self.keyword_chain = keyword_prompt | self.llm | StrOutputParser()
    
    def _setup_rag_chain(self):
        """设置RAG处理链"""
        try:
            # 创建检索器（如果还没有创建）
            if self.retriever is None:
                self.retriever = self.local_searcher.as_retriever()

            # 创建历史感知检索器
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            history_aware_retriever = create_history_aware_retriever(
                self.llm, self.retriever, contextualize_q_prompt
            )
            
            # 创建问答链
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", LC_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", """
                基于以下上下文信息回答问题：

                {context}

                问题: {input}

                请按以下格式输出回答：
                1. 使用三级标题(###)标记主题
                2. 主要内容用清晰的段落展示
                3. 最后必须用"#### 引用数据"标记引用部分，列出用到的数据来源
                """),
            ])
            
            question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
            
            # 创建完整的RAG链
            self.rag_chain = create_retrieval_chain(
                history_aware_retriever, 
                question_answer_chain
            )
            
        except Exception as e:
            print(f"RAG链设置失败: {e}")
            raise
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 分类关键词字典
        """
        # 检查缓存
        cache_key = f"keywords:{query}"
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
            tuple: (query, keywords)
        """
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
            keywords = query_input.get("keywords", [])
        else:
            query = str(query_input)
            keywords = []
        
        return query, keywords
    
    def search(self, query_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行本地搜索

        参数:
            query_input: 查询输入，可以是字符串或字典

        返回:
            Dict[str, Any]: 包含answer和sources的搜索结果
        """
        overall_start = time.time()
        self._reset_metrics()
        
        try:
            # 解析输入
            query, keywords = self._parse_query_input(query_input)
            
            # 生成缓存键
            cache_key = self._get_cache_key(query, keywords=keywords)
            
            # 检查缓存
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                print(f"本地搜索缓存命中: {query[:50]}...")
                # 如果缓存的是字符串，转换为字典格式
                if isinstance(cached_result, str):
                    return {
                        "answer": cached_result,
                        "sources": [],
                        "metadata": {"from_cache": True}
                    }
                return cached_result

            print(f"开始本地搜索: {query[:100]}...")
            
            # 使用RAG链执行搜索
            search_start = time.time()
            ai_msg = self.rag_chain.invoke({
                "input": query,
                "chat_history": self.chat_history,
            })
            self.performance_metrics["query_time"] = time.time() - search_start
            
            # 获取结果
            answer = ai_msg.get("answer", "抱歉，我无法回答这个问题。")

            # 构建返回结果
            if not answer or answer.strip() == "":
                search_result = {
                    "answer": "未找到相关信息",
                    "sources": [],
                    "metadata": {
                        "query_time": self.performance_metrics["query_time"]
                    }
                }
            else:
                search_result = {
                    "answer": answer,
                    "sources": [],  # 本地搜索的sources信息在context中，这里简化处理
                    "metadata": {
                        "query_time": self.performance_metrics["query_time"]
                    }
                }

            # 更新聊天历史
            self.chat_history.append({"human": query, "ai": answer})

            # 限制聊天历史长度
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]

            # 缓存结果
            self._set_to_cache(cache_key, search_result)

            # 记录性能指标
            self.performance_metrics["total_time"] = time.time() - overall_start

            print(f"本地搜索完成，耗时: {self.performance_metrics['total_time']:.2f}s")
            return search_result

        except Exception as e:
            print(f"本地搜索失败: {e}")
            self.error_stats["query_errors"] += 1
            self.performance_metrics["total_time"] = time.time() - overall_start

            return {
                "answer": f"搜索过程中出现问题: {str(e)}",
                "sources": [],
                "metadata": {"error": True}
            }
    
    def search_with_details(self, query_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行本地搜索并返回详细信息
        
        参数:
            query_input: 查询输入
            
        返回:
            Dict: 包含搜索结果和详细信息的字典
        """
        start_time = time.time()
        
        try:
            # 解析输入
            query, keywords = self._parse_query_input(query_input)
            
            # 执行搜索
            result = self.search(query_input)
            
            # 提取关键词
            extracted_keywords = self.extract_keywords(query)
            
            # 获取检索到的文档
            docs = self.retriever.get_relevant_documents(query)
            
            return {
                "result": result,
                "query": query,
                "keywords": keywords,
                "extracted_keywords": extracted_keywords,
                "documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in docs
                ],
                "chat_history_length": len(self.chat_history),
                "performance": self.get_performance_metrics(),
                "error_stats": self.get_error_stats(),
                "total_time": time.time() - start_time,
                "tool_name": "LocalSearchTool"
            }
            
        except Exception as e:
            print(f"详细本地搜索失败: {e}")
            return {
                "result": f"搜索失败: {str(e)}",
                "query": str(query_input),
                "error": str(e),
                "performance": self.get_performance_metrics(),
                "error_stats": self.get_error_stats(),
                "total_time": time.time() - start_time,
                "tool_name": "LocalSearchTool"
            }
    
    def get_tool(self):
        """
        返回LangChain兼容的检索工具
        
        返回:
            BaseTool: 检索工具实例
        """
        return create_retriever_tool(
            self.retriever,
            "local_search_tool",
            lc_description,
        )
    
    def clear_chat_history(self):
        """清空聊天历史"""
        self.chat_history = []
        print("聊天历史已清空")
    
    def get_chat_history(self) -> List[Dict]:
        """获取聊天历史"""
        return self.chat_history.copy()
    
    def close(self):
        """关闭资源"""
        try:
            # 调用父类方法
            super().close()
            
            # 关闭本地搜索器
            if hasattr(self, 'local_searcher'):
                self.local_searcher.close()
                
        except Exception as e:
            print(f"本地搜索工具关闭失败: {e}")
