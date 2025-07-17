"""
本地搜索实现

基于向量检索的社区内精确查询
"""

from typing import Dict, Any
import time

from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.prompt import LC_SYSTEM_PROMPT
from search_new.core.base_search import BaseSearch


class LocalSearch(BaseSearch):
    """
    本地搜索类：使用Neo4j和LangChain实现基于向量检索的本地搜索功能
    
    该类通过向量相似度搜索在知识图谱中查找相关内容，并生成回答
    主要功能包括：
    1. 基于向量相似度的文本检索
    2. 社区内容和关系的检索
    3. 使用LLM生成最终答案
    """
    
    def __init__(self, 
                 llm=None, 
                 embeddings=None,
                 response_type: str = "多个段落"):
        """
        初始化本地搜索类
        
        参数:
            llm: 大语言模型实例
            embeddings: 嵌入模型实例
            response_type: 响应类型
        """
        super().__init__(
            llm=llm,
            embeddings=embeddings,
            cache_dir=None  # 将在父类中设置
        )
        
        # 搜索配置
        self.response_type = response_type
        self.local_config = self.config.local_search

        # 设置缓存目录
        if not hasattr(self, 'cache_manager') or self.cache_manager is None:
            self._setup_cache(self.config.cache.local_search_cache_dir)
        
        # 检索参数
        self.top_entities = self.local_config.top_entities
        self.top_chunks = self.local_config.top_chunks
        self.top_communities = self.local_config.top_communities
        self.top_outside_rels = self.local_config.top_outside_rels
        self.top_inside_rels = self.local_config.top_inside_rels
        self.index_name = self.local_config.index_name
        
        # 检索查询模板
        self.retrieval_query = self.local_config.retrieval_query
        
        # 数据库连接信息
        self.neo4j_uri = self.db_manager.neo4j_uri
        self.neo4j_username = self.db_manager.neo4j_username
        self.neo4j_password = self.db_manager.neo4j_password
        
        print(f"本地搜索初始化完成，配置: top_entities={self.top_entities}")
    
    def _build_final_query(self) -> str:
        """构建最终的检索查询"""
        return self.retrieval_query.replace("$topChunks", str(self.top_chunks))\
            .replace("$topCommunities", str(self.top_communities))\
            .replace("$topOutsideRels", str(self.top_outside_rels))\
            .replace("$topInsideRels", str(self.top_inside_rels))
    
    def _create_vector_store(self) -> Neo4jVector:
        """创建向量存储实例"""
        try:
            final_query = self._build_final_query()
            
            vector_store = Neo4jVector.from_existing_index(
                self.embeddings,
                url=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password,
                index_name=self.index_name,
                retrieval_query=final_query
            )
            
            return vector_store
            
        except Exception as e:
            print(f"创建向量存储失败: {e}")
            raise
    
    def as_retriever(self, **kwargs):
        """
        返回检索器实例，用于链式调用
        
        返回:
            检索器实例
        """
        try:
            vector_store = self._create_vector_store()
            
            return vector_store.as_retriever(
                search_kwargs={"k": self.top_entities, **kwargs}
            )
            
        except Exception as e:
            print(f"创建检索器失败: {e}")
            raise
    
    def _similarity_search(self, query: str) -> list:
        """
        执行相似度搜索
        
        参数:
            query: 搜索查询
            
        返回:
            list: 搜索结果文档列表
        """
        try:
            vector_store = self._create_vector_store()
            
            docs = vector_store.similarity_search(
                query,
                k=self.top_entities,
                params={
                    "topChunks": self.top_chunks,
                    "topCommunities": self.top_communities,
                    "topOutsideRels": self.top_outside_rels,
                    "topInsideRels": self.top_inside_rels,
                }
            )
            
            return docs
            
        except Exception as e:
            print(f"相似度搜索失败: {e}")
            raise
    
    def _generate_response(self, query: str, context: str) -> str:
        """
        使用LLM生成响应
        
        参数:
            query: 用户查询
            context: 检索到的上下文
            
        返回:
            str: 生成的响应
        """
        try:
            # 创建提示模板
            prompt = ChatPromptTemplate.from_messages([
                ("system", LC_SYSTEM_PROMPT),
                ("human", """
                    ---分析报告--- 
                    请注意，下面提供的分析报告按**重要性降序排列**。

                    {context}

                    用户的问题是：
                    {input}

                    请按以下格式输出回答：
                    1. 使用三级标题(###)标记主题
                    2. 主要内容用清晰的段落展示
                    3. 最后必须用"#### 引用数据"标记引用部分，列出用到的数据来源
                    """
                 )
            ])
            
            # 创建处理链
            chain = prompt | self.llm | StrOutputParser()
            
            # 生成响应
            response = chain.invoke({
                "context": context,
                "input": query,
                "response_type": self.response_type
            })
            
            return response
            
        except Exception as e:
            print(f"响应生成失败: {e}")
            raise
    
    def search(self, query: str, **kwargs) -> str:
        """
        执行本地搜索
        
        参数:
            query: 搜索查询字符串
            **kwargs: 其他参数
            
        返回:
            str: 生成的最终答案
        """
        overall_start = time.time()
        self._reset_metrics()
        
        try:
            # 生成缓存键
            cache_key = self._get_cache_key(query, **kwargs)
            
            # 检查缓存
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                print(f"本地搜索缓存命中: {query[:50]}...")
                return cached_result
            
            print(f"开始本地搜索: {query[:100]}...")
            
            # 执行相似度搜索
            search_start = time.time()
            docs = self._similarity_search(query)
            self.performance_metrics["query_time"] = time.time() - search_start
            
            # 提取上下文
            context = ""
            if docs:
                context = docs[0].page_content
                print(f"检索到 {len(docs)} 个文档，上下文长度: {len(context)}")
            else:
                print("未检索到相关文档")
                context = "未找到相关信息"
            
            # 生成响应
            if context and context != "未找到相关信息":
                response = self._generate_response(query, context)
            else:
                response = "抱歉，我无法在知识库中找到相关信息来回答您的问题。"
            
            # 缓存结果
            self._set_to_cache(cache_key, response)
            
            # 记录总时间
            self.performance_metrics["total_time"] = time.time() - overall_start
            
            print(f"本地搜索完成，耗时: {self.performance_metrics['total_time']:.2f}s")
            
            return response
            
        except Exception as e:
            print(f"本地搜索失败: {e}")
            self.error_stats["query_errors"] += 1
            self.performance_metrics["total_time"] = time.time() - overall_start
            
            return f"搜索过程中出现问题: {str(e)}"
    
    def search_with_details(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        执行本地搜索并返回详细信息
        
        参数:
            query: 搜索查询字符串
            **kwargs: 其他参数
            
        返回:
            Dict: 包含搜索结果和详细信息的字典
        """
        start_time = time.time()
        
        try:
            # 执行搜索
            result = self.search(query, **kwargs)
            
            # 获取详细信息
            docs = self._similarity_search(query)
            
            return {
                "result": result,
                "documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in docs
                ],
                "performance": self.get_performance_metrics(),
                "config": {
                    "top_entities": self.top_entities,
                    "top_chunks": self.top_chunks,
                    "top_communities": self.top_communities,
                    "response_type": self.response_type
                },
                "total_time": time.time() - start_time
            }
            
        except Exception as e:
            print(f"详细搜索失败: {e}")
            return {
                "result": f"搜索失败: {str(e)}",
                "documents": [],
                "performance": self.get_performance_metrics(),
                "error": str(e),
                "total_time": time.time() - start_time
            }
