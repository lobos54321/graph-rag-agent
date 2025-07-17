"""
简单搜索工具

仅使用向量搜索的简单实现
"""

from typing import List, Dict, Any, Union
import time
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.neo4jdb import get_db_manager
from search_new.tools.base_tool import BaseSearchTool


class NaiveSearchTool(BaseSearchTool):
    """简单搜索工具，仅使用向量搜索实现"""
    
    def __init__(self):
        """初始化简单搜索工具"""
        super().__init__(cache_dir=f"{self.config.cache.base_cache_dir}/naive_search")
        
        # 搜索配置
        self.top_k = 5  # 返回的最大结果数
        self.similarity_threshold = 0.7  # 相似度阈值
        
        # 数据库连接
        self.db_manager = get_db_manager()
        self.graph = self.db_manager.get_graph()
        
        print("简单搜索工具初始化完成")
    
    def _setup_chains(self):
        """设置处理链"""
        try:
            # 设置关键词提取链
            self._setup_keyword_chain()
            
            # 设置答案生成链
            self._setup_answer_chain()
            
        except Exception as e:
            print(f"处理链设置失败: {e}")
            raise
    
    def _setup_keyword_chain(self):
        """设置关键词提取链"""
        keyword_prompt = ChatPromptTemplate.from_template("""
        请从以下查询中提取关键词，重点关注具体的实体和概念。

        查询: {query}

        请以JSON格式返回结果：
        {{
            "low_level": ["关键词1", "关键词2"],
            "high_level": ["概念1", "概念2"]
        }}
        """)
        
        self.keyword_chain = keyword_prompt | self.llm | StrOutputParser()
    
    def _setup_answer_chain(self):
        """设置答案生成链"""
        answer_prompt = ChatPromptTemplate.from_template("""
        基于以下检索到的信息回答用户问题：

        用户问题: {query}

        检索信息:
        {context}

        请生成一个简洁、准确的答案。如果信息不足，请说明。
        请按以下格式输出：
        1. 使用三级标题(###)标记主题
        2. 主要内容用清晰的段落展示
        3. 最后必须用"#### 引用数据"标记引用部分，列出用到的数据来源
        """)
        
        self.answer_chain = answer_prompt | self.llm | StrOutputParser()
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 分类关键词字典
        """
        # 检查缓存
        cache_key = f"naive_keywords:{query}"
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
    
    def _vector_search_chunks(self, query: str) -> List[Dict[str, Any]]:
        """
        在Chunk节点上执行向量搜索
        
        参数:
            query: 搜索查询
            
        返回:
            List[Dict]: 搜索结果列表
        """
        try:
            # 生成查询向量
            query_embedding = self.embeddings.embed_query(query)
            
            # 构建向量搜索查询
            cypher = """
            CALL db.index.vector.queryNodes('vector', $top_k, $embedding)
            YIELD node, score
            WHERE score >= $threshold
            MATCH (node)-[:PART_OF]->(d:__Document__)
            RETURN {
                content: node.text,
                score: score,
                document_id: d.id,
                document_title: d.title,
                chunk_id: node.id
            } AS result
            ORDER BY score DESC
            """
            
            # 执行查询
            start_time = time.time()
            results = self.graph.query(cypher, {
                "embedding": query_embedding,
                "top_k": self.top_k,
                "threshold": self.similarity_threshold
            })
            self.performance_metrics["query_time"] += time.time() - start_time
            
            # 处理结果
            chunks = []
            for record in results:
                result_data = record['result']
                chunks.append({
                    "content": result_data['content'],
                    "score": result_data['score'],
                    "document_id": result_data['document_id'],
                    "document_title": result_data['document_title'],
                    "chunk_id": result_data['chunk_id']
                })
            
            print(f"向量搜索找到 {len(chunks)} 个相关块")
            return chunks
            
        except Exception as e:
            print(f"向量搜索失败: {e}")
            self.error_stats["query_errors"] += 1
            return []
    
    def _text_search_fallback(self, query: str) -> List[Dict[str, Any]]:
        """
        文本搜索备选方案
        
        参数:
            query: 搜索查询
            
        返回:
            List[Dict]: 搜索结果列表
        """
        try:
            cypher = """
            MATCH (chunk:__Chunk__)-[:PART_OF]->(d:__Document__)
            WHERE chunk.text CONTAINS $query
            RETURN {
                content: chunk.text,
                score: 0.5,
                document_id: d.id,
                document_title: d.title,
                chunk_id: chunk.id
            } AS result
            LIMIT $top_k
            """
            
            results = self.graph.query(cypher, {
                "query": query,
                "top_k": self.top_k
            })
            
            chunks = []
            for record in results:
                result_data = record['result']
                chunks.append({
                    "content": result_data['content'],
                    "score": result_data['score'],
                    "document_id": result_data['document_id'],
                    "document_title": result_data['document_title'],
                    "chunk_id": result_data['chunk_id']
                })
            
            print(f"文本搜索找到 {len(chunks)} 个相关块")
            return chunks
            
        except Exception as e:
            print(f"文本搜索失败: {e}")
            return []
    
    def _generate_answer(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        基于检索到的块生成答案
        
        参数:
            query: 用户查询
            chunks: 检索到的文档块
            
        返回:
            str: 生成的答案
        """
        if not chunks:
            return "抱歉，我无法在知识库中找到相关信息来回答您的问题。"
        
        try:
            # 构建上下文
            context_parts = []
            for i, chunk in enumerate(chunks):
                context_parts.append(
                    f"**文档 {i+1}** (来源: {chunk['document_title']}, 相似度: {chunk['score']:.3f}):\n"
                    f"{chunk['content']}\n"
                )
            
            context = "\n".join(context_parts)
            
            # 生成答案
            answer = self.answer_chain.invoke({
                "query": query,
                "context": context
            })
            
            return answer
            
        except Exception as e:
            print(f"答案生成失败: {e}")
            return f"答案生成过程中出现问题: {str(e)}"
    
    def search(self, query_input: Union[str, Dict[str, Any]]) -> str:
        """
        执行简单搜索
        
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
                top_k = query_input.get("top_k", self.top_k)
            else:
                query = str(query_input)
                top_k = self.top_k
            
            # 生成缓存键
            cache_key = self._get_cache_key(query, top_k=top_k)
            
            # 检查缓存
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                print(f"简单搜索缓存命中: {query[:50]}...")
                return cached_result
            
            print(f"开始简单搜索: {query[:100]}...")
            
            # 执行向量搜索
            chunks = self._vector_search_chunks(query)
            
            # 如果向量搜索没有结果，尝试文本搜索
            if not chunks:
                print("向量搜索无结果，尝试文本搜索")
                chunks = self._text_search_fallback(query)
            
            # 生成答案
            result = self._generate_answer(query, chunks)
            
            # 缓存结果
            self._set_to_cache(cache_key, result)
            
            # 记录性能指标
            self.performance_metrics["total_time"] = time.time() - overall_start
            
            print(f"简单搜索完成，耗时: {self.performance_metrics['total_time']:.2f}s")
            return result
            
        except Exception as e:
            print(f"简单搜索失败: {e}")
            self.error_stats["query_errors"] += 1
            self.performance_metrics["total_time"] = time.time() - overall_start
            
            return f"搜索过程中出现问题: {str(e)}"
    
    def search_with_details(self, query_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行简单搜索并返回详细信息
        
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
            
            # 获取检索到的块
            chunks = self._vector_search_chunks(query)
            if not chunks:
                chunks = self._text_search_fallback(query)
            
            # 提取关键词
            keywords = self.extract_keywords(query)
            
            return {
                "result": result,
                "query": query,
                "keywords": keywords,
                "chunks_found": len(chunks),
                "chunks": chunks,
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold,
                "performance": self.get_performance_metrics(),
                "error_stats": self.get_error_stats(),
                "total_time": time.time() - start_time,
                "tool_name": "NaiveSearchTool"
            }
            
        except Exception as e:
            print(f"详细简单搜索失败: {e}")
            return {
                "result": f"搜索失败: {str(e)}",
                "query": str(query_input),
                "error": str(e),
                "performance": self.get_performance_metrics(),
                "error_stats": self.get_error_stats(),
                "total_time": time.time() - start_time,
                "tool_name": "NaiveSearchTool"
            }
    
    def close(self):
        """关闭资源"""
        try:
            # 调用父类方法
            super().close()
            
            # 关闭图数据库连接
            if hasattr(self, 'graph') and hasattr(self.graph, 'close'):
                self.graph.close()
                
        except Exception as e:
            print(f"简单搜索工具关闭失败: {e}")
