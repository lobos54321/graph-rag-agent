"""
全局搜索实现

基于Map-Reduce模式的跨社区查询
"""

from typing import List, Dict, Any, Optional
import time
from tqdm import tqdm

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.prompt import MAP_SYSTEM_PROMPT, REDUCE_SYSTEM_PROMPT
from search_new.core.base_search import BaseSearch


class GlobalSearch(BaseSearch):
    """
    全局搜索类：使用Neo4j和LangChain实现基于Map-Reduce模式的全局搜索功能
    
    该类主要用于在整个知识图谱范围内进行搜索，采用以下步骤：
    1. 获取指定层级的所有社区数据
    2. Map阶段：为每个社区生成中间结果
    3. Reduce阶段：整合所有中间结果生成最终答案
    """
    
    def __init__(self, 
                 llm=None, 
                 response_type: str = "多个段落"):
        """
        初始化全局搜索类
        
        参数:
            llm: 大语言模型实例
            response_type: 响应类型，默认为"多个段落"
        """
        super().__init__(
            llm=llm,
            cache_dir=None  # 将在父类中设置
        )
        
        # 搜索配置
        self.response_type = response_type
        self.global_config = self.config.global_search

        # 设置缓存目录
        if not hasattr(self, 'cache_manager') or self.cache_manager is None:
            self._setup_cache(self.config.cache.global_search_cache_dir)
        
        # 全局搜索参数
        self.default_level = self.global_config.default_level
        self.batch_size = self.global_config.batch_size
        self.max_communities = self.global_config.max_communities
        
        print(f"全局搜索初始化完成，默认层级: {self.default_level}")
    
    def _get_community_data(self, level: int) -> List[Dict]:
        """
        获取指定层级的社区数据
        
        参数:
            level: 社区层级
            
        返回:
            List[Dict]: 社区数据字典列表
        """
        try:
            cypher = """
            MATCH (c:__Community__)
            WHERE c.level = $level
            RETURN {communityId:c.id, full_content:c.full_content} AS output
            LIMIT $max_communities
            """
            
            result = self._execute_db_query(cypher, {
                "level": level,
                "max_communities": self.max_communities
            })
            
            # 转换结果格式
            communities = []
            if hasattr(result, 'data'):
                communities = [record['output'] for record in result.data()]
            elif hasattr(result, 'to_dict'):
                result_dict = result.to_dict()
                if 'data' in result_dict:
                    communities = [record['output'] for record in result_dict['data']]
            else:
                # 处理pandas DataFrame格式
                if hasattr(result, 'iterrows'):
                    communities = [row['output'] for _, row in result.iterrows()]
                else:
                    communities = result if isinstance(result, list) else []
            
            print(f"获取到 {len(communities)} 个层级 {level} 的社区")
            return communities
            
        except Exception as e:
            print(f"获取社区数据失败: {e}")
            return []
    
    def _process_single_community(self, query: str, community: Dict) -> str:
        """
        处理单个社区数据（Map阶段）
        
        参数:
            query: 搜索查询字符串
            community: 社区数据字典
            
        返回:
            str: 该社区的中间结果
        """
        try:
            # 创建Map阶段的提示模板
            map_prompt = ChatPromptTemplate.from_messages([
                ("system", MAP_SYSTEM_PROMPT),
                ("human", """
                    ---社区数据---
                    {community_data}

                    用户的问题是：
                    {question}
                    
                    请基于这个社区的数据，提供与问题相关的信息摘要。
                    如果社区数据与问题无关，请回答"无相关信息"。
                    """),
            ])
            
            # 创建Map阶段的处理链
            map_chain = map_prompt | self.llm | StrOutputParser()
            
            # 处理社区数据
            community_content = community.get('full_content', '')
            if not community_content:
                return "无相关信息"
            
            # 生成中间结果
            intermediate_result = map_chain.invoke({
                "community_data": community_content,
                "question": query,
                "response_type": self.response_type,
            })
            
            return intermediate_result.strip()
            
        except Exception as e:
            print(f"处理社区数据失败: {e}")
            return "处理失败"
    
    def _process_communities(self, query: str, communities: List[Dict]) -> List[str]:
        """
        批量处理社区数据（Map阶段）
        
        参数:
            query: 搜索查询字符串
            communities: 社区数据列表
            
        返回:
            List[str]: 中间结果列表
        """
        if not communities:
            print("没有社区数据需要处理")
            return []
        
        intermediate_results = []
        
        # 使用进度条显示处理进度
        for community in tqdm(communities, desc="处理社区数据"):
            try:
                result = self._process_single_community(query, community)
                
                # 过滤无效结果
                if result and result.strip() and result.strip() != "无相关信息":
                    intermediate_results.append(result)
                    
            except Exception as e:
                print(f"处理社区失败: {e}")
                continue
        
        print(f"成功处理 {len(intermediate_results)} 个社区的数据")
        return intermediate_results
    
    def _reduce_results(self, query: str, intermediate_results: List[str]) -> str:
        """
        整合中间结果生成最终答案（Reduce阶段）
        
        参数:
            query: 搜索查询字符串
            intermediate_results: 中间结果列表
            
        返回:
            str: 最终生成的答案
        """
        if not intermediate_results:
            return "抱歉，我无法在知识库中找到相关信息来回答您的问题。"
        
        try:
            # 设置Reduce阶段的提示模板
            reduce_prompt = ChatPromptTemplate.from_messages([
                ("system", REDUCE_SYSTEM_PROMPT),
                ("human", """
                    ---分析报告--- 
                    {report_data}

                    用户的问题是：
                    {question}
                    
                    请基于以上分析报告，生成一个全面、准确的答案。
                    请按以下格式输出：
                    1. 使用三级标题(###)标记主题
                    2. 主要内容用清晰的段落展示
                    3. 最后必须用"#### 引用数据"标记引用部分，列出用到的数据来源
                    """),
            ])
            
            # 创建Reduce阶段的处理链
            reduce_chain = reduce_prompt | self.llm | StrOutputParser()
            
            # 合并中间结果
            report_data = "\n\n".join([
                f"**报告 {i+1}:**\n{result}" 
                for i, result in enumerate(intermediate_results)
            ])
            
            # 生成最终答案
            final_answer = reduce_chain.invoke({
                "report_data": report_data,
                "question": query,
                "response_type": self.response_type,
            })
            
            return final_answer
            
        except Exception as e:
            print(f"结果整合失败: {e}")
            return f"结果整合过程中出现问题: {str(e)}"
    
    def search(self, query: str, level: Optional[int] = None, **kwargs) -> str:
        """
        执行全局搜索
        
        参数:
            query: 搜索查询字符串
            level: 要搜索的社区层级，如果为None则使用默认层级
            **kwargs: 其他参数
            
        返回:
            str: 生成的最终答案
        """
        overall_start = time.time()
        self._reset_metrics()
        
        # 使用默认层级
        if level is None:
            level = self.default_level
        
        try:
            # 生成缓存键
            cache_key = self._get_cache_key(query, level=level, **kwargs)
            
            # 检查缓存
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                print(f"全局搜索缓存命中: {query[:50]}...")
                return cached_result
            
            print(f"开始全局搜索: {query[:100]}..., 层级: {level}")
            
            # 获取社区数据
            communities = self._get_community_data(level)
            if not communities:
                result = "抱歉，未找到相关的社区数据。"
                self._set_to_cache(cache_key, result)
                return result
            
            # 处理社区数据（Map阶段）
            intermediate_results = self._process_communities(query, communities)
            
            # 生成最终答案（Reduce阶段）
            final_answer = self._reduce_results(query, intermediate_results)
            
            # 缓存结果
            self._set_to_cache(cache_key, final_answer)
            
            # 记录总时间
            self.performance_metrics["total_time"] = time.time() - overall_start
            
            print(f"全局搜索完成，耗时: {self.performance_metrics['total_time']:.2f}s")
            
            return final_answer
            
        except Exception as e:
            print(f"全局搜索失败: {e}")
            self.error_stats["query_errors"] += 1
            self.performance_metrics["total_time"] = time.time() - overall_start
            
            return f"搜索过程中出现问题: {str(e)}"
    
    def search_with_details(self, query: str, level: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        执行全局搜索并返回详细信息
        
        参数:
            query: 搜索查询字符串
            level: 社区层级
            **kwargs: 其他参数
            
        返回:
            Dict: 包含搜索结果和详细信息的字典
        """
        start_time = time.time()
        
        if level is None:
            level = self.default_level
        
        try:
            # 执行搜索
            result = self.search(query, level, **kwargs)
            
            # 获取社区数据
            communities = self._get_community_data(level)
            
            return {
                "result": result,
                "communities_count": len(communities),
                "level": level,
                "performance": self.get_performance_metrics(),
                "config": {
                    "default_level": self.default_level,
                    "batch_size": self.batch_size,
                    "max_communities": self.max_communities,
                    "response_type": self.response_type
                },
                "total_time": time.time() - start_time
            }
            
        except Exception as e:
            print(f"详细搜索失败: {e}")
            return {
                "result": f"搜索失败: {str(e)}",
                "communities_count": 0,
                "level": level,
                "performance": self.get_performance_metrics(),
                "error": str(e),
                "total_time": time.time() - start_time
            }
