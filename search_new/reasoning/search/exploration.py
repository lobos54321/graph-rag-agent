"""
链式探索组件

支持在知识图谱中进行多步探索和推理
"""

from typing import List, Dict, Any, Optional, Set
import time
from dataclasses import dataclass, field

from search_new.config import get_reasoning_config


@dataclass
class ExplorationNode:
    """探索节点数据类"""
    node_id: str
    node_type: str  # entity, document, community
    content: str
    relevance_score: float = 0.0
    depth: int = 0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExplorationPath:
    """探索路径数据类"""
    path_id: str
    query: str
    nodes: List[ExplorationNode] = field(default_factory=list)
    total_score: float = 0.0
    status: str = "active"  # active, completed, abandoned
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class ChainedExploration:
    """
    链式探索：在知识图谱中进行多步探索
    
    主要功能：
    1. 实体邻居探索
    2. 关系链追踪
    3. 社区内容探索
    4. 路径评分和排序
    """
    
    def __init__(self, graph_query_func: callable, max_steps: Optional[int] = None,
                 exploration_width: Optional[int] = None):
        """
        初始化链式探索
        
        参数:
            graph_query_func: 图查询函数
            max_steps: 最大探索步数
            exploration_width: 探索宽度
        """
        self.graph_query_func = graph_query_func
        self.config = get_reasoning_config()
        
        # 探索配置
        self.max_steps = max_steps or self.config.exploration.max_exploration_steps
        self.exploration_width = exploration_width or self.config.exploration.exploration_width
        self.relevance_threshold = self.config.exploration.relevance_threshold
        self.decay_factor = self.config.exploration.exploration_decay_factor
        self.enable_backtracking = self.config.exploration.enable_backtracking
        
        # 探索状态
        self.exploration_paths: Dict[str, ExplorationPath] = {}
        self.current_path_id: Optional[str] = None
        self.visited_nodes: Set[str] = set()
        
        print(f"链式探索初始化完成，最大步数: {self.max_steps}")
    
    def start_exploration(self, query: str, seed_entities: List[str]) -> str:
        """
        开始探索
        
        参数:
            query: 探索查询
            seed_entities: 种子实体列表
            
        返回:
            str: 探索路径ID
        """
        try:
            path_id = f"path_{int(time.time() * 1000)}"
            
            # 创建探索路径
            path = ExplorationPath(
                path_id=path_id,
                query=query
            )
            
            # 添加种子节点
            for i, entity in enumerate(seed_entities):
                seed_node = ExplorationNode(
                    node_id=entity,
                    node_type="entity",
                    content=entity,
                    relevance_score=1.0,  # 种子节点初始相关性为1
                    depth=0
                )
                path.nodes.append(seed_node)
                self.visited_nodes.add(entity)
            
            self.exploration_paths[path_id] = path
            self.current_path_id = path_id
            
            print(f"开始探索: {path_id}，种子实体: {len(seed_entities)}")
            return path_id
            
        except Exception as e:
            print(f"开始探索失败: {e}")
            return ""
    
    def explore_next_step(self, path_id: Optional[str] = None) -> Dict[str, Any]:
        """
        执行下一步探索
        
        参数:
            path_id: 探索路径ID
            
        返回:
            Dict: 探索结果
        """
        if path_id is None:
            path_id = self.current_path_id
        
        if not path_id or path_id not in self.exploration_paths:
            return {"status": "error", "message": "探索路径不存在"}
        
        path = self.exploration_paths[path_id]
        
        # 检查是否达到最大步数
        current_depth = max((node.depth for node in path.nodes), default=0)
        if current_depth >= self.max_steps:
            path.status = "completed"
            return {"status": "completed", "message": "达到最大探索深度"}
        
        try:
            # 获取当前层的节点
            current_nodes = [node for node in path.nodes if node.depth == current_depth]
            
            if not current_nodes:
                return {"status": "error", "message": "没有可探索的节点"}
            
            # 为每个当前节点探索邻居
            new_nodes = []
            for node in current_nodes:
                neighbors = self._explore_neighbors(node, path.query)
                new_nodes.extend(neighbors)
            
            # 过滤和排序新节点
            filtered_nodes = self._filter_and_rank_nodes(new_nodes, path.query)
            
            # 限制探索宽度
            if len(filtered_nodes) > self.exploration_width:
                filtered_nodes = filtered_nodes[:self.exploration_width]
            
            # 添加到路径
            for node in filtered_nodes:
                node.depth = current_depth + 1
                path.nodes.append(node)
                self.visited_nodes.add(node.node_id)
            
            path.updated_at = time.time()
            
            print(f"探索步骤完成，新增节点: {len(filtered_nodes)}")
            
            return {
                "status": "continue",
                "new_nodes_count": len(filtered_nodes),
                "current_depth": current_depth + 1,
                "total_nodes": len(path.nodes)
            }
            
        except Exception as e:
            print(f"探索步骤失败: {e}")
            return {"status": "error", "message": str(e)}
    
    def _explore_neighbors(self, node: ExplorationNode, query: str) -> List[ExplorationNode]:
        """
        探索节点的邻居
        
        参数:
            node: 当前节点
            query: 探索查询
            
        返回:
            List[ExplorationNode]: 邻居节点列表
        """
        try:
            neighbors = []
            
            if node.node_type == "entity":
                # 探索实体的关系和连接
                cypher = """
                MATCH (e:__Entity__ {id: $entity_id})-[r:RELATED]->(neighbor:__Entity__)
                WHERE neighbor.id <> $entity_id
                RETURN neighbor.id as id, neighbor.description as description, r.rank as relevance
                ORDER BY r.rank DESC
                LIMIT $limit
                """
                
                results = self.graph_query_func(cypher, {
                    "entity_id": node.node_id,
                    "limit": self.exploration_width * 2
                })
                
                for result in results:
                    if result["id"] not in self.visited_nodes:
                        neighbor = ExplorationNode(
                            node_id=result["id"],
                            node_type="entity",
                            content=result.get("description", result["id"]),
                            relevance_score=float(result.get("relevance", 0.5)),
                            parent_id=node.node_id
                        )
                        neighbors.append(neighbor)
                        
                        # 更新父节点的子节点列表
                        if neighbor.node_id not in node.children_ids:
                            node.children_ids.append(neighbor.node_id)
            
            elif node.node_type == "document":
                # 探索文档的实体
                cypher = """
                MATCH (d:__Document__ {id: $doc_id})<-[:PART_OF]-(chunk:__Chunk__)
                MATCH (chunk)-[:HAS_ENTITY]->(e:__Entity__)
                RETURN DISTINCT e.id as id, e.description as description, e.rank as relevance
                ORDER BY e.rank DESC
                LIMIT $limit
                """
                
                results = self.graph_query_func(cypher, {
                    "doc_id": node.node_id,
                    "limit": self.exploration_width
                })
                
                for result in results:
                    if result["id"] not in self.visited_nodes:
                        neighbor = ExplorationNode(
                            node_id=result["id"],
                            node_type="entity",
                            content=result.get("description", result["id"]),
                            relevance_score=float(result.get("relevance", 0.5)),
                            parent_id=node.node_id
                        )
                        neighbors.append(neighbor)
            
            return neighbors
            
        except Exception as e:
            print(f"探索邻居失败: {e}")
            return []
    
    def _filter_and_rank_nodes(self, nodes: List[ExplorationNode], query: str) -> List[ExplorationNode]:
        """
        过滤和排序节点
        
        参数:
            nodes: 节点列表
            query: 查询字符串
            
        返回:
            List[ExplorationNode]: 过滤排序后的节点列表
        """
        try:
            # 过滤低相关性节点
            filtered_nodes = [
                node for node in nodes 
                if node.relevance_score >= self.relevance_threshold
            ]
            
            # 计算查询相关性（简单的关键词匹配）
            query_lower = query.lower()
            for node in filtered_nodes:
                content_lower = node.content.lower()
                
                # 简单的关键词匹配评分
                keyword_score = 0.0
                query_words = query_lower.split()
                for word in query_words:
                    if word in content_lower:
                        keyword_score += 1.0
                
                # 结合原始相关性和查询相关性
                node.relevance_score = (
                    node.relevance_score * 0.7 + 
                    (keyword_score / len(query_words)) * 0.3
                )
                
                # 应用深度衰减
                node.relevance_score *= (self.decay_factor ** node.depth)
            
            # 按相关性排序
            filtered_nodes.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return filtered_nodes
            
        except Exception as e:
            print(f"节点过滤排序失败: {e}")
            return nodes
    
    def get_exploration_summary(self, path_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取探索摘要
        
        参数:
            path_id: 探索路径ID
            
        返回:
            Dict: 探索摘要
        """
        if path_id is None:
            path_id = self.current_path_id
        
        if not path_id or path_id not in self.exploration_paths:
            return {"error": "探索路径不存在"}
        
        path = self.exploration_paths[path_id]
        
        try:
            # 按深度统计节点
            nodes_by_depth = {}
            nodes_by_type = {}
            total_score = 0.0
            
            for node in path.nodes:
                # 按深度统计
                depth = node.depth
                if depth not in nodes_by_depth:
                    nodes_by_depth[depth] = 0
                nodes_by_depth[depth] += 1
                
                # 按类型统计
                node_type = node.node_type
                if node_type not in nodes_by_type:
                    nodes_by_type[node_type] = 0
                nodes_by_type[node_type] += 1
                
                # 累计分数
                total_score += node.relevance_score
            
            return {
                "path_id": path_id,
                "query": path.query,
                "status": path.status,
                "total_nodes": len(path.nodes),
                "max_depth": max((node.depth for node in path.nodes), default=0),
                "nodes_by_depth": nodes_by_depth,
                "nodes_by_type": nodes_by_type,
                "total_score": total_score,
                "avg_score": total_score / len(path.nodes) if path.nodes else 0.0,
                "created_at": path.created_at,
                "updated_at": path.updated_at,
                "duration": path.updated_at - path.created_at
            }
            
        except Exception as e:
            print(f"获取探索摘要失败: {e}")
            return {"error": str(e)}
    
    def get_exploration_path(self, path_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取探索路径详情
        
        参数:
            path_id: 探索路径ID
            
        返回:
            List[Dict]: 路径节点列表
        """
        if path_id is None:
            path_id = self.current_path_id
        
        if not path_id or path_id not in self.exploration_paths:
            return []
        
        path = self.exploration_paths[path_id]
        
        return [
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "content": node.content,
                "relevance_score": node.relevance_score,
                "depth": node.depth,
                "parent_id": node.parent_id,
                "children_count": len(node.children_ids),
                "timestamp": node.timestamp
            }
            for node in path.nodes
        ]
    
    def backtrack(self, path_id: Optional[str] = None, target_depth: int = 0) -> bool:
        """
        回溯到指定深度
        
        参数:
            path_id: 探索路径ID
            target_depth: 目标深度
            
        返回:
            bool: 是否成功
        """
        if not self.enable_backtracking:
            print("回溯功能未启用")
            return False
        
        if path_id is None:
            path_id = self.current_path_id
        
        if not path_id or path_id not in self.exploration_paths:
            return False
        
        path = self.exploration_paths[path_id]
        
        try:
            # 移除深度大于目标深度的节点
            nodes_to_keep = [node for node in path.nodes if node.depth <= target_depth]
            removed_nodes = [node for node in path.nodes if node.depth > target_depth]
            
            # 更新路径
            path.nodes = nodes_to_keep
            path.updated_at = time.time()
            
            # 更新访问记录
            for node in removed_nodes:
                self.visited_nodes.discard(node.node_id)
            
            print(f"回溯到深度 {target_depth}，移除节点: {len(removed_nodes)}")
            return True
            
        except Exception as e:
            print(f"回溯失败: {e}")
            return False
    
    def clear_path(self, path_id: Optional[str] = None):
        """
        清空探索路径
        
        参数:
            path_id: 探索路径ID
        """
        if path_id is None:
            path_id = self.current_path_id
        
        if path_id and path_id in self.exploration_paths:
            del self.exploration_paths[path_id]
            
            if path_id == self.current_path_id:
                self.current_path_id = None
            
            print(f"清空探索路径: {path_id}")
    
    def close(self):
        """关闭链式探索"""
        self.exploration_paths.clear()
        self.current_path_id = None
        self.visited_nodes.clear()
        print("链式探索已关闭")
