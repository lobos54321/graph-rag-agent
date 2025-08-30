"""
内存图数据库实现 - Neo4j的免费替代方案
使用NetworkX + JSON文件存储实现图数据库功能
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
from datetime import datetime


class MemoryGraphDB:
    """基于NetworkX的内存图数据库，支持文件持久化"""
    
    def __init__(self, data_dir: str = "./graph_data"):
        """
        初始化内存图数据库
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建有向图
        self.graph = nx.DiGraph()
        
        # 存储文件路径
        self.graph_file = self.data_dir / "graph.pickle"
        self.metadata_file = self.data_dir / "metadata.json"
        
        # 加载已存在的数据
        self.load()
    
    def save(self):
        """保存图数据到文件"""
        try:
            # 保存图结构
            with open(self.graph_file, 'wb') as f:
                pickle.dump(self.graph, f)
            
            # 保存元数据
            metadata = {
                "nodes_count": self.graph.number_of_nodes(),
                "edges_count": self.graph.number_of_edges(),
                "last_saved": datetime.now().isoformat()
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"保存图数据失败: {e}")
    
    def load(self):
        """从文件加载图数据"""
        try:
            if self.graph_file.exists():
                with open(self.graph_file, 'rb') as f:
                    self.graph = pickle.load(f)
                print(f"加载图数据: {self.graph.number_of_nodes()} 节点, {self.graph.number_of_edges()} 边")
        except Exception as e:
            print(f"加载图数据失败: {e}")
            self.graph = nx.DiGraph()
    
    def create_node(self, node_id: str, properties: Dict[str, Any]):
        """创建节点"""
        self.graph.add_node(node_id, **properties)
        self.save()
    
    def create_relationship(self, from_node: str, to_node: str, rel_type: str, properties: Dict[str, Any] = None):
        """创建关系"""
        if properties is None:
            properties = {}
        properties['type'] = rel_type
        
        self.graph.add_edge(from_node, to_node, **properties)
        self.save()
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取节点"""
        if self.graph.has_node(node_id):
            return dict(self.graph.nodes[node_id])
        return None
    
    def find_nodes(self, label: str = None, properties: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """查找节点"""
        results = []
        
        for node_id, attrs in self.graph.nodes(data=True):
            match = True
            
            # 检查标签
            if label and attrs.get('label') != label:
                match = False
            
            # 检查属性
            if properties:
                for key, value in properties.items():
                    if attrs.get(key) != value:
                        match = False
                        break
            
            if match:
                result = dict(attrs)
                result['id'] = node_id
                results.append(result)
        
        return results
    
    def find_relationships(self, from_node: str = None, to_node: str = None, rel_type: str = None) -> List[Dict[str, Any]]:
        """查找关系"""
        results = []
        
        edges = self.graph.edges(data=True)
        if from_node:
            edges = [(u, v, d) for u, v, d in edges if u == from_node]
        if to_node:
            edges = [(u, v, d) for u, v, d in edges if v == to_node]
        if rel_type:
            edges = [(u, v, d) for u, v, d in edges if d.get('type') == rel_type]
        
        for from_id, to_id, attrs in edges:
            result = dict(attrs)
            result.update({
                'from': from_id,
                'to': to_id
            })
            results.append(result)
        
        return results
    
    def get_neighbors(self, node_id: str, direction: str = 'both') -> List[str]:
        """获取邻居节点"""
        if not self.graph.has_node(node_id):
            return []
        
        if direction == 'out':
            return list(self.graph.successors(node_id))
        elif direction == 'in':
            return list(self.graph.predecessors(node_id))
        else:  # both
            return list(set(list(self.graph.successors(node_id)) + list(self.graph.predecessors(node_id))))
    
    def delete_node(self, node_id: str):
        """删除节点"""
        if self.graph.has_node(node_id):
            self.graph.remove_node(node_id)
            self.save()
    
    def delete_relationship(self, from_node: str, to_node: str):
        """删除关系"""
        if self.graph.has_edge(from_node, to_node):
            self.graph.remove_edge(from_node, to_node)
            self.save()
    
    def clear(self):
        """清空所有数据"""
        self.graph.clear()
        self.save()
    
    def execute_cypher_like_query(self, query_type: str, **kwargs) -> List[Dict[str, Any]]:
        """执行类Cypher查询"""
        if query_type == "MATCH_NODES":
            return self.find_nodes(kwargs.get('label'), kwargs.get('properties'))
        elif query_type == "MATCH_RELATIONSHIPS":
            return self.find_relationships(
                kwargs.get('from_node'), 
                kwargs.get('to_node'), 
                kwargs.get('rel_type')
            )
        elif query_type == "GET_NEIGHBORS":
            neighbors = self.get_neighbors(kwargs.get('node_id'), kwargs.get('direction', 'both'))
            return [{'id': n, **self.graph.nodes[n]} for n in neighbors if self.graph.has_node(n)]
        else:
            return []
    
    def get_stats(self) -> Dict[str, int]:
        """获取图统计信息"""
        return {
            "nodes_count": self.graph.number_of_nodes(),
            "edges_count": self.graph.number_of_edges(),
            "connected_components": nx.number_weakly_connected_components(self.graph)
        }


# 全局实例
memory_graph_db = None

def get_memory_graph_db() -> MemoryGraphDB:
    """获取内存图数据库实例"""
    global memory_graph_db
    if memory_graph_db is None:
        data_dir = os.getenv('GRAPH_DATA_DIR', './graph_data')
        memory_graph_db = MemoryGraphDB(data_dir)
    return memory_graph_db