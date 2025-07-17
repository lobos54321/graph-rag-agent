"""
向量工具模块

提供向量相似度计算、排序和过滤等功能
"""

import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple

class VectorUtils:
    """向量搜索和相似度计算的统一工具类"""
    
    @staticmethod
    def cosine_similarity(vec1: Union[List[float], np.ndarray], 
                         vec2: Union[List[float], np.ndarray]) -> float:
        """
        计算两个向量的余弦相似度
        
        参数:
            vec1: 第一个向量
            vec2: 第二个向量
            
        返回:
            float: 相似度值 (0-1)
        """
        try:
            # 确保向量是numpy数组
            if not isinstance(vec1, np.ndarray):
                vec1 = np.array(vec1)
            if not isinstance(vec2, np.ndarray):
                vec2 = np.array(vec2)
                
            # 检查向量维度
            if vec1.shape != vec2.shape:
                print(f"向量维度不匹配: {vec1.shape} vs {vec2.shape}")
                return 0.0
                
            # 计算余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm_a = np.linalg.norm(vec1)
            norm_b = np.linalg.norm(vec2)
            
            # 避免被零除
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            similarity = dot_product / (norm_a * norm_b)
            
            # 确保结果在有效范围内
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            print(f"计算余弦相似度失败: {e}")
            return 0.0
    
    @staticmethod
    def euclidean_distance(vec1: Union[List[float], np.ndarray], 
                          vec2: Union[List[float], np.ndarray]) -> float:
        """
        计算两个向量的欧几里得距离
        
        参数:
            vec1: 第一个向量
            vec2: 第二个向量
            
        返回:
            float: 距离值
        """
        try:
            if not isinstance(vec1, np.ndarray):
                vec1 = np.array(vec1)
            if not isinstance(vec2, np.ndarray):
                vec2 = np.array(vec2)
                
            if vec1.shape != vec2.shape:
                print(f"向量维度不匹配: {vec1.shape} vs {vec2.shape}")
                return float('inf')
                
            return float(np.linalg.norm(vec1 - vec2))
            
        except Exception as e:
            print(f"计算欧几里得距离失败: {e}")
            return float('inf')
    
    @staticmethod
    def manhattan_distance(vec1: Union[List[float], np.ndarray], 
                          vec2: Union[List[float], np.ndarray]) -> float:
        """
        计算两个向量的曼哈顿距离
        
        参数:
            vec1: 第一个向量
            vec2: 第二个向量
            
        返回:
            float: 距离值
        """
        try:
            if not isinstance(vec1, np.ndarray):
                vec1 = np.array(vec1)
            if not isinstance(vec2, np.ndarray):
                vec2 = np.array(vec2)
                
            if vec1.shape != vec2.shape:
                print(f"向量维度不匹配: {vec1.shape} vs {vec2.shape}")
                return float('inf')
                
            return float(np.sum(np.abs(vec1 - vec2)))
            
        except Exception as e:
            print(f"计算曼哈顿距离失败: {e}")
            return float('inf')
    
    @staticmethod
    def rank_by_similarity(query_embedding: List[float], 
                          candidates: List[Dict[str, Any]], 
                          embedding_field: str = "embedding",
                          similarity_metric: str = "cosine",
                          top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        对候选项按与查询向量的相似度排序
        
        参数:
            query_embedding: 查询向量
            candidates: 候选项列表，每项都包含embedding_field指定的字段
            embedding_field: 包含嵌入向量的字段名
            similarity_metric: 相似度度量方法 ("cosine", "euclidean", "manhattan")
            top_k: 返回的最大结果数，None表示返回所有结果
            
        返回:
            按相似度排序的候选项列表，每项增加"score"字段表示相似度
        """
        if not candidates:
            return []
            
        scored_items = []
        
        for item in candidates:
            if embedding_field not in item or not item[embedding_field]:
                continue
                
            try:
                # 计算相似度
                if similarity_metric == "cosine":
                    score = VectorUtils.cosine_similarity(query_embedding, item[embedding_field])
                elif similarity_metric == "euclidean":
                    # 转换距离为相似度 (距离越小，相似度越高)
                    distance = VectorUtils.euclidean_distance(query_embedding, item[embedding_field])
                    score = 1.0 / (1.0 + distance) if distance != float('inf') else 0.0
                elif similarity_metric == "manhattan":
                    # 转换距离为相似度
                    distance = VectorUtils.manhattan_distance(query_embedding, item[embedding_field])
                    score = 1.0 / (1.0 + distance) if distance != float('inf') else 0.0
                else:
                    print(f"未知的相似度度量方法: {similarity_metric}，使用余弦相似度")
                    score = VectorUtils.cosine_similarity(query_embedding, item[embedding_field])
                
                # 复制item并添加分数
                scored_item = item.copy()
                scored_item["score"] = score
                scored_items.append(scored_item)
                
            except Exception as e:
                print(f"计算相似度失败: {e}")
                continue
        
        # 按分数降序排序
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        
        # 返回top_k结果
        if top_k is not None:
            return scored_items[:top_k]
        return scored_items
    
    @staticmethod
    def filter_by_threshold(scored_items: List[Dict[str, Any]], 
                           threshold: float = 0.0,
                           score_field: str = "score") -> List[Dict[str, Any]]:
        """
        根据阈值过滤结果
        
        参数:
            scored_items: 包含分数的项目列表
            threshold: 最小分数阈值
            score_field: 分数字段名
            
        返回:
            过滤后的项目列表
        """
        return [
            item for item in scored_items 
            if score_field in item and item[score_field] >= threshold
        ]
    
    @staticmethod
    def normalize_scores(scored_items: List[Dict[str, Any]], 
                        score_field: str = "score") -> List[Dict[str, Any]]:
        """
        归一化分数到[0,1]范围
        
        参数:
            scored_items: 包含分数的项目列表
            score_field: 分数字段名
            
        返回:
            归一化后的项目列表
        """
        if not scored_items:
            return scored_items
            
        # 获取所有分数
        scores = [item.get(score_field, 0) for item in scored_items]
        min_score = min(scores)
        max_score = max(scores)
        
        # 避免除零
        if max_score == min_score:
            for item in scored_items:
                item[score_field] = 1.0
            return scored_items
        
        # 归一化
        for item in scored_items:
            original_score = item.get(score_field, 0)
            normalized_score = (original_score - min_score) / (max_score - min_score)
            item[score_field] = normalized_score
            
        return scored_items
    
    @staticmethod
    def batch_similarity(query_embedding: List[float],
                        candidate_embeddings: List[List[float]],
                        similarity_metric: str = "cosine") -> List[float]:
        """
        批量计算相似度
        
        参数:
            query_embedding: 查询向量
            candidate_embeddings: 候选向量列表
            similarity_metric: 相似度度量方法
            
        返回:
            相似度分数列表
        """
        scores = []
        
        for candidate_embedding in candidate_embeddings:
            if similarity_metric == "cosine":
                score = VectorUtils.cosine_similarity(query_embedding, candidate_embedding)
            elif similarity_metric == "euclidean":
                distance = VectorUtils.euclidean_distance(query_embedding, candidate_embedding)
                score = 1.0 / (1.0 + distance) if distance != float('inf') else 0.0
            elif similarity_metric == "manhattan":
                distance = VectorUtils.manhattan_distance(query_embedding, candidate_embedding)
                score = 1.0 / (1.0 + distance) if distance != float('inf') else 0.0
            else:
                score = VectorUtils.cosine_similarity(query_embedding, candidate_embedding)
                
            scores.append(score)
            
        return scores
    
    @staticmethod
    def find_most_similar(query_embedding: List[float],
                         candidates: List[Dict[str, Any]],
                         embedding_field: str = "embedding",
                         similarity_metric: str = "cosine") -> Optional[Tuple[Dict[str, Any], float]]:
        """
        找到最相似的候选项
        
        参数:
            query_embedding: 查询向量
            candidates: 候选项列表
            embedding_field: 嵌入向量字段名
            similarity_metric: 相似度度量方法
            
        返回:
            最相似的候选项和其相似度分数，如果没有找到则返回None
        """
        ranked_candidates = VectorUtils.rank_by_similarity(
            query_embedding, candidates, embedding_field, similarity_metric, top_k=1
        )
        
        if ranked_candidates:
            best_candidate = ranked_candidates[0]
            return best_candidate, best_candidate["score"]
        
        return None
