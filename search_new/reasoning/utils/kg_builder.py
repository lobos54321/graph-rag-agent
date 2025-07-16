"""
知识图谱构建器

从文本中构建知识图谱，支持实体识别、关系抽取和图谱更新
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import time
import logging
from dataclasses import dataclass, field

from search_new.reasoning.utils.nlp_utils import extract_entities, clean_text
from search_new.reasoning.utils.prompts import get_prompt

logger = logging.getLogger(__name__)


@dataclass
class KGEntity:
    """知识图谱实体数据类"""
    entity_id: str
    entity_type: str
    name: str
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class KGRelation:
    """知识图谱关系数据类"""
    relation_id: str
    source_entity: str
    target_entity: str
    relation_type: str
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class KnowledgeGraph:
    """知识图谱数据类"""
    graph_id: str
    entities: Dict[str, KGEntity] = field(default_factory=dict)
    relations: Dict[str, KGRelation] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class KnowledgeGraphBuilder:
    """
    知识图谱构建器：从文本构建和更新知识图谱
    
    主要功能：
    1. 实体识别和抽取
    2. 关系识别和抽取
    3. 知识图谱构建
    4. 图谱合并和更新
    """
    
    def __init__(self, llm=None):
        """
        初始化知识图谱构建器
        
        参数:
            llm: 大语言模型实例，用于实体和关系抽取
        """
        self.llm = llm
        
        # 构建配置
        self.entity_confidence_threshold = 0.7
        self.relation_confidence_threshold = 0.6
        self.max_entities_per_text = 50
        self.max_relations_per_text = 100
        
        # 知识图谱存储
        self.knowledge_graphs: Dict[str, KnowledgeGraph] = {}
        self.current_graph_id: Optional[str] = None
        
        # 实体和关系缓存
        self.entity_cache: Dict[str, KGEntity] = {}
        self.relation_cache: Dict[str, KGRelation] = {}
        
        logger.info("知识图谱构建器初始化完成")
    
    def create_knowledge_graph(self, graph_id: Optional[str] = None) -> str:
        """
        创建新的知识图谱
        
        参数:
            graph_id: 图谱ID，如果为None则自动生成
            
        返回:
            str: 图谱ID
        """
        if graph_id is None:
            graph_id = f"kg_{int(time.time() * 1000)}"
        
        kg = KnowledgeGraph(graph_id=graph_id)
        self.knowledge_graphs[graph_id] = kg
        self.current_graph_id = graph_id
        
        logger.info(f"创建知识图谱: {graph_id}")
        return graph_id
    
    def extract_entities_from_text(self, text: str, source: str = "") -> List[KGEntity]:
        """
        从文本中抽取实体
        
        参数:
            text: 输入文本
            source: 来源标识
            
        返回:
            List[KGEntity]: 抽取的实体列表
        """
        try:
            entities = []
            
            # 使用NLP工具进行基础实体识别
            basic_entities = extract_entities(text)
            
            # 转换为KGEntity格式
            for entity_type, entity_list in basic_entities.items():
                for entity_name in entity_list:
                    entity_id = f"{entity_type}_{hash(entity_name) % 10000}"
                    
                    entity = KGEntity(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        name=entity_name,
                        description=f"{entity_type}: {entity_name}",
                        confidence=0.8,  # 基础识别的置信度
                        source=source
                    )
                    entities.append(entity)
            
            # 如果有LLM，使用更高级的实体识别
            if self.llm:
                llm_entities = self._extract_entities_with_llm(text, source)
                entities.extend(llm_entities)
            
            # 去重和过滤
            entities = self._deduplicate_entities(entities)
            entities = [e for e in entities if e.confidence >= self.entity_confidence_threshold]
            
            # 限制数量
            if len(entities) > self.max_entities_per_text:
                entities = sorted(entities, key=lambda x: x.confidence, reverse=True)
                entities = entities[:self.max_entities_per_text]
            
            logger.info(f"从文本中抽取实体: {len(entities)} 个")
            return entities
            
        except Exception as e:
            logger.error(f"实体抽取失败: {e}")
            return []
    
    def _extract_entities_with_llm(self, text: str, source: str) -> List[KGEntity]:
        """使用LLM进行实体抽取"""
        try:
            prompt = get_prompt("entity_recognition", text=text)
            if not prompt:
                return []
            
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            
            # 解析LLM响应（简化版本）
            entities = []
            lines = response.content.split('\n') if hasattr(response, 'content') else str(response).split('\n')
            
            current_type = None
            for line in lines:
                line = line.strip()
                if line.endswith('：'):
                    current_type = line[:-1]
                elif line and current_type:
                    entity_names = [name.strip() for name in line.split('、') if name.strip()]
                    for name in entity_names:
                        entity_id = f"{current_type}_{hash(name) % 10000}"
                        entity = KGEntity(
                            entity_id=entity_id,
                            entity_type=current_type,
                            name=name,
                            description=f"{current_type}: {name}",
                            confidence=0.9,  # LLM识别的置信度更高
                            source=source
                        )
                        entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"LLM实体抽取失败: {e}")
            return []
    
    def extract_relations_from_text(self, text: str, entities: List[KGEntity], 
                                   source: str = "") -> List[KGRelation]:
        """
        从文本中抽取关系
        
        参数:
            text: 输入文本
            entities: 已识别的实体列表
            source: 来源标识
            
        返回:
            List[KGRelation]: 抽取的关系列表
        """
        try:
            relations = []
            
            if not entities or len(entities) < 2:
                return relations
            
            # 使用LLM进行关系抽取
            if self.llm:
                entity_names = [entity.name for entity in entities]
                relations = self._extract_relations_with_llm(text, entity_names, source)
            
            # 过滤和验证关系
            valid_relations = []
            entity_names_set = {entity.name for entity in entities}
            
            for relation in relations:
                # 检查关系的实体是否存在
                if (relation.source_entity in entity_names_set and 
                    relation.target_entity in entity_names_set and
                    relation.confidence >= self.relation_confidence_threshold):
                    valid_relations.append(relation)
            
            # 限制数量
            if len(valid_relations) > self.max_relations_per_text:
                valid_relations = sorted(valid_relations, key=lambda x: x.confidence, reverse=True)
                valid_relations = valid_relations[:self.max_relations_per_text]
            
            logger.info(f"从文本中抽取关系: {len(valid_relations)} 个")
            return valid_relations
            
        except Exception as e:
            logger.error(f"关系抽取失败: {e}")
            return []
    
    def _extract_relations_with_llm(self, text: str, entity_names: List[str], 
                                   source: str) -> List[KGRelation]:
        """使用LLM进行关系抽取"""
        try:
            prompt = get_prompt("relation_extraction", 
                              text=text, 
                              entities=", ".join(entity_names))
            if not prompt:
                return []
            
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            
            # 解析LLM响应
            relations = []
            lines = response.content.split('\n') if hasattr(response, 'content') else str(response).split('\n')
            
            for line in lines:
                line = line.strip()
                if ' - ' in line:
                    parts = line.split(' - ')
                    if len(parts) == 3:
                        source_entity, relation_type, target_entity = parts
                        
                        relation_id = f"rel_{hash(line) % 10000}"
                        relation = KGRelation(
                            relation_id=relation_id,
                            source_entity=source_entity.strip(),
                            target_entity=target_entity.strip(),
                            relation_type=relation_type.strip(),
                            description=line,
                            confidence=0.8,
                            source=source
                        )
                        relations.append(relation)
            
            return relations
            
        except Exception as e:
            logger.error(f"LLM关系抽取失败: {e}")
            return []
    
    def build_knowledge_graph_from_text(self, text: str, source: str = "", 
                                       graph_id: Optional[str] = None) -> str:
        """
        从文本构建知识图谱
        
        参数:
            text: 输入文本
            source: 来源标识
            graph_id: 目标图谱ID，如果为None则使用当前图谱
            
        返回:
            str: 图谱ID
        """
        try:
            # 确定目标图谱
            if graph_id is None:
                if self.current_graph_id is None:
                    graph_id = self.create_knowledge_graph()
                else:
                    graph_id = self.current_graph_id
            
            if graph_id not in self.knowledge_graphs:
                self.create_knowledge_graph(graph_id)
            
            kg = self.knowledge_graphs[graph_id]
            
            # 抽取实体
            entities = self.extract_entities_from_text(text, source)
            
            # 抽取关系
            relations = self.extract_relations_from_text(text, entities, source)
            
            # 添加到知识图谱
            for entity in entities:
                kg.entities[entity.entity_id] = entity
            
            for relation in relations:
                kg.relations[relation.relation_id] = relation
            
            kg.updated_at = time.time()
            
            logger.info(f"构建知识图谱完成: {len(entities)} 个实体, {len(relations)} 个关系")
            return graph_id
            
        except Exception as e:
            logger.error(f"知识图谱构建失败: {e}")
            return ""
    
    def _deduplicate_entities(self, entities: List[KGEntity]) -> List[KGEntity]:
        """去重实体"""
        try:
            seen_names = set()
            unique_entities = []
            
            for entity in entities:
                name_key = f"{entity.entity_type}:{entity.name.lower()}"
                if name_key not in seen_names:
                    seen_names.add(name_key)
                    unique_entities.append(entity)
                else:
                    # 如果重复，保留置信度更高的
                    for i, existing in enumerate(unique_entities):
                        if (existing.entity_type == entity.entity_type and 
                            existing.name.lower() == entity.name.lower()):
                            if entity.confidence > existing.confidence:
                                unique_entities[i] = entity
                            break
            
            return unique_entities
            
        except Exception as e:
            logger.error(f"实体去重失败: {e}")
            return entities
    
    def get_knowledge_graph_summary(self, graph_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取知识图谱摘要
        
        参数:
            graph_id: 图谱ID
            
        返回:
            Dict: 图谱摘要
        """
        if graph_id is None:
            graph_id = self.current_graph_id
        
        if not graph_id or graph_id not in self.knowledge_graphs:
            return {"error": "知识图谱不存在"}
        
        kg = self.knowledge_graphs[graph_id]
        
        try:
            # 统计实体类型
            entity_types = {}
            for entity in kg.entities.values():
                entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1
            
            # 统计关系类型
            relation_types = {}
            for relation in kg.relations.values():
                relation_types[relation.relation_type] = relation_types.get(relation.relation_type, 0) + 1
            
            return {
                "graph_id": graph_id,
                "entities_count": len(kg.entities),
                "relations_count": len(kg.relations),
                "entity_types": entity_types,
                "relation_types": relation_types,
                "created_at": kg.created_at,
                "updated_at": kg.updated_at
            }
            
        except Exception as e:
            logger.error(f"获取图谱摘要失败: {e}")
            return {"error": str(e)}
    
    def merge_knowledge_graphs(self, source_graph_id: str, target_graph_id: str) -> bool:
        """
        合并知识图谱
        
        参数:
            source_graph_id: 源图谱ID
            target_graph_id: 目标图谱ID
            
        返回:
            bool: 是否成功
        """
        try:
            if (source_graph_id not in self.knowledge_graphs or 
                target_graph_id not in self.knowledge_graphs):
                logger.error("源图谱或目标图谱不存在")
                return False
            
            source_kg = self.knowledge_graphs[source_graph_id]
            target_kg = self.knowledge_graphs[target_graph_id]
            
            # 合并实体
            for entity_id, entity in source_kg.entities.items():
                if entity_id not in target_kg.entities:
                    target_kg.entities[entity_id] = entity
                else:
                    # 如果实体已存在，保留置信度更高的
                    if entity.confidence > target_kg.entities[entity_id].confidence:
                        target_kg.entities[entity_id] = entity
            
            # 合并关系
            for relation_id, relation in source_kg.relations.items():
                if relation_id not in target_kg.relations:
                    target_kg.relations[relation_id] = relation
                else:
                    # 如果关系已存在，保留置信度更高的
                    if relation.confidence > target_kg.relations[relation_id].confidence:
                        target_kg.relations[relation_id] = relation
            
            target_kg.updated_at = time.time()
            
            logger.info(f"知识图谱合并完成: {source_graph_id} -> {target_graph_id}")
            return True
            
        except Exception as e:
            logger.error(f"知识图谱合并失败: {e}")
            return False
    
    def clear_knowledge_graph(self, graph_id: Optional[str] = None):
        """
        清空知识图谱
        
        参数:
            graph_id: 图谱ID
        """
        if graph_id is None:
            graph_id = self.current_graph_id
        
        if graph_id and graph_id in self.knowledge_graphs:
            del self.knowledge_graphs[graph_id]
            
            if graph_id == self.current_graph_id:
                self.current_graph_id = None
            
            logger.info(f"清空知识图谱: {graph_id}")
    
    def close(self):
        """关闭知识图谱构建器"""
        self.knowledge_graphs.clear()
        self.entity_cache.clear()
        self.relation_cache.clear()
        self.current_graph_id = None
        logger.info("知识图谱构建器已关闭")
