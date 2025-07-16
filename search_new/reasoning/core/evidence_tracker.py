"""
证据跟踪器

收集和管理推理过程中使用的证据，提高透明度和可解释性
"""

from typing import Dict, List, Any, Optional, Tuple
import time
import hashlib
import logging
from dataclasses import dataclass, field

from search_new.config import get_reasoning_config

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    """证据数据类"""
    evidence_id: str
    source_id: str
    content: str
    source_type: str  # document, entity, relationship, community
    relevance_score: float = 0.0
    confidence_score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningStep:
    """推理步骤数据类"""
    step_id: str
    description: str
    evidence_ids: List[str] = field(default_factory=list)
    reasoning_type: str = "general"  # general, deduction, induction, abduction
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceChain:
    """证据链数据类"""
    chain_id: str
    query: str
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    evidence_items: Dict[str, Evidence] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class EvidenceTracker:
    """
    证据跟踪器：收集和管理推理过程中的证据
    
    主要功能：
    1. 收集证据项
    2. 管理推理步骤
    3. 构建证据链
    4. 验证证据质量
    """
    
    def __init__(self, max_evidence_items: Optional[int] = None):
        """
        初始化证据跟踪器
        
        参数:
            max_evidence_items: 最大证据项数量
        """
        self.config = get_reasoning_config()
        
        # 配置参数
        self.max_evidence_items = max_evidence_items or self.config.evidence.max_evidence_per_step
        self.relevance_threshold = self.config.evidence.evidence_relevance_threshold
        self.enable_validation = self.config.evidence.enable_evidence_validation
        self.enable_deduplication = self.config.evidence.evidence_deduplication
        
        # 证据链管理
        self.evidence_chains: Dict[str, EvidenceChain] = {}
        self.current_chain_id: Optional[str] = None
        
        logger.info(f"证据跟踪器初始化完成，最大证据项: {self.max_evidence_items}")
    
    def create_evidence_chain(self, query: str) -> str:
        """
        创建新的证据链
        
        参数:
            query: 查询字符串
            
        返回:
            str: 证据链ID
        """
        chain_id = f"chain_{int(time.time() * 1000)}"
        
        chain = EvidenceChain(
            chain_id=chain_id,
            query=query
        )
        
        self.evidence_chains[chain_id] = chain
        self.current_chain_id = chain_id
        
        logger.info(f"创建证据链: {chain_id}")
        return chain_id
    
    def add_evidence(self, source_id: str, content: str, source_type: str,
                    step_id: Optional[str] = None, relevance_score: float = 0.0,
                    confidence_score: float = 0.0, metadata: Optional[Dict] = None) -> str:
        """
        添加证据项
        
        参数:
            source_id: 来源ID
            content: 证据内容
            source_type: 来源类型
            step_id: 步骤ID
            relevance_score: 相关性分数
            confidence_score: 置信度分数
            metadata: 元数据
            
        返回:
            str: 证据ID
        """
        if not self.current_chain_id:
            logger.warning("没有活跃的证据链")
            return ""
        
        chain = self.evidence_chains[self.current_chain_id]
        
        # 生成证据ID
        evidence_id = hashlib.md5(f"{source_id}:{content[:50]}".encode()).hexdigest()[:10]
        
        # 检查重复（如果启用去重）
        if self.enable_deduplication and evidence_id in chain.evidence_items:
            logger.debug(f"证据已存在，跳过: {evidence_id}")
            return evidence_id
        
        # 检查相关性阈值
        if relevance_score < self.relevance_threshold:
            logger.debug(f"证据相关性过低，跳过: {relevance_score}")
            return ""
        
        # 检查证据数量限制
        if len(chain.evidence_items) >= self.max_evidence_items:
            logger.warning("证据项数量已达上限")
            # 可以选择移除最旧的证据或相关性最低的证据
            self._remove_least_relevant_evidence(chain)
        
        # 创建证据记录
        evidence = Evidence(
            evidence_id=evidence_id,
            source_id=source_id,
            content=content,
            source_type=source_type,
            relevance_score=relevance_score,
            confidence_score=confidence_score,
            metadata=metadata or {}
        )
        
        # 验证证据（如果启用）
        if self.enable_validation:
            if not self._validate_evidence(evidence):
                logger.warning(f"证据验证失败: {evidence_id}")
                return ""
        
        # 存储证据
        chain.evidence_items[evidence_id] = evidence
        chain.updated_at = time.time()
        
        # 关联到推理步骤
        if step_id:
            self._associate_evidence_to_step(chain, step_id, evidence_id)
        
        logger.debug(f"添加证据: {evidence_id} (类型: {source_type})")
        return evidence_id
    
    def add_reasoning_step(self, description: str, reasoning_type: str = "general",
                          confidence: float = 0.0, metadata: Optional[Dict] = None) -> str:
        """
        添加推理步骤
        
        参数:
            description: 步骤描述
            reasoning_type: 推理类型
            confidence: 置信度
            metadata: 元数据
            
        返回:
            str: 步骤ID
        """
        if not self.current_chain_id:
            logger.warning("没有活跃的证据链")
            return ""
        
        chain = self.evidence_chains[self.current_chain_id]
        
        # 生成步骤ID
        step_id = f"step_{len(chain.reasoning_steps) + 1}"
        
        # 创建推理步骤
        step = ReasoningStep(
            step_id=step_id,
            description=description,
            reasoning_type=reasoning_type,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        # 添加到证据链
        chain.reasoning_steps.append(step)
        chain.updated_at = time.time()
        
        logger.debug(f"添加推理步骤: {step_id} (类型: {reasoning_type})")
        return step_id
    
    def _associate_evidence_to_step(self, chain: EvidenceChain, step_id: str, evidence_id: str):
        """关联证据到推理步骤"""
        for step in chain.reasoning_steps:
            if step.step_id == step_id:
                if evidence_id not in step.evidence_ids:
                    step.evidence_ids.append(evidence_id)
                break
    
    def _validate_evidence(self, evidence: Evidence) -> bool:
        """
        验证证据质量
        
        参数:
            evidence: 证据对象
            
        返回:
            bool: 是否有效
        """
        try:
            # 基本验证
            if not evidence.content or len(evidence.content.strip()) < 10:
                return False
            
            # 置信度验证
            if evidence.confidence_score < 0.5:
                return False
            
            # 内容质量验证（简单检查）
            content = evidence.content.lower()
            if any(word in content for word in ["错误", "不确定", "可能错误"]):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"证据验证失败: {e}")
            return False
    
    def _remove_least_relevant_evidence(self, chain: EvidenceChain):
        """移除相关性最低的证据"""
        if not chain.evidence_items:
            return
        
        # 找到相关性最低的证据
        min_relevance = float('inf')
        least_relevant_id = None
        
        for evidence_id, evidence in chain.evidence_items.items():
            if evidence.relevance_score < min_relevance:
                min_relevance = evidence.relevance_score
                least_relevant_id = evidence_id
        
        # 移除证据
        if least_relevant_id:
            del chain.evidence_items[least_relevant_id]
            
            # 从推理步骤中移除关联
            for step in chain.reasoning_steps:
                if least_relevant_id in step.evidence_ids:
                    step.evidence_ids.remove(least_relevant_id)
            
            logger.debug(f"移除低相关性证据: {least_relevant_id}")
    
    def get_evidence_chain_summary(self, chain_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取证据链摘要
        
        参数:
            chain_id: 证据链ID
            
        返回:
            Dict: 证据链摘要
        """
        if chain_id is None:
            chain_id = self.current_chain_id
        
        if not chain_id or chain_id not in self.evidence_chains:
            return {"error": "证据链不存在"}
        
        chain = self.evidence_chains[chain_id]
        
        # 计算统计信息
        evidence_by_type = {}
        total_confidence = 0.0
        total_relevance = 0.0
        
        for evidence in chain.evidence_items.values():
            evidence_by_type[evidence.source_type] = evidence_by_type.get(evidence.source_type, 0) + 1
            total_confidence += evidence.confidence_score
            total_relevance += evidence.relevance_score
        
        evidence_count = len(chain.evidence_items)
        avg_confidence = total_confidence / evidence_count if evidence_count > 0 else 0.0
        avg_relevance = total_relevance / evidence_count if evidence_count > 0 else 0.0
        
        return {
            "chain_id": chain_id,
            "query": chain.query,
            "evidence_count": evidence_count,
            "reasoning_steps_count": len(chain.reasoning_steps),
            "evidence_by_type": evidence_by_type,
            "avg_confidence": avg_confidence,
            "avg_relevance": avg_relevance,
            "created_at": chain.created_at,
            "updated_at": chain.updated_at,
            "duration": chain.updated_at - chain.created_at
        }
    
    def get_evidence_details(self, evidence_id: str, chain_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        获取证据详情
        
        参数:
            evidence_id: 证据ID
            chain_id: 证据链ID
            
        返回:
            Dict: 证据详情
        """
        if chain_id is None:
            chain_id = self.current_chain_id
        
        if not chain_id or chain_id not in self.evidence_chains:
            return None
        
        chain = self.evidence_chains[chain_id]
        evidence = chain.evidence_items.get(evidence_id)
        
        if not evidence:
            return None
        
        return {
            "evidence_id": evidence.evidence_id,
            "source_id": evidence.source_id,
            "content": evidence.content,
            "source_type": evidence.source_type,
            "relevance_score": evidence.relevance_score,
            "confidence_score": evidence.confidence_score,
            "timestamp": evidence.timestamp,
            "metadata": evidence.metadata
        }
    
    def get_reasoning_trace(self, chain_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取推理轨迹
        
        参数:
            chain_id: 证据链ID
            
        返回:
            List[Dict]: 推理轨迹
        """
        if chain_id is None:
            chain_id = self.current_chain_id
        
        if not chain_id or chain_id not in self.evidence_chains:
            return []
        
        chain = self.evidence_chains[chain_id]
        
        trace = []
        for step in chain.reasoning_steps:
            # 获取步骤相关的证据
            step_evidence = []
            for evidence_id in step.evidence_ids:
                if evidence_id in chain.evidence_items:
                    evidence = chain.evidence_items[evidence_id]
                    step_evidence.append({
                        "evidence_id": evidence_id,
                        "source_type": evidence.source_type,
                        "relevance_score": evidence.relevance_score,
                        "content_preview": evidence.content[:100] + "..." if len(evidence.content) > 100 else evidence.content
                    })
            
            trace.append({
                "step_id": step.step_id,
                "description": step.description,
                "reasoning_type": step.reasoning_type,
                "confidence": step.confidence,
                "evidence_count": len(step_evidence),
                "evidence": step_evidence,
                "timestamp": step.timestamp
            })
        
        return trace
    
    def clear_chain(self, chain_id: Optional[str] = None):
        """
        清空证据链
        
        参数:
            chain_id: 证据链ID
        """
        if chain_id is None:
            chain_id = self.current_chain_id
        
        if chain_id and chain_id in self.evidence_chains:
            del self.evidence_chains[chain_id]
            
            if chain_id == self.current_chain_id:
                self.current_chain_id = None
            
            logger.info(f"清空证据链: {chain_id}")
    
    def close(self):
        """关闭证据跟踪器"""
        self.evidence_chains.clear()
        self.current_chain_id = None
        logger.info("证据跟踪器已关闭")
