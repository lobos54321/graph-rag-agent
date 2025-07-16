"""
推理配置管理模块

管理推理相关的配置参数，包括提示模板、推理策略等
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ThinkingConfig:
    """思考引擎配置"""
    # 思考深度配置
    max_thinking_depth: int = 5
    thinking_timeout: int = 30
    
    # 查询生成配置
    max_queries_per_step: int = 3
    query_diversity_threshold: float = 0.8
    
    # 推理策略
    reasoning_strategy: str = "depth_first"  # depth_first, breadth_first, adaptive
    enable_parallel_thinking: bool = False


@dataclass
class ExplorationConfig:
    """链式探索配置"""
    # 探索参数
    max_exploration_steps: int = 5
    exploration_width: int = 3
    max_neighbors_per_step: int = 10
    
    # 相关性阈值
    relevance_threshold: float = 0.6
    exploration_decay_factor: float = 0.9
    
    # 探索策略
    exploration_strategy: str = "semantic"  # semantic, structural, hybrid
    enable_backtracking: bool = True


@dataclass
class EvidenceConfig:
    """证据链配置"""
    # 证据收集
    max_evidence_per_step: int = 10
    evidence_relevance_threshold: float = 0.7
    
    # 证据验证
    enable_evidence_validation: bool = True
    evidence_confidence_threshold: float = 0.8
    
    # 证据链管理
    max_evidence_chain_length: int = 20
    evidence_deduplication: bool = True


@dataclass
class CommunityConfig:
    """社区感知配置"""
    # 社区检测
    community_detection_algorithm: str = "louvain"  # louvain, leiden, infomap
    min_community_size: int = 3
    
    # 社区搜索
    max_communities_per_query: int = 5
    community_relevance_threshold: float = 0.6
    
    # 社区增强
    enable_community_expansion: bool = True
    expansion_factor: float = 1.5


@dataclass
class ValidationConfig:
    """验证配置"""
    # 答案验证
    enable_answer_validation: bool = True
    validation_threshold: float = 0.8
    
    # 复杂度评估
    enable_complexity_estimation: bool = True
    complexity_threshold: float = 0.7
    
    # 一致性检查
    enable_consistency_check: bool = True
    consistency_threshold: float = 0.9


@dataclass
class PromptConfig:
    """提示模板配置"""
    # 系统提示
    system_prompt_template: str = """你是一个专业的知识分析助手，擅长从复杂信息中提取关键洞察。
请基于提供的信息进行深入分析和推理。"""
    
    # 思考提示
    thinking_prompt_template: str = """基于当前信息，请分析以下问题：
{query}

当前已知信息：
{context}

请进行深入思考并生成下一步的搜索查询。"""
    
    # 探索提示
    exploration_prompt_template: str = """在知识图谱中探索与以下查询相关的信息：
{query}

当前实体：{current_entities}
邻居实体：{neighbors}

请选择最相关的实体进行下一步探索。"""
    
    # 证据提示
    evidence_prompt_template: str = """请评估以下证据与查询的相关性：
查询：{query}
证据：{evidence}

请给出相关性评分（0-1）和理由。"""
    
    # 验证提示
    validation_prompt_template: str = """请验证以下答案的准确性和完整性：
问题：{query}
答案：{answer}
证据：{evidence}

请给出验证结果和改进建议。"""


@dataclass
class ReasoningConfig:
    """推理模块总配置"""
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    community: CommunityConfig = field(default_factory=CommunityConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    
    # 全局推理配置
    enable_reasoning: bool = True
    reasoning_mode: str = "full"  # full, fast, minimal
    max_reasoning_time: int = 300  # 5分钟
    
    def get_strategy_config(self, strategy: str) -> Dict:
        """获取特定策略的配置"""
        strategy_configs = {
            "thinking": {
                "max_depth": self.thinking.max_thinking_depth,
                "timeout": self.thinking.thinking_timeout,
                "strategy": self.thinking.reasoning_strategy,
                "parallel": self.thinking.enable_parallel_thinking
            },
            "exploration": {
                "max_steps": self.exploration.max_exploration_steps,
                "width": self.exploration.exploration_width,
                "threshold": self.exploration.relevance_threshold,
                "strategy": self.exploration.exploration_strategy,
                "backtrack": self.exploration.enable_backtracking
            },
            "evidence": {
                "max_per_step": self.evidence.max_evidence_per_step,
                "threshold": self.evidence.evidence_relevance_threshold,
                "validation": self.evidence.enable_evidence_validation,
                "dedup": self.evidence.evidence_deduplication
            },
            "community": {
                "algorithm": self.community.community_detection_algorithm,
                "min_size": self.community.min_community_size,
                "max_communities": self.community.max_communities_per_query,
                "expansion": self.community.enable_community_expansion
            },
            "validation": {
                "enable": self.validation.enable_answer_validation,
                "threshold": self.validation.validation_threshold,
                "complexity": self.validation.enable_complexity_estimation,
                "consistency": self.validation.enable_consistency_check
            }
        }
        
        return strategy_configs.get(strategy, {})
    
    def get_prompt_template(self, prompt_type: str) -> str:
        """获取指定类型的提示模板"""
        prompt_templates = {
            "system": self.prompts.system_prompt_template,
            "thinking": self.prompts.thinking_prompt_template,
            "exploration": self.prompts.exploration_prompt_template,
            "evidence": self.prompts.evidence_prompt_template,
            "validation": self.prompts.validation_prompt_template
        }
        
        return prompt_templates.get(prompt_type, "")
    
    def validate(self) -> bool:
        """验证推理配置"""
        try:
            # 验证数值范围
            assert self.thinking.max_thinking_depth > 0, "max_thinking_depth must be positive"
            assert self.exploration.max_exploration_steps > 0, "max_exploration_steps must be positive"
            assert 0 <= self.exploration.relevance_threshold <= 1, "relevance_threshold must be in [0,1]"
            assert 0 <= self.evidence.evidence_relevance_threshold <= 1, "evidence_relevance_threshold must be in [0,1]"
            assert self.community.min_community_size > 0, "min_community_size must be positive"
            assert 0 <= self.validation.validation_threshold <= 1, "validation_threshold must be in [0,1]"
            
            # 验证策略选项
            valid_reasoning_strategies = ["depth_first", "breadth_first", "adaptive"]
            assert self.thinking.reasoning_strategy in valid_reasoning_strategies, f"Invalid reasoning strategy"
            
            valid_exploration_strategies = ["semantic", "structural", "hybrid"]
            assert self.exploration.exploration_strategy in valid_exploration_strategies, f"Invalid exploration strategy"
            
            valid_community_algorithms = ["louvain", "leiden", "infomap"]
            assert self.community.community_detection_algorithm in valid_community_algorithms, f"Invalid community algorithm"
            
            valid_reasoning_modes = ["full", "fast", "minimal"]
            assert self.reasoning_mode in valid_reasoning_modes, f"Invalid reasoning mode"
            
            return True
            
        except Exception as e:
            print(f"推理配置验证失败: {e}")
            return False
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "thinking": {
                "max_thinking_depth": self.thinking.max_thinking_depth,
                "thinking_timeout": self.thinking.thinking_timeout,
                "max_queries_per_step": self.thinking.max_queries_per_step,
                "reasoning_strategy": self.thinking.reasoning_strategy,
                "enable_parallel_thinking": self.thinking.enable_parallel_thinking
            },
            "exploration": {
                "max_exploration_steps": self.exploration.max_exploration_steps,
                "exploration_width": self.exploration.exploration_width,
                "relevance_threshold": self.exploration.relevance_threshold,
                "exploration_strategy": self.exploration.exploration_strategy,
                "enable_backtracking": self.exploration.enable_backtracking
            },
            "evidence": {
                "max_evidence_per_step": self.evidence.max_evidence_per_step,
                "evidence_relevance_threshold": self.evidence.evidence_relevance_threshold,
                "enable_evidence_validation": self.evidence.enable_evidence_validation,
                "evidence_deduplication": self.evidence.evidence_deduplication
            },
            "community": {
                "community_detection_algorithm": self.community.community_detection_algorithm,
                "min_community_size": self.community.min_community_size,
                "max_communities_per_query": self.community.max_communities_per_query,
                "enable_community_expansion": self.community.enable_community_expansion
            },
            "validation": {
                "enable_answer_validation": self.validation.enable_answer_validation,
                "validation_threshold": self.validation.validation_threshold,
                "enable_complexity_estimation": self.validation.enable_complexity_estimation,
                "enable_consistency_check": self.validation.enable_consistency_check
            },
            "enable_reasoning": self.enable_reasoning,
            "reasoning_mode": self.reasoning_mode,
            "max_reasoning_time": self.max_reasoning_time
        }


# 全局推理配置实例
_global_reasoning_config: Optional[ReasoningConfig] = None


def get_reasoning_config() -> ReasoningConfig:
    """获取全局推理配置实例"""
    global _global_reasoning_config
    if _global_reasoning_config is None:
        _global_reasoning_config = ReasoningConfig()
        if not _global_reasoning_config.validate():
            raise ValueError("推理配置验证失败")
    return _global_reasoning_config


def set_reasoning_config(config: ReasoningConfig):
    """设置全局推理配置"""
    global _global_reasoning_config
    if not config.validate():
        raise ValueError("推理配置验证失败")
    _global_reasoning_config = config
