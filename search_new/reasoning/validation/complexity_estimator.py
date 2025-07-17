"""
复杂度评估器

评估查询和推理任务的复杂度，指导搜索策略选择
"""

from typing import Dict, List, Optional
import re
from dataclasses import dataclass, field
from enum import Enum

from search_new.config import get_reasoning_config
from search_new.reasoning.utils.nlp_utils import extract_keywords, extract_entities

class ComplexityLevel(Enum):
    """复杂度级别枚举"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class ComplexityMetrics:
    """复杂度指标数据类"""
    lexical_complexity: float = 0.0
    semantic_complexity: float = 0.0
    structural_complexity: float = 0.0
    reasoning_complexity: float = 0.0
    overall_complexity: float = 0.0
    complexity_level: ComplexityLevel = ComplexityLevel.SIMPLE
    factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class ComplexityEstimator:
    """
    复杂度评估器：评估查询和任务的复杂度
    
    主要功能：
    1. 词汇复杂度分析
    2. 语义复杂度评估
    3. 结构复杂度检测
    4. 推理复杂度判断
    """
    
    def __init__(self):
        """初始化复杂度评估器"""
        self.config = get_reasoning_config()
        
        # 复杂度阈值
        self.simple_threshold = 0.3
        self.moderate_threshold = 0.6
        self.complex_threshold = 0.8
        
        # 复杂度指标权重
        self.lexical_weight = 0.2
        self.semantic_weight = 0.3
        self.structural_weight = 0.25
        self.reasoning_weight = 0.25
        
        # 复杂度关键词
        self._setup_complexity_patterns()
        
        print("复杂度评估器初始化完成")
    
    def _setup_complexity_patterns(self):
        """设置复杂度模式"""
        # 高复杂度关键词
        self.high_complexity_keywords = {
            '比较': ['比较', '对比', '区别', '差异', '相似', '不同'],
            '分析': ['分析', '解释', '原因', '为什么', '如何', '机制'],
            '评估': ['评估', '评价', '判断', '优缺点', '利弊', '影响'],
            '综合': ['综合', '总结', '概括', '整合', '关联', '联系'],
            '推理': ['推断', '推理', '预测', '假设', '可能', '趋势'],
            '多步骤': ['步骤', '过程', '流程', '阶段', '顺序', '先后']
        }
        
        # 复杂结构模式
        self.complex_patterns = [
            r'如果.*那么.*',  # 条件句
            r'不仅.*而且.*',  # 递进句
            r'虽然.*但是.*',  # 转折句
            r'一方面.*另一方面.*',  # 对比句
            r'首先.*其次.*最后.*',  # 列举句
        ]
        
        # 简单查询模式
        self.simple_patterns = [
            r'^什么是.*',  # 定义查询
            r'^.*是什么.*',  # 定义查询
            r'^.*在哪里.*',  # 位置查询
            r'^.*什么时候.*',  # 时间查询
        ]
    
    def estimate_complexity(self, query: str, context: Optional[str] = None) -> ComplexityMetrics:
        """
        评估查询复杂度
        
        参数:
            query: 查询字符串
            context: 上下文信息
            
        返回:
            ComplexityMetrics: 复杂度指标
        """
        try:
            print(f"评估查询复杂度: {query[:50]}...")
            
            # 计算各项复杂度指标
            lexical_complexity = self._calculate_lexical_complexity(query)
            semantic_complexity = self._calculate_semantic_complexity(query, context)
            structural_complexity = self._calculate_structural_complexity(query)
            reasoning_complexity = self._calculate_reasoning_complexity(query)
            
            # 计算总体复杂度
            overall_complexity = (
                lexical_complexity * self.lexical_weight +
                semantic_complexity * self.semantic_weight +
                structural_complexity * self.structural_weight +
                reasoning_complexity * self.reasoning_weight
            )
            
            # 确定复杂度级别
            complexity_level = self._determine_complexity_level(overall_complexity)
            
            # 识别复杂度因素
            factors = self._identify_complexity_factors(query, {
                'lexical': lexical_complexity,
                'semantic': semantic_complexity,
                'structural': structural_complexity,
                'reasoning': reasoning_complexity
            })
            
            # 生成建议
            recommendations = self._generate_recommendations(complexity_level, factors)
            
            return ComplexityMetrics(
                lexical_complexity=lexical_complexity,
                semantic_complexity=semantic_complexity,
                structural_complexity=structural_complexity,
                reasoning_complexity=reasoning_complexity,
                overall_complexity=overall_complexity,
                complexity_level=complexity_level,
                factors=factors,
                recommendations=recommendations
            )
            
        except Exception as e:
            print(f"复杂度评估失败: {e}")
            return ComplexityMetrics()
    
    def _calculate_lexical_complexity(self, query: str) -> float:
        """计算词汇复杂度"""
        try:
            score = 0.0
            
            # 查询长度
            length_score = min(len(query) / 200, 1.0)  # 标准化到0-1
            score += length_score * 0.3
            
            # 词汇多样性
            words = query.split()
            unique_words = set(words)
            diversity_score = len(unique_words) / len(words) if words else 0
            score += diversity_score * 0.4
            
            # 专业术语检测（简单版本）
            technical_terms = 0
            for word in words:
                if len(word) > 8:  # 长词通常更专业
                    technical_terms += 1
            
            technical_score = min(technical_terms / len(words), 0.5) if words else 0
            score += technical_score * 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            print(f"词汇复杂度计算失败: {e}")
            return 0.5
    
    def _calculate_semantic_complexity(self, query: str, context: Optional[str] = None) -> float:
        """计算语义复杂度"""
        try:
            score = 0.0
            
            # 实体数量
            entities = extract_entities(query)
            entity_count = sum(len(entity_list) for entity_list in entities.values())
            entity_score = min(entity_count / 10, 1.0)
            score += entity_score * 0.4
            
            # 关键词复杂度
            keywords = extract_keywords(query)
            keyword_score = min(len(keywords) / 15, 1.0)
            score += keyword_score * 0.3
            
            # 语义关系复杂度
            relation_score = 0.0
            for category, keywords_list in self.high_complexity_keywords.items():
                for keyword in keywords_list:
                    if keyword in query:
                        relation_score += 0.1
            
            score += min(relation_score, 0.5) * 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            print(f"语义复杂度计算失败: {e}")
            return 0.5
    
    def _calculate_structural_complexity(self, query: str) -> float:
        """计算结构复杂度"""
        try:
            score = 0.0
            
            # 检查复杂结构模式
            complex_pattern_count = 0
            for pattern in self.complex_patterns:
                if re.search(pattern, query):
                    complex_pattern_count += 1
            
            pattern_score = min(complex_pattern_count / 3, 1.0)
            score += pattern_score * 0.5
            
            # 句子数量
            sentences = query.split('。')
            sentence_score = min(len(sentences) / 5, 1.0)
            score += sentence_score * 0.3
            
            # 标点符号复杂度
            punctuation_count = len(re.findall(r'[,;:!?()]', query))
            punctuation_score = min(punctuation_count / 10, 1.0)
            score += punctuation_score * 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            print(f"结构复杂度计算失败: {e}")
            return 0.5
    
    def _calculate_reasoning_complexity(self, query: str) -> float:
        """计算推理复杂度"""
        try:
            score = 0.0
            
            # 推理关键词检测
            reasoning_keywords = [
                '为什么', '如何', '原因', '机制', '影响', '结果',
                '比较', '分析', '评估', '预测', '推断', '假设'
            ]
            
            reasoning_count = 0
            for keyword in reasoning_keywords:
                if keyword in query:
                    reasoning_count += 1
            
            reasoning_score = min(reasoning_count / 5, 1.0)
            score += reasoning_score * 0.6
            
            # 多步骤推理检测
            multi_step_indicators = ['首先', '然后', '接着', '最后', '步骤', '阶段']
            multi_step_count = sum(1 for indicator in multi_step_indicators if indicator in query)
            multi_step_score = min(multi_step_count / 3, 1.0)
            score += multi_step_score * 0.4
            
            return min(score, 1.0)
            
        except Exception as e:
            print(f"推理复杂度计算失败: {e}")
            return 0.5
    
    def _determine_complexity_level(self, overall_complexity: float) -> ComplexityLevel:
        """确定复杂度级别"""
        if overall_complexity < self.simple_threshold:
            return ComplexityLevel.SIMPLE
        elif overall_complexity < self.moderate_threshold:
            return ComplexityLevel.MODERATE
        elif overall_complexity < self.complex_threshold:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.VERY_COMPLEX
    
    def _identify_complexity_factors(self, query: str, scores: Dict[str, float]) -> List[str]:
        """识别复杂度因素"""
        factors = []
        
        try:
            # 检查各项指标
            if scores['lexical'] > 0.7:
                factors.append("词汇复杂度高")
            if scores['semantic'] > 0.7:
                factors.append("语义关系复杂")
            if scores['structural'] > 0.7:
                factors.append("句法结构复杂")
            if scores['reasoning'] > 0.7:
                factors.append("需要复杂推理")
            
            # 检查特定模式
            if any(re.search(pattern, query) for pattern in self.complex_patterns):
                factors.append("包含复杂语法结构")
            
            if len(query) > 200:
                factors.append("查询长度较长")
            
            # 检查多个问题
            question_marks = query.count('?') + query.count('？')
            if question_marks > 1:
                factors.append("包含多个子问题")
            
            return factors
            
        except Exception as e:
            print(f"复杂度因素识别失败: {e}")
            return ["因素识别失败"]
    
    def _generate_recommendations(self, complexity_level: ComplexityLevel, 
                                factors: List[str]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        try:
            if complexity_level == ComplexityLevel.SIMPLE:
                recommendations.append("使用基础搜索策略")
                recommendations.append("单步搜索即可满足需求")
            
            elif complexity_level == ComplexityLevel.MODERATE:
                recommendations.append("使用本地搜索或全局搜索")
                recommendations.append("可能需要2-3轮搜索")
            
            elif complexity_level == ComplexityLevel.COMPLEX:
                recommendations.append("建议使用混合搜索策略")
                recommendations.append("需要多轮迭代搜索")
                recommendations.append("考虑使用推理组件")
            
            else:  # VERY_COMPLEX
                recommendations.append("使用完整的推理搜索流程")
                recommendations.append("启用思考引擎和证据跟踪")
                recommendations.append("需要多步骤分解和验证")
            
            # 基于具体因素的建议
            if "词汇复杂度高" in factors:
                recommendations.append("注意专业术语的处理")
            
            if "需要复杂推理" in factors:
                recommendations.append("启用推理验证功能")
            
            if "包含多个子问题" in factors:
                recommendations.append("考虑问题分解策略")
            
            return recommendations
            
        except Exception as e:
            print(f"建议生成失败: {e}")
            return ["建议生成失败"]
    
    def batch_estimate_complexity(self, queries: List[str]) -> Dict[str, ComplexityMetrics]:
        """
        批量评估复杂度
        
        参数:
            queries: 查询列表
            
        返回:
            Dict[str, ComplexityMetrics]: 查询到复杂度指标的映射
        """
        try:
            results = {}
            
            for query in queries:
                results[query] = self.estimate_complexity(query)
            
            return results
            
        except Exception as e:
            print(f"批量复杂度评估失败: {e}")
            return {}
    
    def get_complexity_distribution(self, queries: List[str]) -> Dict[str, int]:
        """
        获取复杂度分布
        
        参数:
            queries: 查询列表
            
        返回:
            Dict[str, int]: 复杂度级别分布
        """
        try:
            distribution = {level.value: 0 for level in ComplexityLevel}
            
            for query in queries:
                complexity = self.estimate_complexity(query)
                distribution[complexity.complexity_level.value] += 1
            
            return distribution
            
        except Exception as e:
            print(f"复杂度分布计算失败: {e}")
            return {}
    
    def close(self):
        """关闭复杂度评估器"""
        print("复杂度评估器已关闭")
