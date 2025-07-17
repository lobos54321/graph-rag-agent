"""
答案验证器

验证生成答案的准确性、完整性和一致性
"""

from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from search_new.config import get_reasoning_config


@dataclass
class ValidationResult:
    """验证结果数据类"""
    is_valid: bool
    confidence_score: float
    validation_type: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AnswerQuality:
    """答案质量评估数据类"""
    accuracy_score: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    clarity_score: float = 0.0
    overall_score: float = 0.0
    feedback: List[str] = field(default_factory=list)


class AnswerValidator:
    """
    答案验证器：验证生成答案的质量
    
    主要功能：
    1. 准确性验证
    2. 完整性检查
    3. 一致性验证
    4. 质量评估
    """
    
    def __init__(self, llm):
        """
        初始化答案验证器
        
        参数:
            llm: 大语言模型实例
        """
        self.llm = llm
        self.config = get_reasoning_config()
        
        # 验证配置
        self.enable_validation = self.config.validation.enable_answer_validation
        self.validation_threshold = self.config.validation.validation_threshold
        self.enable_complexity = self.config.validation.enable_complexity_estimation
        self.consistency_threshold = self.config.validation.consistency_threshold
        
        # 设置验证链
        self._setup_validation_chains()
        
        # 验证历史
        self.validation_history: List[ValidationResult] = []
        
        print("答案验证器初始化完成")
    
    def _setup_validation_chains(self):
        """设置验证处理链"""
        try:
            # 准确性验证链
            accuracy_prompt = ChatPromptTemplate.from_template("""
            请验证以下答案的准确性：

            问题: {question}
            答案: {answer}
            参考信息: {evidence}

            请从以下方面评估答案的准确性：
            1. 事实是否正确
            2. 信息是否与参考资料一致
            3. 是否存在明显错误

            请以JSON格式返回评估结果：
            {{
                "accuracy_score": 0.8,
                "is_accurate": true,
                "issues": ["问题1", "问题2"],
                "suggestions": ["建议1", "建议2"]
            }}
            """)
            self.accuracy_chain = accuracy_prompt | self.llm | StrOutputParser()
            
            # 完整性验证链
            completeness_prompt = ChatPromptTemplate.from_template("""
            请评估以下答案的完整性：

            问题: {question}
            答案: {answer}
            
            请评估答案是否：
            1. 完整回答了问题的所有方面
            2. 包含了必要的细节
            3. 没有遗漏重要信息

            请以JSON格式返回评估结果：
            {{
                "completeness_score": 0.7,
                "is_complete": true,
                "missing_aspects": ["缺失方面1", "缺失方面2"],
                "suggestions": ["建议1", "建议2"]
            }}
            """)
            self.completeness_chain = completeness_prompt | self.llm | StrOutputParser()
            
            # 一致性验证链
            consistency_prompt = ChatPromptTemplate.from_template("""
            请检查以下答案的内部一致性：

            答案: {answer}
            
            请检查答案是否：
            1. 内部逻辑一致
            2. 没有自相矛盾的表述
            3. 前后表述协调

            请以JSON格式返回评估结果：
            {{
                "consistency_score": 0.9,
                "is_consistent": true,
                "contradictions": ["矛盾1", "矛盾2"],
                "suggestions": ["建议1", "建议2"]
            }}
            """)
            self.consistency_chain = consistency_prompt | self.llm | StrOutputParser()
            
        except Exception as e:
            print(f"验证链设置失败: {e}")
            raise
    
    def validate_answer(self, question: str, answer: str, 
                       evidence: Optional[List[str]] = None) -> ValidationResult:
        """
        验证答案
        
        参数:
            question: 问题
            answer: 答案
            evidence: 证据列表
            
        返回:
            ValidationResult: 验证结果
        """
        if not self.enable_validation:
            return ValidationResult(
                is_valid=True,
                confidence_score=1.0,
                validation_type="disabled"
            )
        
        try:
            print(f"开始验证答案: {question[:50]}...")
            
            # 准备证据文本
            evidence_text = "\n".join(evidence) if evidence else "无参考信息"
            
            # 执行各项验证
            accuracy_result = self._validate_accuracy(question, answer, evidence_text)
            completeness_result = self._validate_completeness(question, answer)
            consistency_result = self._validate_consistency(answer)
            
            # 综合评估
            overall_score = (
                accuracy_result.get("accuracy_score", 0.0) * 0.4 +
                completeness_result.get("completeness_score", 0.0) * 0.3 +
                consistency_result.get("consistency_score", 0.0) * 0.3
            )
            
            is_valid = overall_score >= self.validation_threshold
            
            # 收集问题和建议
            all_issues = []
            all_suggestions = []
            
            for result in [accuracy_result, completeness_result, consistency_result]:
                all_issues.extend(result.get("issues", []))
                all_issues.extend(result.get("contradictions", []))
                all_issues.extend(result.get("missing_aspects", []))
                all_suggestions.extend(result.get("suggestions", []))
            
            # 创建验证结果
            validation_result = ValidationResult(
                is_valid=is_valid,
                confidence_score=overall_score,
                validation_type="comprehensive",
                issues=all_issues,
                suggestions=all_suggestions,
                metadata={
                    "accuracy_score": accuracy_result.get("accuracy_score", 0.0),
                    "completeness_score": completeness_result.get("completeness_score", 0.0),
                    "consistency_score": consistency_result.get("consistency_score", 0.0)
                }
            )
            
            # 记录验证历史
            self.validation_history.append(validation_result)
            
            print(f"答案验证完成，总分: {overall_score:.2f}")
            return validation_result
            
        except Exception as e:
            print(f"答案验证失败: {e}")
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                validation_type="error",
                issues=[f"验证过程出错: {str(e)}"]
            )
    
    def _validate_accuracy(self, question: str, answer: str, evidence: str) -> Dict[str, Any]:
        """验证准确性"""
        try:
            result = self.accuracy_chain.invoke({
                "question": question,
                "answer": answer,
                "evidence": evidence
            })
            
            return self._parse_validation_result(result)
            
        except Exception as e:
            print(f"准确性验证失败: {e}")
            return {"accuracy_score": 0.5, "issues": [f"准确性验证失败: {str(e)}"]}
    
    def _validate_completeness(self, question: str, answer: str) -> Dict[str, Any]:
        """验证完整性"""
        try:
            result = self.completeness_chain.invoke({
                "question": question,
                "answer": answer
            })
            
            return self._parse_validation_result(result)
            
        except Exception as e:
            print(f"完整性验证失败: {e}")
            return {"completeness_score": 0.5, "issues": [f"完整性验证失败: {str(e)}"]}
    
    def _validate_consistency(self, answer: str) -> Dict[str, Any]:
        """验证一致性"""
        try:
            result = self.consistency_chain.invoke({
                "answer": answer
            })
            
            return self._parse_validation_result(result)
            
        except Exception as e:
            print(f"一致性验证失败: {e}")
            return {"consistency_score": 0.5, "issues": [f"一致性验证失败: {str(e)}"]}
    
    def _parse_validation_result(self, result: str) -> Dict[str, Any]:
        """解析验证结果"""
        try:
            import json
            return json.loads(result)
        except json.JSONDecodeError:
            print(f"验证结果解析失败: {result}")
            return {"score": 0.5, "issues": ["结果解析失败"]}
    
    def assess_answer_quality(self, question: str, answer: str, 
                             evidence: Optional[List[str]] = None) -> AnswerQuality:
        """
        评估答案质量
        
        参数:
            question: 问题
            answer: 答案
            evidence: 证据列表
            
        返回:
            AnswerQuality: 质量评估结果
        """
        try:
            # 执行验证
            validation_result = self.validate_answer(question, answer, evidence)
            
            # 提取各项分数
            metadata = validation_result.metadata
            accuracy_score = metadata.get("accuracy_score", 0.0)
            completeness_score = metadata.get("completeness_score", 0.0)
            consistency_score = metadata.get("consistency_score", 0.0)
            
            # 简单的清晰度评估
            clarity_score = self._assess_clarity(answer)
            
            # 计算总分
            overall_score = (
                accuracy_score * 0.3 +
                completeness_score * 0.25 +
                consistency_score * 0.25 +
                clarity_score * 0.2
            )
            
            # 生成反馈
            feedback = []
            if accuracy_score < 0.7:
                feedback.append("答案准确性需要改进")
            if completeness_score < 0.7:
                feedback.append("答案不够完整，缺少重要信息")
            if consistency_score < 0.7:
                feedback.append("答案内部存在不一致之处")
            if clarity_score < 0.7:
                feedback.append("答案表达不够清晰")
            
            if overall_score >= 0.8:
                feedback.append("答案质量良好")
            elif overall_score >= 0.6:
                feedback.append("答案质量一般，有改进空间")
            else:
                feedback.append("答案质量较差，需要重新生成")
            
            return AnswerQuality(
                accuracy_score=accuracy_score,
                completeness_score=completeness_score,
                consistency_score=consistency_score,
                clarity_score=clarity_score,
                overall_score=overall_score,
                feedback=feedback
            )
            
        except Exception as e:
            print(f"质量评估失败: {e}")
            return AnswerQuality(
                overall_score=0.0,
                feedback=[f"质量评估失败: {str(e)}"]
            )
    
    def _assess_clarity(self, answer: str) -> float:
        """评估答案清晰度"""
        try:
            # 简单的清晰度指标
            score = 1.0
            
            # 长度检查
            if len(answer) < 50:
                score -= 0.2  # 太短
            elif len(answer) > 2000:
                score -= 0.1  # 太长
            
            # 结构检查
            if "###" in answer or "####" in answer:
                score += 0.1  # 有标题结构
            
            if "引用数据" in answer or "参考" in answer:
                score += 0.1  # 有引用
            
            # 句子长度检查
            sentences = answer.split('。')
            avg_sentence_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
            if avg_sentence_length > 100:
                score -= 0.1  # 句子太长
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"清晰度评估失败: {e}")
            return 0.5
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """
        获取验证统计信息
        
        返回:
            Dict: 统计信息
        """
        try:
            if not self.validation_history:
                return {"total_validations": 0}
            
            total_validations = len(self.validation_history)
            valid_count = sum(1 for result in self.validation_history if result.is_valid)
            
            avg_confidence = sum(result.confidence_score for result in self.validation_history) / total_validations
            
            # 按验证类型统计
            validation_types = {}
            for result in self.validation_history:
                vtype = result.validation_type
                validation_types[vtype] = validation_types.get(vtype, 0) + 1
            
            return {
                "total_validations": total_validations,
                "valid_count": valid_count,
                "invalid_count": total_validations - valid_count,
                "success_rate": valid_count / total_validations,
                "avg_confidence": avg_confidence,
                "validation_types": validation_types
            }
            
        except Exception as e:
            print(f"获取验证统计失败: {e}")
            return {"error": str(e)}
    
    def clear_history(self):
        """清空验证历史"""
        self.validation_history.clear()
        print("验证历史已清空")
    
    def close(self):
        """关闭验证器"""
        self.clear_history()
        print("答案验证器已关闭")
