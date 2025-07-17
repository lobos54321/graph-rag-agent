"""
提示模板模块

提供推理过程中使用的各种提示模板
"""

from typing import Dict, Any, Optional

class PromptTemplates:
    """提示模板管理类"""
    
    def __init__(self):
        """初始化提示模板"""
        self._templates = self._load_default_templates()
        print("提示模板初始化完成")
    
    def _load_default_templates(self) -> Dict[str, str]:
        """加载默认提示模板"""
        return {
            # 系统提示
            "system": """你是一个专业的知识分析助手，擅长从复杂信息中提取关键洞察。
请基于提供的信息进行深入分析和推理。""",
            
            # 思考提示
            "thinking": """你是一个深度思考的AI助手。请对以下问题进行深入分析：

问题：{query}

当前已知信息：
{context}

请按照以下步骤进行思考：
1. 分析问题的核心要素
2. 识别需要进一步了解的信息
3. 提出具体的搜索查询
4. 推理可能的答案方向

如果需要更多信息，请提出具体的搜索查询。
如果信息足够，请说明"答案准备好了"。""",
            
            # 查询生成提示
            "query_generation": """基于以下信息生成搜索查询：

原始问题：{original_query}
当前上下文：{context}
生成目标：{goal}

请生成3-5个具体的搜索查询，每个查询应该：
1. 针对问题的特定方面
2. 可以独立搜索
3. 有助于回答原始问题

查询列表：
1. 
2. 
3. 
4. 
5. """,
            
            # 证据评估提示
            "evidence_evaluation": """请评估以下证据与查询的相关性：

查询：{query}
证据：{evidence}

请从以下方面评估：
1. 相关性（0-1分）
2. 可信度（0-1分）
3. 完整性（0-1分）

评估结果：
- 相关性分数：
- 可信度分数：
- 完整性分数：
- 评估理由：""",
            
            # 答案生成提示
            "answer_generation": """基于以下信息生成完整的答案：

问题：{query}
证据信息：
{evidence}

请生成一个结构化的答案，包括：
1. 主要观点（使用三级标题###）
2. 支撑细节
3. 引用数据（使用####标记）

答案要求：
- 准确性：基于证据，避免推测
- 完整性：全面回答问题各个方面
- 清晰性：结构清晰，表达简洁""",
            
            # 答案验证提示
            "answer_validation": """请验证以下答案的质量：

问题：{query}
答案：{answer}
参考证据：{evidence}

请从以下方面验证：
1. 准确性：答案是否与证据一致
2. 完整性：是否完整回答了问题
3. 一致性：答案内部是否逻辑一致
4. 清晰性：表达是否清晰易懂

验证结果：
- 准确性评分（0-1）：
- 完整性评分（0-1）：
- 一致性评分（0-1）：
- 清晰性评分（0-1）：
- 主要问题：
- 改进建议：""",
            
            # 复杂度评估提示
            "complexity_assessment": """请评估以下查询的复杂度：

查询：{query}

请从以下维度评估：
1. 词汇复杂度：专业术语、词汇难度
2. 语义复杂度：概念关系、抽象程度
3. 结构复杂度：句法结构、逻辑关系
4. 推理复杂度：推理步骤、思维深度

评估结果：
- 词汇复杂度（0-1）：
- 语义复杂度（0-1）：
- 结构复杂度（0-1）：
- 推理复杂度（0-1）：
- 总体复杂度：简单/中等/复杂/非常复杂
- 复杂度因素：
- 处理建议：""",
            
            # 关键词提取提示
            "keyword_extraction": """请从以下文本中提取关键词：

文本：{text}

请提取：
1. 低级关键词：具体实体、人名、地名、专有名词
2. 高级关键词：概念、主题、领域术语

提取结果：
低级关键词：
高级关键词：""",
            
            # 实体识别提示
            "entity_recognition": """请识别以下文本中的实体：

文本：{text}

请识别：
1. 人物：人名、职位、角色
2. 组织：公司、机构、团体
3. 地点：地名、位置、区域
4. 时间：日期、时间、时期
5. 其他：产品、概念、术语

识别结果：
人物：
组织：
地点：
时间：
其他：""",
            
            # 关系抽取提示
            "relation_extraction": """请抽取以下文本中的实体关系：

文本：{text}
实体列表：{entities}

请识别实体之间的关系，格式：实体1 - 关系 - 实体2

关系类型包括：
- 属于、包含、位于
- 影响、导致、促进
- 相关、关联、连接
- 时间关系、空间关系

抽取结果：""",
            
            # 摘要生成提示
            "summarization": """请对以下内容生成摘要：

内容：{content}
摘要长度：{max_length}字以内

摘要要求：
1. 保留核心信息
2. 逻辑清晰
3. 语言简洁
4. 突出重点

摘要：""",
            
            # 问题分解提示
            "question_decomposition": """请将以下复杂问题分解为简单子问题：

复杂问题：{complex_query}

分解原则：
1. 每个子问题独立可答
2. 子问题组合能回答原问题
3. 按逻辑顺序排列
4. 避免重复和冗余

子问题列表：
1. 
2. 
3. 
4. 
5. """,
            
            # 结果融合提示
            "result_fusion": """请融合以下多个搜索结果：

原始问题：{query}

搜索结果1：
{result1}

搜索结果2：
{result2}

搜索结果3：
{result3}

融合要求：
1. 整合互补信息
2. 解决冲突信息
3. 保持逻辑一致
4. 突出重要内容

融合结果：""",
            
            # 反思提示
            "reflection": """请对以下推理过程进行反思：

推理过程：{reasoning_process}
得出结论：{conclusion}

反思要点：
1. 推理逻辑是否合理
2. 证据是否充分
3. 结论是否可靠
4. 是否存在偏见或错误

反思结果：
- 推理质量评估：
- 发现的问题：
- 改进建议：
- 置信度评估："""
        }
    
    def get_template(self, template_name: str) -> Optional[str]:
        """
        获取指定的提示模板
        
        参数:
            template_name: 模板名称
            
        返回:
            str: 提示模板，如果不存在则返回None
        """
        return self._templates.get(template_name)
    
    def format_template(self, template_name: str, **kwargs) -> Optional[str]:
        """
        格式化提示模板
        
        参数:
            template_name: 模板名称
            **kwargs: 模板参数
            
        返回:
            str: 格式化后的提示，如果模板不存在则返回None
        """
        template = self.get_template(template_name)
        if template is None:
            print(f"模板不存在: {template_name}")
            return None
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            print(f"模板参数缺失: {e}")
            return None
        except Exception as e:
            print(f"模板格式化失败: {e}")
            return None
    
    def add_template(self, template_name: str, template_content: str):
        """
        添加新的提示模板
        
        参数:
            template_name: 模板名称
            template_content: 模板内容
        """
        self._templates[template_name] = template_content
        print(f"添加提示模板: {template_name}")
    
    def update_template(self, template_name: str, template_content: str):
        """
        更新提示模板
        
        参数:
            template_name: 模板名称
            template_content: 新的模板内容
        """
        if template_name in self._templates:
            self._templates[template_name] = template_content
            print(f"更新提示模板: {template_name}")
        else:
            print(f"模板不存在，无法更新: {template_name}")
    
    def remove_template(self, template_name: str):
        """
        删除提示模板
        
        参数:
            template_name: 模板名称
        """
        if template_name in self._templates:
            del self._templates[template_name]
            print(f"删除提示模板: {template_name}")
        else:
            print(f"模板不存在，无法删除: {template_name}")
    
    def list_templates(self) -> list:
        """
        列出所有可用的模板名称
        
        返回:
            list: 模板名称列表
        """
        return list(self._templates.keys())
    
    def get_template_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有模板的信息
        
        返回:
            Dict: 模板信息字典
        """
        info = {}
        for name, template in self._templates.items():
            info[name] = {
                "length": len(template),
                "parameters": self._extract_parameters(template),
                "description": self._get_template_description(name)
            }
        return info
    
    def _extract_parameters(self, template: str) -> list:
        """从模板中提取参数"""
        import re
        parameters = re.findall(r'\{(\w+)\}', template)
        return list(set(parameters))
    
    def _get_template_description(self, template_name: str) -> str:
        """获取模板描述"""
        descriptions = {
            "system": "系统级提示，定义AI助手的角色和行为",
            "thinking": "思考过程提示，引导深度分析和推理",
            "query_generation": "查询生成提示，用于生成搜索查询",
            "evidence_evaluation": "证据评估提示，评估证据质量",
            "answer_generation": "答案生成提示，基于证据生成答案",
            "answer_validation": "答案验证提示，验证答案质量",
            "complexity_assessment": "复杂度评估提示，评估查询复杂度",
            "keyword_extraction": "关键词提取提示，从文本提取关键词",
            "entity_recognition": "实体识别提示，识别文本中的实体",
            "relation_extraction": "关系抽取提示，抽取实体间关系",
            "summarization": "摘要生成提示，生成内容摘要",
            "question_decomposition": "问题分解提示，分解复杂问题",
            "result_fusion": "结果融合提示，融合多个搜索结果",
            "reflection": "反思提示，对推理过程进行反思"
        }
        return descriptions.get(template_name, "无描述")


# 全局提示模板实例
_global_templates: Optional[PromptTemplates] = None


def get_prompt_templates() -> PromptTemplates:
    """获取全局提示模板实例"""
    global _global_templates
    if _global_templates is None:
        _global_templates = PromptTemplates()
    return _global_templates


def get_prompt(template_name: str, **kwargs) -> Optional[str]:
    """
    获取格式化的提示
    
    参数:
        template_name: 模板名称
        **kwargs: 模板参数
        
    返回:
        str: 格式化后的提示
    """
    templates = get_prompt_templates()
    return templates.format_template(template_name, **kwargs)
