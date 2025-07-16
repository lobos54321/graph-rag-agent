"""
NLP工具模块

提供自然语言处理相关的工具函数
"""

import re
import string
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    清理文本
    
    参数:
        text: 原始文本
        
    返回:
        str: 清理后的文本
    """
    if not text:
        return ""
    
    try:
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 移除特殊字符（保留基本标点）
        # text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()"-]', '', text)
        
        return text
        
    except Exception as e:
        logger.error(f"文本清理失败: {e}")
        return text


def extract_queries_from_text(text: str) -> List[str]:
    """
    从文本中提取查询
    
    参数:
        text: 包含查询的文本
        
    返回:
        List[str]: 提取的查询列表
    """
    if not text:
        return []
    
    try:
        queries = []
        
        # 方法1: 提取列表格式的查询 (1. 查询1 2. 查询2)
        list_pattern = r'^\s*\d+\.\s*(.+)$'
        for line in text.split('\n'):
            match = re.match(list_pattern, line.strip())
            if match:
                query = match.group(1).strip()
                if query and len(query) > 3:
                    queries.append(query)
        
        # 方法2: 提取引号中的查询
        if not queries:
            quote_pattern = r'["""\'](.*?)["""\']'
            matches = re.findall(quote_pattern, text)
            for match in matches:
                query = match.strip()
                if query and len(query) > 3:
                    queries.append(query)
        
        # 方法3: 提取问号结尾的句子
        if not queries:
            question_pattern = r'([^.!?]*\?)'
            matches = re.findall(question_pattern, text)
            for match in matches:
                query = match.strip()
                if query and len(query) > 3:
                    queries.append(query)
        
        # 方法4: 按句子分割并过滤
        if not queries:
            sentences = split_sentences(text)
            for sentence in sentences:
                if is_query_like(sentence):
                    queries.append(sentence)
        
        # 清理和去重
        cleaned_queries = []
        seen = set()
        for query in queries:
            cleaned = clean_text(query)
            if cleaned and cleaned not in seen and len(cleaned) > 3:
                cleaned_queries.append(cleaned)
                seen.add(cleaned)
        
        return cleaned_queries[:10]  # 限制数量
        
    except Exception as e:
        logger.error(f"提取查询失败: {e}")
        return []


def split_sentences(text: str) -> List[str]:
    """
    分割句子
    
    参数:
        text: 输入文本
        
    返回:
        List[str]: 句子列表
    """
    if not text:
        return []
    
    try:
        # 简单的句子分割
        sentences = re.split(r'[.!?。！？]', text)
        
        # 清理和过滤
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 5:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
        
    except Exception as e:
        logger.error(f"句子分割失败: {e}")
        return [text]


def is_query_like(text: str) -> bool:
    """
    判断文本是否像查询
    
    参数:
        text: 输入文本
        
    返回:
        bool: 是否像查询
    """
    if not text or len(text) < 3:
        return False
    
    try:
        text_lower = text.lower().strip()
        
        # 查询关键词
        query_keywords = [
            '什么', '如何', '为什么', '哪里', '哪个', '谁', '何时',
            'what', 'how', 'why', 'where', 'which', 'who', 'when',
            '请', '帮助', '查找', '搜索', '告诉我'
        ]
        
        # 检查是否包含查询关键词
        for keyword in query_keywords:
            if keyword in text_lower:
                return True
        
        # 检查是否以问号结尾
        if text.endswith('?') or text.endswith('？'):
            return True
        
        # 检查是否是祈使句
        imperative_patterns = [
            r'^(请|帮|告诉|说明|解释|描述)',
            r'^(find|search|tell|explain|describe|show)'
        ]
        
        for pattern in imperative_patterns:
            if re.match(pattern, text_lower):
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"查询判断失败: {e}")
        return False


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    提取关键词
    
    参数:
        text: 输入文本
        max_keywords: 最大关键词数量
        
    返回:
        List[str]: 关键词列表
    """
    if not text:
        return []
    
    try:
        # 简单的关键词提取
        # 移除标点符号
        text_clean = text.translate(str.maketrans('', '', string.punctuation))
        
        # 分词（简单按空格分割）
        words = text_clean.split()
        
        # 过滤停用词（简单版本）
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        }
        
        # 提取关键词
        keywords = []
        word_freq = {}
        
        for word in words:
            word = word.lower().strip()
            if len(word) > 1 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 取前N个关键词
        keywords = [word for word, freq in sorted_words[:max_keywords]]
        
        return keywords
        
    except Exception as e:
        logger.error(f"关键词提取失败: {e}")
        return []


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    提取实体（简单版本）
    
    参数:
        text: 输入文本
        
    返回:
        Dict[str, List[str]]: 实体字典
    """
    if not text:
        return {}
    
    try:
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'others': []
        }
        
        # 简单的实体识别模式
        patterns = {
            'persons': [
                r'([A-Z][a-z]+ [A-Z][a-z]+)',  # 英文人名
                r'([\u4e00-\u9fff]{2,4}(?:先生|女士|教授|博士|老师))',  # 中文人名+称谓
            ],
            'organizations': [
                r'([A-Z][a-zA-Z\s]+(?:Inc|Corp|Ltd|Company|University|Institute))',  # 英文机构
                r'([\u4e00-\u9fff]+(?:公司|大学|学院|研究所|机构|组织))',  # 中文机构
            ],
            'locations': [
                r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*(?:\sCity|\sState|\sCountry))',  # 英文地名
                r'([\u4e00-\u9fff]+(?:市|省|县|区|国|州))',  # 中文地名
            ]
        }
        
        for entity_type, type_patterns in patterns.items():
            for pattern in type_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if match and match not in entities[entity_type]:
                        entities[entity_type].append(match)
        
        return entities
        
    except Exception as e:
        logger.error(f"实体提取失败: {e}")
        return {}


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    计算文本相似度（简单版本）
    
    参数:
        text1: 文本1
        text2: 文本2
        
    返回:
        float: 相似度分数 (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    try:
        # 提取关键词
        keywords1 = set(extract_keywords(text1))
        keywords2 = set(extract_keywords(text2))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        # 计算Jaccard相似度
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        similarity = intersection / union if union > 0 else 0.0
        
        return similarity
        
    except Exception as e:
        logger.error(f"相似度计算失败: {e}")
        return 0.0


def summarize_text(text: str, max_length: int = 200) -> str:
    """
    文本摘要（简单版本）
    
    参数:
        text: 输入文本
        max_length: 最大长度
        
    返回:
        str: 摘要文本
    """
    if not text:
        return ""
    
    try:
        # 如果文本已经很短，直接返回
        if len(text) <= max_length:
            return text
        
        # 按句子分割
        sentences = split_sentences(text)
        
        if not sentences:
            return text[:max_length] + "..."
        
        # 简单选择前几个句子
        summary = ""
        for sentence in sentences:
            if len(summary + sentence) <= max_length:
                summary += sentence + "。"
            else:
                break
        
        if not summary:
            summary = text[:max_length] + "..."
        
        return summary.strip()
        
    except Exception as e:
        logger.error(f"文本摘要失败: {e}")
        return text[:max_length] + "..."


def detect_language(text: str) -> str:
    """
    检测语言（简单版本）
    
    参数:
        text: 输入文本
        
    返回:
        str: 语言代码 (zh, en, unknown)
    """
    if not text:
        return "unknown"
    
    try:
        # 统计中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # 统计英文字符
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = len(text)
        
        if chinese_chars / total_chars > 0.3:
            return "zh"
        elif english_chars / total_chars > 0.5:
            return "en"
        else:
            return "unknown"
            
    except Exception as e:
        logger.error(f"语言检测失败: {e}")
        return "unknown"
