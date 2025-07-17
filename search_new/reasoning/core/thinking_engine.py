"""
思考引擎

管理多轮迭代的思考过程，支持分支推理和反事实分析
"""

from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass, field

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from search_new.config import get_reasoning_config
from search_new.reasoning.utils.nlp_utils import extract_queries_from_text, clean_text


@dataclass
class ThinkingStep:
    """思考步骤数据类"""
    step_id: str
    content: str
    queries: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThinkingSession:
    """思考会话数据类"""
    session_id: str
    query: str
    steps: List[ThinkingStep] = field(default_factory=list)
    current_step: int = 0
    status: str = "active"  # active, completed, failed
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class ThinkingEngine:
    """
    思考引擎：管理多轮迭代的思考过程
    
    主要功能：
    1. 生成思考步骤
    2. 管理思考历史
    3. 支持分支推理
    4. 生成搜索查询
    """
    
    def __init__(self, llm, max_depth: Optional[int] = None):
        """
        初始化思考引擎
        
        参数:
            llm: 大语言模型实例
            max_depth: 最大思考深度
        """
        self.llm = llm
        self.config = get_reasoning_config()
        
        # 思考配置
        self.max_depth = max_depth or self.config.thinking.max_thinking_depth
        self.timeout = self.config.thinking.thinking_timeout
        self.max_queries_per_step = self.config.thinking.max_queries_per_step
        
        # 思考会话管理
        self.sessions: Dict[str, ThinkingSession] = {}
        self.current_session_id: Optional[str] = None
        
        # 消息历史
        self.msg_history: List[Any] = []
        
        print(f"思考引擎初始化完成，最大深度: {self.max_depth}")
    
    def create_session(self, query: str) -> str:
        """
        创建新的思考会话
        
        参数:
            query: 初始查询
            
        返回:
            str: 会话ID
        """
        session_id = f"thinking_{int(time.time() * 1000)}"
        
        session = ThinkingSession(
            session_id=session_id,
            query=query
        )
        
        self.sessions[session_id] = session
        self.current_session_id = session_id
        
        # 初始化消息历史
        self.msg_history = [
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content=f"请分析以下问题并进行深入思考：\n\n{query}")
        ]
        
        print(f"创建思考会话: {session_id}")
        return session_id
    
    def _get_system_prompt(self) -> str:
        """获取系统提示"""
        return self.config.get_prompt_template("thinking")
    
    def initialize_with_query(self, query: str):
        """
        使用查询初始化思考引擎（兼容原接口）
        
        参数:
            query: 查询字符串
        """
        self.create_session(query)
    
    def generate_next_query(self) -> Dict[str, Any]:
        """
        生成下一步搜索查询
        
        返回:
            Dict: 包含查询和状态信息的字典
        """
        if not self.current_session_id:
            return {
                "status": "error",
                "content": "没有活跃的思考会话",
                "queries": []
            }
        
        session = self.sessions[self.current_session_id]
        
        # 检查是否达到最大深度
        if len(session.steps) >= self.max_depth:
            return {
                "status": "answer_ready",
                "content": "已达到最大思考深度，准备生成答案",
                "queries": []
            }
        
        try:
            start_time = time.time()
            
            # 调用LLM进行推理分析
            response = self.llm.invoke(self.msg_history)
            
            # 检查超时
            if time.time() - start_time > self.timeout:
                print("思考过程超时")
                return {
                    "status": "timeout",
                    "content": "思考过程超时",
                    "queries": []
                }
            
            # 提取响应内容
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # 从响应中提取查询
            queries = self._extract_queries(content)
            
            # 创建思考步骤
            step_id = f"step_{len(session.steps) + 1}"
            step = ThinkingStep(
                step_id=step_id,
                content=content,
                queries=queries
            )
            
            # 添加到会话
            session.steps.append(step)
            session.current_step = len(session.steps) - 1
            session.updated_at = time.time()
            
            # 更新消息历史
            self.msg_history.append(AIMessage(content=content))
            
            # 判断状态
            if queries:
                status = "has_query"
            elif self._is_answer_ready(content):
                status = "answer_ready"
            else:
                status = "continue_thinking"
            
            print(f"生成思考步骤 {step_id}，状态: {status}，查询数: {len(queries)}")
            
            return {
                "status": status,
                "content": content,
                "queries": queries,
                "step_id": step_id
            }
            
        except Exception as e:
            print(f"生成查询失败: {e}")
            return {
                "status": "error",
                "content": f"生成查询失败: {str(e)}",
                "queries": []
            }
    
    def _extract_queries(self, content: str) -> List[str]:
        """
        从AI响应中提取搜索查询
        
        参数:
            content: AI响应内容
            
        返回:
            List[str]: 提取的查询列表
        """
        try:
            # 使用NLP工具提取查询
            queries = extract_queries_from_text(content)
            
            # 限制查询数量
            if len(queries) > self.max_queries_per_step:
                queries = queries[:self.max_queries_per_step]
                print(f"查询数量超限，截取前 {self.max_queries_per_step} 个")
            
            # 清理和验证查询
            cleaned_queries = []
            for query in queries:
                cleaned = clean_text(query)
                if cleaned and len(cleaned) > 3:  # 过滤太短的查询
                    cleaned_queries.append(cleaned)
            
            return cleaned_queries
            
        except Exception as e:
            print(f"提取查询失败: {e}")
            return []
    
    def _is_answer_ready(self, content: str) -> bool:
        """
        判断是否准备好生成答案
        
        参数:
            content: AI响应内容
            
        返回:
            bool: 是否准备好
        """
        # 简单的关键词检测
        ready_keywords = [
            "答案准备好了", "可以回答", "总结如下", "结论是",
            "综合以上", "基于分析", "最终答案"
        ]
        
        content_lower = content.lower()
        for keyword in ready_keywords:
            if keyword in content_lower:
                return True
        
        return False
    
    def add_executed_query(self, query: str, results: Optional[str] = None):
        """
        添加已执行的查询结果
        
        参数:
            query: 执行的查询
            results: 查询结果
        """
        if not self.current_session_id:
            return
        
        # 添加到消息历史
        if results:
            self.msg_history.append(HumanMessage(
                content=f"查询 '{query}' 的结果：\n{results}\n\n请基于这些信息继续思考。"
            ))
        else:
            self.msg_history.append(HumanMessage(
                content=f"已执行查询 '{query}'，请继续思考。"
            ))
        
        print(f"添加执行查询: {query}")
    
    def add_reasoning_step(self, info: str):
        """
        添加推理步骤信息
        
        参数:
            info: 推理信息
        """
        if not self.current_session_id:
            return
        
        self.msg_history.append(HumanMessage(
            content=f"获得新信息：\n{info}\n\n请基于这些信息继续分析。"
        ))
        
        print("添加推理步骤信息")
    
    def get_session_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取会话摘要
        
        参数:
            session_id: 会话ID，如果为None则使用当前会话
            
        返回:
            Dict: 会话摘要
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if not session_id or session_id not in self.sessions:
            return {"error": "会话不存在"}
        
        session = self.sessions[session_id]
        
        return {
            "session_id": session_id,
            "query": session.query,
            "status": session.status,
            "steps_count": len(session.steps),
            "current_step": session.current_step,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "duration": session.updated_at - session.created_at,
            "steps": [
                {
                    "step_id": step.step_id,
                    "queries_count": len(step.queries),
                    "timestamp": step.timestamp
                }
                for step in session.steps
            ]
        }
    
    def get_thinking_history(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取思考历史
        
        参数:
            session_id: 会话ID
            
        返回:
            List[Dict]: 思考历史
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if not session_id or session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        
        return [
            {
                "step_id": step.step_id,
                "content": step.content,
                "queries": step.queries,
                "timestamp": step.timestamp,
                "metadata": step.metadata
            }
            for step in session.steps
        ]
    
    def reset_session(self, session_id: Optional[str] = None):
        """
        重置会话
        
        参数:
            session_id: 会话ID
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id and session_id in self.sessions:
            del self.sessions[session_id]
            
            if session_id == self.current_session_id:
                self.current_session_id = None
                self.msg_history = []
            
            print(f"重置思考会话: {session_id}")
    
    def close(self):
        """关闭思考引擎"""
        self.sessions.clear()
        self.current_session_id = None
        self.msg_history = []
        print("思考引擎已关闭")
