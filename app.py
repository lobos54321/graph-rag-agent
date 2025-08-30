#!/usr/bin/env python3
"""
GraphRAG Agent 主应用入口
简化版本，避免复杂的模块导入问题
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "server"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 创建FastAPI应用
app = FastAPI(
    title="GraphRAG Agent API",
    description="基于知识图谱的智能文档分析系统",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

@app.get("/")
async def root():
    """根路径"""
    return {"message": "GraphRAG Agent API is running!", "version": "1.0.0"}

@app.get("/api/graphrag/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "GraphRAG Agent",
        "database": "memory" if os.getenv("DATABASE_TYPE", "memory") == "memory" else "neo4j",
        "embedding_provider": os.getenv("CACHE_EMBEDDING_PROVIDER", "openai")
    }

@app.post("/api/graphrag/analyze")
async def analyze_document(request: dict):
    """文档分析端点"""
    try:
        # 模拟GraphRAG分析结果
        return {
            "status": "success",
            "analysis": {
                "content": "这是一个关于智能内容创作系统的技术文档，详细描述了基于AI技术的全流程内容生产解决方案。",
                "concepts": ["人工智能", "内容创作", "自动化工作流", "数据分析"],
                "entities": ["AI系统", "内容创作者", "营销团队", "数据分析师"],
                "knowledgeTreeSuggestion": "技术文档/人工智能/内容创作系统",
                "confidence": 0.85
            },
            "service_ready": True
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"分析失败: {str(e)}",
            "service_ready": False
        }

@app.post("/api/chat")
async def chat():
    """对话端点 - 临时实现"""
    return {
        "status": "success", 
        "message": "Chat endpoint - coming soon",
        "service_ready": True
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 启动GraphRAG Agent (简化版)...")
    print(f"📡 端口: {port}")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )