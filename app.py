#!/usr/bin/env python3
"""
GraphRAG Agent 主应用入口
简化版本，避免复杂的模块导入问题
"""

import os
import sys
from pathlib import Path
import openai
import PyPDF2
import io

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "server"))

from fastapi import FastAPI, HTTPException, File, UploadFile
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

def extract_text_from_file(content: bytes, filename: str) -> str:
    """从文件内容提取文本"""
    try:
        if filename.lower().endswith('.pdf'):
            # PDF文件提取
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        elif filename.lower().endswith(('.txt', '.md')):
            # 文本文件
            return content.decode('utf-8', errors='ignore')
        else:
            # 其他文件类型，尝试解码为文本
            return content.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"文件内容提取失败: {e}")
        return f"无法提取文件内容，文件类型: {filename.split('.')[-1] if '.' in filename else 'unknown'}"

async def analyze_with_openai(text_content: str, filename: str) -> dict:
    """使用OpenAI进行真正的AI内容分析"""
    try:
        # 设置OpenAI API key
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # 限制内容长度，避免token超限
        if len(text_content) > 8000:
            text_content = text_content[:8000] + "..."
            
        prompt = f"""
请分析以下文档内容，并以JSON格式返回分析结果：

文档名称: {filename}
文档内容:
{text_content}

请返回以下格式的JSON：
{{
    "content": "文档内容的详细摘要(150字以内)",
    "concepts": ["提取的关键概念1", "概念2", "概念3", "概念4"],
    "entities": ["重要实体1", "实体2", "实体3", "实体4"],
    "knowledgeTreeSuggestion": "建议的知识树分类路径(如:技术文档/AI开发/系统架构)",
    "confidence": 0.9
}}

注意：
1. 请基于文档的实际内容进行分析，不要只依赖文件名
2. 概念和实体要从文档内容中真实提取
3. 知识树建议要准确反映文档的主题分类
4. 置信度为0-1之间的数字
"""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一个专业的文档分析助手，擅长提取文档的核心内容、概念和实体。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        
        result_text = response.choices[0].message.content
        print(f"🤖 OpenAI分析结果: {result_text}")
        
        # 解析JSON响应
        import json
        result = json.loads(result_text)
        return result
        
    except Exception as e:
        print(f"❌ OpenAI分析失败: {e}")
        # 回退到基础分析
        return {
            "content": f"基于AI分析的文档摘要生成失败，文档名称：{filename}",
            "concepts": ["文档分析", "内容提取"],
            "entities": ["AI系统", "用户"],
            "knowledgeTreeSuggestion": "文档管理/AI分析/待处理",
            "confidence": 0.6
        }

@app.post("/api/graphrag/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """真正的AI文档分析端点"""
    try:
        # 读取文件内容
        content = await file.read()
        filename = file.filename
        file_size = len(content)
        
        print(f"📄 接收到文件: {filename}, 大小: {file_size} bytes")
        
        # 🔥 提取文件文本内容
        text_content = extract_text_from_file(content, filename)
        print(f"📝 提取文本长度: {len(text_content)} 字符")
        
        # 🤖 使用OpenAI进行真正的AI分析
        if text_content and len(text_content) > 50:  # 确保有足够内容分析
            ai_analysis = await analyze_with_openai(text_content, filename)
        else:
            # 如果内容太少，使用基础分析
            ai_analysis = {
                "content": f"文档内容较少或无法提取，文件名：{filename}",
                "concepts": ["文档处理", "内容提取"],
                "entities": ["文档", "系统"],
                "knowledgeTreeSuggestion": "文档管理/待分类/需要处理",
                "confidence": 0.5
            }
        
        return {
            "status": "success",
            "analysis": {
                "content": ai_analysis.get("content", "AI分析完成"),
                "concepts": ai_analysis.get("concepts", []),
                "entities": ai_analysis.get("entities", []),
                "knowledgeTreeSuggestion": ai_analysis.get("knowledgeTreeSuggestion", "文档管理/AI分析"),
                "confidence": ai_analysis.get("confidence", 0.85),
                "fileInfo": {
                    "filename": filename,
                    "size": file_size,
                    "type": file.content_type or "unknown",
                    "textLength": len(text_content) if 'text_content' in locals() else 0
                }
            },
            "service_ready": True
        }
    except Exception as e:
        print(f"❌ 分析错误: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"分析失败: {str(e)}",
                "service_ready": False
            }
        )

@app.post("/api/chat")
async def chat():
    """对话端点"""
    try:
        return {
            "status": "success", 
            "response": "GraphRAG智能对话功能正在开发中，敬请期待！",
            "service_ready": True
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"对话失败: {str(e)}",
                "service_ready": False
            }
        )

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