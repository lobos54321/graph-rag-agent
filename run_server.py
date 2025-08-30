#!/usr/bin/env python3
"""
GraphRAG Agent 服务启动脚本
适配Render等云平台部署
"""

import os
import sys
import uvicorn
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """启动服务"""
    # 从环境变量获取端口，默认8000
    port = int(os.getenv("PORT", 8000))
    
    print(f"🚀 启动GraphRAG Agent服务...")
    print(f"📡 端口: {port}")
    print(f"🌐 环境: {'生产' if os.getenv('RENDER') else '开发'}")
    
    # 启动uvicorn服务
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=port,
        workers=1,  # 免费层使用单worker
        access_log=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()