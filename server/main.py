import uvicorn
from fastapi import FastAPI

try:
    # 尝试相对导入
    from .routers import api_router
    from .server_config.database import get_db_manager
    from .services.agent_service import agent_manager
except ImportError:
    # 回退到绝对导入
    from server.routers import api_router
    from server.server_config.database import get_db_manager
    from server.services.agent_service import agent_manager

# 初始化 FastAPI 应用
app = FastAPI(title="知识图谱问答系统", description="基于知识图谱的智能问答系统后端API")

# 添加路由
app.include_router(api_router)

# 获取数据库连接
try:
    db_manager = get_db_manager()
    driver = db_manager.driver if hasattr(db_manager, 'driver') else None
except Exception as e:
    print(f"数据库初始化失败，使用内存模式: {e}")
    driver = None


@app.on_event("shutdown")
def shutdown_event():
    """应用关闭时清理资源"""
    # 关闭所有Agent资源
    agent_manager.close_all()
    
    # 关闭Neo4j连接
    if driver:
        driver.close()
        print("已关闭Neo4j连接")


# 启动服务器
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=workers)