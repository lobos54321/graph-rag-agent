# 使用官方 Python 3.10 镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python-dev-is-python3 \
    libxml2-dev \
    libxslt1-dev \
    antiword \
    unrtf \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 升级pip并安装Python依赖
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 安装项目
RUN pip install -e .

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/graphrag/health || exit 1

# 启动命令
CMD ["python", "server/main.py"]