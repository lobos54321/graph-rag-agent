# Render优化的Dockerfile - 功能完整版
FROM python:3.10-slim

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 设置工作目录
WORKDIR /app

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 安装项目
RUN pip install -e . --no-deps

# 创建必要目录
RUN mkdir -p cache files graph_data

# Render自动分配端口，使用环境变量
EXPOSE $PORT

# 启动命令 - 使用独立启动脚本
CMD ["python", "run_server.py"]