# 多阶段构建，减少最终镜像大小
FROM python:3.10-alpine as builder

# 安装构建依赖
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    openssl-dev \
    python3-dev \
    build-base

# 设置工作目录
WORKDIR /build

# 复制并安装Python依赖到虚拟环境
COPY requirements.minimal.txt requirements.txt
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# 生产阶段
FROM python:3.10-alpine

# 安装运行时依赖（最小化）
RUN apk add --no-cache \
    curl \
    poppler-utils \
    && rm -rf /var/cache/apk/*

# 从构建阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# 设置工作目录
WORKDIR /app

# 复制应用代码（仅必要文件）
COPY server/ ./server/
COPY graph_rag/ ./graph_rag/
COPY *.py ./
COPY setup.py ./

# 创建必要目录
RUN mkdir -p cache files

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/graphrag/health || exit 1

# 启动命令
CMD ["python", "server/main.py"]