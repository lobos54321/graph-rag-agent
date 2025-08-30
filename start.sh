#!/bin/bash

# GraphRAG Agent 启动脚本
echo "🚀 启动 GraphRAG Agent..."

# 设置环境变量
export PYTHONPATH=/app
export PYTHONUNBUFFERED=1

# 等待 Neo4j 启动
if [ -n "$NEO4J_URI" ]; then
    echo "⏳ 等待 Neo4j 数据库启动..."
    while ! nc -z ${NEO4J_URI#*://} 2>/dev/null; do
        echo "等待 Neo4j..."
        sleep 2
    done
    echo "✅ Neo4j 已启动"
fi

# 创建必要目录
mkdir -p /app/cache
mkdir -p /app/files

# 检查环境变量
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ 错误：未设置 OPENAI_API_KEY"
    exit 1
fi

# 启动服务
echo "🎯 启动 GraphRAG Agent 服务..."
cd /app && python server/main.py