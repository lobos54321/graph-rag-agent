# GraphRAG Agent - Zeabur 部署指南

## 🚀 快速部署步骤

### 1. 准备GitHub仓库
```bash
# 如果还没有fork，先fork原仓库
# https://github.com/lobos54321/graph-rag-agent

# 或者克隆到你的GitHub账户
git clone https://github.com/lobos54321/graph-rag-agent
cd graph-rag-agent
git remote set-url origin https://github.com/你的用户名/graph-rag-agent
git push -u origin main
```

### 2. 在Zeabur创建项目
1. 访问 [Zeabur](https://zeabur.com)
2. 点击 "New Project"
3. 选择 "Deploy from GitHub"
4. 选择你的 `graph-rag-agent` 仓库

### 3. 添加Neo4j数据库
1. 在Zeabur项目中点击 "Add Service"
2. 选择 "Database" → "Neo4j"
3. Zeabur会自动创建Neo4j实例并设置环境变量

### 4. 配置环境变量
在Zeabur项目设置中添加以下环境变量：

```
OPENAI_API_KEY=你的OpenAI_API_密钥
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_EMBEDDINGS_MODEL=text-embedding-3-large
OPENAI_LLM_MODEL=gpt-4o-mini
TEMPERATURE=0
MAX_TOKENS=2000
VERBOSE=true
CACHE_EMBEDDING_PROVIDER=openai
LANGSMITH_TRACING=false
PYTHONPATH=/app
PYTHONUNBUFFERED=1
```

### 5. 部署完成
- Zeabur会自动构建和部署
- 获得类似 `https://your-project.zeabur.app` 的URL
- 健康检查：`https://your-project.zeabur.app/api/graphrag/health`

## 🔧 API端点

部署成功后，你可以使用以下API端点：

```javascript
// 文件分析
const response = await fetch('https://your-project.zeabur.app/api/graphrag/analyze', {
    method: 'POST',
    body: formData // 包含file字段的FormData
});

// 健康检查
const health = await fetch('https://your-project.zeabur.app/api/graphrag/health');

// 对话查询
const chat = await fetch('https://your-project.zeabur.app/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        message: "你的问题",
        session_id: "session_123"
    })
});
```

## 📱 集成到你的前端

修改你的前端代码，将GraphRAG服务地址更新为：

```javascript
// 在 index.html 中更新
class GraphRAGService {
    constructor() {
        this.baseUrl = 'https://your-project.zeabur.app/api/graphrag';
        this.isConnected = true; // Zeabur部署后应该始终连接
    }
}
```

## 🔍 监控和调试

1. **Zeabur控制台** - 查看部署状态和日志
2. **Neo4j浏览器** - Zeabur会提供Neo4j Web界面链接
3. **API测试** - 使用Postman或curl测试API端点

## ⚡ 优势

- ✅ **零配置数据库** - Zeabur自动配置Neo4j
- ✅ **自动扩容** - 根据流量自动调整资源
- ✅ **HTTPS支持** - 自动SSL证书
- ✅ **一键部署** - 从GitHub自动部署
- ✅ **实时日志** - 在线查看应用日志

## 🚨 注意事项

1. **API密钥安全** - 确保OpenAI API密钥安全存储
2. **资源限制** - 根据使用量选择合适的Zeabur套餐
3. **数据持久化** - Neo4j数据会自动持久化存储