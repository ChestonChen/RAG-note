# RAG Note

RAG Note 是一个面向个人资料的检索增强问答项目。它把 PDF 和 Markdown 文档整理成可检索的知识库，用户可以通过网页提问、查看引用证据，也可以调用 API 获取检索结果。

## 功能

- 上传并管理 PDF、Markdown 文件，支持异步入库、任务进度和逐文件失败记录。
- 提取文档文本，按 token 长度切片，生成向量并保存来源、标题和页码等信息。
- 使用向量、关键词、短语、全文、精确匹配和 trigram 六路召回，通过加权 RRF 合并结果。
- 展示答案引用和检索证据，可从引用跳转到 PDF 对应页码；提供检索调试接口查看各通道命中情况。
- 调用 OpenAI-compatible 模型生成回答；可选联网搜索补充证据，并对证据不足的问题进行拒答处理。
- 提供检索与答案评测、可选 AI Judge、评测记录持久化和检索参数调优。
- 提供本地 Agent 可调用的检索 API；可选启用 Cross-Encoder 二阶段重排，默认关闭。

## 架构与流程

```text
浏览器（React + TypeScript）
        │
        ▼
FastAPI：文件管理、入库任务、检索、问答、评测 API
        │
        ├── PostgreSQL + pgvector：文档、切片、向量、任务和评测结果
        └── 本地文件：knowledge/ 下的 PDF 与 Markdown 原文
```

入库时，服务解析文件并切片，为切片生成 embedding，再保存到数据库。提问时，查询规划模块先规范化问题，多个检索通道分别召回候选片段，RRF 融合并经过证据筛选；符合回答条件时，服务将证据交给配置的大模型生成答案，并返回引用。证据不足时可尝试联网补充；仍不足则不让模型凭空作答。

## 本机运行

以下命令以 Windows PowerShell 为例。需要 Python 3.12、Node.js 18 或更新版本、pnpm，以及已启动的 Docker Desktop。Docker Compose 只负责启动 PostgreSQL + pgvector，前端和 API 在本机运行。

### 1. 安装依赖并准备配置

在项目根目录执行：

```powershell
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend
pnpm install
cd ..
```

`.env.example` 默认使用 hashing embedding，启动时不需要 embedding API key，适合检查基本流程，不代表真实语义检索效果。正式测试语义检索时，请在 `.env` 中配置本地 BGE-M3 或兼容的远端 embedding 服务，并填写模型实际维度；切换 embedding 模型或维度后需要重新入库。

要生成大模型回答，还需在 `.env` 配置兼容 OpenAI Chat Completions 的服务：

```dotenv
LLM_BASE_URL=https://api.example.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=your-model-name
```

API key 只写入项目根目录的 `.env`，不要提交到 Git，也不要放进前端环境变量。联网搜索和 AI Judge 是可选功能，按需配置相应服务密钥。

### 2. 启动数据库并初始化表结构

```powershell
docker compose up -d postgres
.\.venv\Scripts\python.exe scripts\init_db.py
```

### 3. 启动 API 和网页

在项目根目录的一个终端启动 API：

```powershell
.\.venv\Scripts\python.exe -m uvicorn rag_service.app.main:app --reload --port 8000
```

在另一个终端启动前端：

```powershell
cd frontend
pnpm dev -- --host 127.0.0.1 --port 5173
```

打开 <http://127.0.0.1:5173>。先进入知识库页面上传文件并启动入库，任务完成后即可新建会话提问。FastAPI 文档位于 <http://127.0.0.1:8000/docs>。

## 给本地 Agent 使用

只需要知识证据时调用检索接口：

```http
POST http://127.0.0.1:8000/v1/retrieval/search
Content-Type: application/json

{"question":"你的问题","limit":5}
```

需要系统直接组织回答时调用 `/v1/qa/ask`。问答 API 以 SSE 返回流式结果，结束后包含引用和检索调试信息。

## 测试

在项目根目录运行后端测试：

```powershell
.\.venv\Scripts\python.exe -m pytest rag_service/tests -q
```

前端测试与生产构建在 `frontend/` 目录运行：

```powershell
cd frontend
pnpm test
pnpm build
```

在项目根目录运行离线检索评测：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --isolated
```

该模式使用临时 SQLite 索引和 hashing embedding 验证评测流程，不代表 PostgreSQL + pgvector 或真实 embedding 模型下的检索质量。
