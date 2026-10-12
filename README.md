# RAG Knowledge Base MVP

这是一个 API-first 的最小 RAG 知识库。它把 `knowledge/` 下的 Markdown/PDF 作为原始资料，解析并切片后写入 PostgreSQL + pgvector，同时保留来源、标题、页码和检索分数。查询时融合向量检索与关键词检索，再把带 `[DOC-n]` 引用的上下文交给可配置的 OpenAI-compatible LLM（默认可对接 DeepSeek）。证据不足时可以选择联网搜索，并把问答、检索调试和评测结果写入数据库。

当前版本包含一个浏览器工作区：左侧管理会话，中间进行流式对话，右侧查看引用和检索命中，另有知识库页面用于上传与入库。MCP、OCR、reranker 和多用户权限仍未加入，重点是让完整的“解析 → 切片 → embedding → 检索 → 引用 → 回答 → 评测”链路可观察、可调用。

## 目录

```text
knowledge/              原始 Markdown/PDF（数据库可从这里重建）
frontend/               React + Vite 浏览器工作区
rag_service/app/        配置、解析、切片、入库、检索、上下文和 API
rag_service/tests/      单元测试（无 Docker 也能运行）
scripts/init_db.py      创建 pgvector/pg_trgm 扩展和表
docker-compose.yml      本机 PostgreSQL + pgvector
```

## 本机运行

环境检查结果：本机已有 Python 3.12、pip 和 Git；项目已创建 `.venv` 并安装 `requirements.txt`。Docker Desktop 已安装，Compose 配置可用，但当前 Windows 尚未启用可用的 WSL2/虚拟化后端，Docker 引擎还不能启动；启用 WSL2（可能需要管理员权限和重启）后，再启动 Docker Desktop 即可运行下面的命令。项目不要求单独安装 PostgreSQL 或 `psql`。

1. 复制环境文件并按实际服务修改：

   ```powershell
   Copy-Item .env.example .env
   ```

   默认 `EMBEDDING_PROVIDER=hashing`，不需要 API key，只用于本地 smoke test，不能代表真实语义质量。使用远端 embedding 时改为 `openai_compatible`，填写 `EMBEDDING_BASE_URL`、`EMBEDDING_API_KEY`、`EMBEDDING_MODEL` 和该模型的真实 `EMBEDDING_DIMENSIONS`。切换模型或维度后应重新初始化/重建索引。

2. 启动数据库（Docker Desktop 运行后）：

   ```powershell
   docker compose up -d postgres
   .\.venv\Scripts\python.exe scripts\init_db.py
   ```

3. 启动 API：

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn rag_service.app.main:app --reload --port 8000
   ```

4. 在另一个终端启动浏览器工作区：

   ```powershell
   cd frontend
   .\node_modules\.bin\vite.cmd --host 127.0.0.1 --port 5173
   ```

   然后打开 [http://127.0.0.1:5173](http://127.0.0.1:5173)。Vite 会把 `/api` 请求转发到本机 FastAPI，API key 不会进入浏览器。

   使用流程：先点左下角 `Knowledge base`，上传 Markdown/PDF，点击 `Ingest files`；回到新建会话后提问。点击回答可以查看来源、分数、命中的 chunk 和联网回退状态。

5. 也可以直接用 API 填充索引并查询：

   ```powershell
   Invoke-RestMethod -Method Post http://127.0.0.1:8000/v1/ingest -ContentType 'application/json' -Body '{"force":false}'
   Invoke-RestMethod -Method Post http://127.0.0.1:8000/v1/retrieval/debug -ContentType 'application/json' -Body '{"question":"RRF 如何合并检索结果"}'
   Invoke-RestMethod -Method Post http://127.0.0.1:8000/v1/qa/ask -ContentType 'application/json' -Body '{"question":"pgvector 是做什么的"}'
   ```

`/v1/retrieval/debug` 会返回每个 chunk 的向量分数、关键词分数、RRF 分数、命中的通道和最终上下文，适合调切片与检索。没有足够证据时 `/v1/qa/ask` 默认尝试联网回退；未配置联网搜索或仍然没有来源时，会返回低证据结果，不调用 LLM。问答结果会写入 `qa_logs`，日志写入失败不会阻断回答。

## 远端 API 配置

Embedding 与回答模型是两条独立配置。Embedding 可以使用任何兼容 `/v1/embeddings` 的服务，回答模型可以使用 DeepSeek 等兼容 `/v1/chat/completions` 的服务：

```dotenv
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_API_KEY=your-embedding-key
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSIONS=1024

LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=your-deepseek-key
LLM_MODEL=deepseek-chat
```

例如使用 AirCode 的 OpenAI-compatible 接口时，将回答模型改成：

```dotenv
LLM_BASE_URL=https://api.aircode.xin/v1
LLM_API_KEY=your-aircode-key
LLM_MODEL=gpt-5.6-sol
```

密钥只放在后端项目根目录的 `.env`，不要提交，也不要写入 `frontend/`。远端调用设置了超时；当前 MVP 没有后台队列，`/v1/ingest` 是同步接口，资料量很大时应再加任务队列。

本地浏览器工作区默认不需要管理密钥。若配置 `ADMIN_API_KEY`，知识库管理接口会要求 `X-Admin-API-Key`；当前 UI 尚无管理密钥输入界面，因此本地使用 UI 时请保持该项为空，不要把管理密钥放进 `VITE_*` 前端变量。

### 联网回退与管理接口

联网回退以 Tavily 为例：

```dotenv
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=tavily
TAVILY_API_KEY=your-tavily-key
```

管理接口要求配置 `ADMIN_API_KEY`，并通过 `X-Admin-API-Key` 传递。未配置管理密钥时 `/v1/ingest` 兼容本地开发；一旦配置密钥，公开入库接口也会要求同一个请求头。生产环境建议直接使用受保护的管理接口：

```powershell
$headers = @{ 'X-Admin-API-Key' = 'replace-with-admin-key' }
Invoke-RestMethod -Method Post http://127.0.0.1:8000/v1/admin/ingest -Headers $headers -ContentType 'application/json' -Body '{"force":false}'
Invoke-RestMethod -Method Get http://127.0.0.1:8000/v1/admin/qa-logs -Headers $headers
```

`POST /v1/admin/eval/judge` 可以直接评测一个答案是否被上下文支持；`POST /v1/admin/eval/run` 运行 `eval/retrieval_cases.jsonl` 的检索评测，并可通过 `generate_answers`、`judge_with_ai` 开启答案生成和 AI judge。AI judge 未配置时会返回 `not_configured`，不会让检索评测失败。

## 给本地 Agent 的最小接口

只需要检索证据时调用：

```http
POST http://127.0.0.1:8000/v1/retrieval/search
Content-Type: application/json

{"question":"你的问题","limit":5}
```

需要直接回答时调用 `/v1/qa/ask`。Agent 可以把返回的 `citations` 和 `retrieval_debug` 原样展示或写入自己的上下文。

## 测试

不需要数据库即可运行单元测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 离线检索评估

使用内置 Markdown fixtures 在临时 SQLite 索引中运行离线评估（不会删除或修改配置的 PostgreSQL 索引）：

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_retrieval.py --isolated
```

报告写入 `reports/retrieval_eval.json`，包含 Hit@K、MRR、来源召回、引用覆盖率和未知问题误报率。评估使用确定性的 hashing embedding，仅用于验证解析、入库、检索和引用统计这条 pipeline 的基线；它不代表真实语义模型的检索质量。

PostgreSQL 只在执行真实入库、向量检索和端到端流程时需要。数据库使用 Docker named volume 持久化，原文文件不会被写入数据库之外的隐式位置。
