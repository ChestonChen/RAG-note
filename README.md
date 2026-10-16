# RAG Knowledge Base MVP

这是一个 API-first 的最小 RAG 知识库。它把 `knowledge/` 下的 Markdown/PDF 作为原始资料，解析并切片后写入 PostgreSQL + pgvector，同时保留来源、标题、页码和检索分数。查询时融合向量检索与关键词检索，再把带 `[DOC-n]` 引用的上下文交给可配置的 OpenAI-compatible LLM（默认可对接 DeepSeek）。证据不足时可以选择联网搜索，并把问答、检索调试和评测结果写入数据库。

当前版本包含一个浏览器工作区：左侧管理会话，中间进行流式对话，右侧查看引用和检索命中，另有知识库页面用于上传与入库。MCP、OCR 和多用户权限仍未加入；reranker 已作为可选的二次排序器加入且默认关闭。重点是让完整的“解析 → 切片 → embedding → 检索 → 引用 → 回答 → 评测”链路可观察、可调用。

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

前置条件：Python 3.12、Git、Docker Desktop（使用 Docker Compose 启动本机 PostgreSQL）。Windows 上 Docker Desktop 需要可用的 WSL2 或 Hyper-V 后端和固件虚拟化；首次启用相关 Windows 功能后可能需要重启。项目不要求单独安装 PostgreSQL 或 `psql`。本仓库已有 `.venv` 时可直接使用它；新环境按 `requirements.txt` 安装依赖。

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

`/v1/retrieval/debug` 会返回每个 chunk 的向量、关键词、PostgreSQL full-text、exact、phrase、trigram 分数、RRF 分数、命中的通道和最终可引用上下文，另外会记录确定性的查询规划、各阶段耗时、证据门控原因和可选 reranker 状态，适合调切片与检索。候选命中仍会显示在调试信息中，但标题、文件名或来源路径本身不会作为答案证据；引用和回答上下文要求正文命中问题相关锚点。明确询问多个主题之间的影响/控制关系时，必须由同一个直接片段支持该关系；明确询问“哪些笔记分别涉及这些主题”则允许多来源发现。没有足够本地证据时 `/v1/qa/ask` 默认尝试联网回退；联网未配置或仍无来源时，会返回低证据结果，不调用 LLM。问答结果会写入 `qa_logs`，日志写入失败不会阻断回答。

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

密钥只放在后端项目根目录的 `.env`，不要提交，也不要写入 `frontend/`。远端调用设置了超时；原有 `/v1/ingest` 和 `/v1/knowledge/ingest` 仍是同步接口，资料量较大时可改用持久化的单 worker 入库任务接口，不需要额外部署 Celery/Redis。

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

### 异步入库、持久化评测和调优

管理员接口会在配置 `ADMIN_API_KEY` 时要求 `X-Admin-API-Key`；未配置时保留本机开发行为。异步入库任务把状态、阶段、当前文件、进度和逐文件失败记录写入数据库，服务重启时会把遗留的 queued/running 任务标成失败，避免页面显示永久运行：

```powershell
$headers = @{ 'X-Admin-API-Key' = 'replace-with-admin-key' }
$job = Invoke-RestMethod -Method Post http://127.0.0.1:8000/v1/admin/ingest/jobs -Headers $headers -ContentType 'application/json' -Body '{"force":false,"run_async":true}'
Invoke-RestMethod -Method Get "http://127.0.0.1:8000/v1/admin/ingest/jobs/$($job.id)" -Headers $headers
Invoke-RestMethod -Method Get "http://127.0.0.1:8000/v1/admin/ingest/jobs/$($job.id)/failures" -Headers $headers
```

评测运行也会保存 `eval_runs` 和 `eval_results`，支持异步进度、单 case 失败后继续、以及结果查询：

评测接口默认只读取项目 `eval/` 目录中的相对路径，避免服务端接口任意读取本机文件。若评测集放在其他受控目录，可在服务端 `.env` 设置 `EVAL_CASES_ROOT`，请求中的 `cases_path` 仍应使用该目录下的相对路径；离线脚本不受这个 API 路径限制。

```powershell
$body = @{ cases_path = 'eval/gitee_note_repository_cases.jsonl'; run_async = $true; limit = 5 } | ConvertTo-Json
$run = Invoke-RestMethod -Method Post http://127.0.0.1:8000/v1/admin/eval/run -Headers $headers -ContentType 'application/json' -Body $body
Invoke-RestMethod -Method Get "http://127.0.0.1:8000/v1/admin/eval/runs/$($run.id)" -Headers $headers
Invoke-RestMethod -Method Get "http://127.0.0.1:8000/v1/admin/eval/runs/$($run.id)/results" -Headers $headers
```

检索调优接口只接受白名单字段，最多运行有限网格，不写回 `.env`。它返回每组参数的来源/证据命中、未知问题误报、耗时、综合分和可复制的环境变量行：

```powershell
$body = @{
  cases_path = 'eval/gitee_note_repository_cases.jsonl'
  grid = @{ retrieval_exact_rrf_weight = @(1.0, 1.2); retrieval_context_limit = @(5, 8) }
  limit_cases = 20
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post http://127.0.0.1:8000/v1/admin/eval/tune -Headers $headers -ContentType 'application/json' -Body $body
```

如需把调优摘要绑定到某次已持久化的评测，可在请求中加入 `run_id`；结果随后可通过 `/v1/admin/eval/runs/{id}/tuning` 查询。失败 trial 不会被选为最佳参数；如果全部 trial 失败，接口返回 `best: null` 和错误标记。

同一份公开 PDF/问题集的对比报告会生成到本地 `reports/gitee_note_repository_comparison.md`。`reports/` 是运行产物，默认不上传到代码仓库；`reference_like` 是根据对方公开资料做的本地代理标签，不是对方服务的实测结果；两 profile 如果配置相同，结果相同是预期现象。

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

对当前 `knowledge/` 下的真实 Markdown/PDF 语料运行同一套离线评估时，可将数据库指向独立的内存 SQLite；该命令不会修改原始知识文件或配置的 PostgreSQL 索引，并会使用 `.env` 中当前配置的 embedding provider：

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_retrieval.py --knowledge-root knowledge --database-url "sqlite+pysqlite:///:memory:" --cases eval/retrieval_cases_knowledge.jsonl --output reports/retrieval_eval_knowledge.json
```

该真实语料评测集覆盖 `example-rag.md`、`knowledge/gitee/` 下的 MySQL、Redis、STL 和 vector PDF，并包含多来源、跨主题比较与未知问题。除来源级 Hit@K、MRR 和来源召回外，已添加 `evidence_groups` 正文锚点标注，独立计算证据段落 Hit@1/3/5 和覆盖度；标题、章节、来源路径不会计作正文证据。PDF 解析器目前将文件标题作为章节元数据（Redis PDF 的标题为 `Docs`），因此报告同时保留来源级指标，但不把它们当作答案段落正确率。

`--isolated` fixture 模式会固定使用确定性的 hashing embedding，验证解析、入库、检索和引用统计流程，但不代表真实语义模型的检索质量。上面的真实语料模式使用 `.env` 配置的 provider，例如本机当前配置的 BGE-M3。两种模式都输出 Hit@K、MRR、来源召回、引用覆盖率和未知问题误报率；标注了 `expected_channels` 或 `relevant_headings` 的 case 还会计算通道召回率和标题元数据召回率。

评估报告还会记录不含密钥的 `embedding_signature`、检索配置、每个 case 的低/中/高置信度分桶、各检索通道的命中贡献、引用来源 precision、证据级命中、仅按 API 实际门控（证据门控、有效引用和置信度阈值）统计的可回答误拒率和未知问题误报明细。`evidence_groups` 的字面命中独立作为答案证据 Hit@K 指标，不会改变 API 是否调用生成模型。证据短语是确定性的字面/归一化匹配，不替代人工标注或 AI Judge 的语义支持判断；小型测试集的结果不能外推成普适准确率。若启用了 reranker，报告会附带它是否实际尝试、状态和可用性；答案生成与 AI judge 仍是可选步骤，检索指标与答案 judge 分开统计。

截至 2026-10-14 的本地收尾基准报告曾生成到 `reports/retrieval_eval_closeout_knowledge_final.json`：BGE-M3 本地模型（1024 维），扫描并重新处理 5 个来源、生成 35 个切片；来源级 Hit@1/3/5 与 MRR 为 1.0，答案证据 Hit@1/3/5 为 0.4/0.8/1.0，可回答误拒率为 0/10，未知问题误报率为 0/4，引用来源覆盖率为 1.0。该结果只描述当前小型且人工标注的语料和问题集，不表示一般场景准确率，也不能据此声称与其他项目达到同等检索质量。报告文件属于本地生成物，未纳入公开代码树。

可以先对当前配置的 embedding provider 做一次无索引探针（只输出 provider、模型、维度和安全错误类型）：

```powershell
.\.venv\Scripts\python.exe scripts/validate_embedding.py
```

该命令只发送一个样本文本，不会读取或打印 API key，不会修改来源文件或数据库，也不会执行重建。

检索默认启用本地查询规划（术语边界匹配、意图词和少量相关词扩展），不调用远端模型。CrossEncoder reranker 默认关闭；配置 `RERANKER_ENABLED=true` 后只对融合候选做二次排序，模型不可用时因 `RERANKER_FAIL_OPEN=true` 自动保留原始结果。启用远端或本地 embedding 后需要重新入库，启用或关闭 reranker 不需要重新入库。

本项目已将 `BAAI/bge-reranker-v2-m3` 下载到 `models/bge-reranker-v2-m3`（约 2.27 GB，目录被 `.gitignore` 忽略）。若要本地试用，先在 `.env` 中设置：

```dotenv
RERANKER_ENABLED=true
RERANKER_MODEL=D:/project/rag-knowledge-base/models/bge-reranker-v2-m3
RERANKER_DEVICE=cpu
```

历史五份知识源的离线 reranker 对比也保存在本地 `reports/` 目录。收尾后的证据门控实测结果以新的 closeout 报告为准；旧报告的来源级 Hit@K/误报数字只作为历史基线，不能代替新的正文证据命中与回答门控指标。评测结果不属于运行代码，默认不纳入公开仓库。

查询术语词典位于 `config/term_dictionary.json`，服务启动后按 `TERM_DICTIONARY_PATH`（默认该路径）加载。词典支持 `entities`、`intent_expansions`、`technical_hints`、`generic_terms` 和 `broad_terms`；修改或损坏文件不会阻断服务，解析失败时会回退到内置词典。检索调试信息只返回词典来源类型（`builtin`/`external`）、版本、意图和扩展数量，不返回词典文件内容或具体术语。

当问题明确包含 `MySQL`、`Redis`、`vector` 等技术主题时，检索会执行可解释的候选主题对齐；最终能否作为答案依据由上下文阶段的正文证据门控决定。`/v1/retrieval/debug` 会同时展示原始候选和 `evidence` 中的通过状态、正文锚点覆盖与拒答原因。

PostgreSQL 只在执行真实入库、向量检索和端到端流程时需要。数据库使用 Docker named volume 持久化，原文文件不会被写入数据库之外的隐式位置。
