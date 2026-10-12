# RAG 知识库示例

## PostgreSQL 与 pgvector

pgvector 是 PostgreSQL 的向量扩展。它允许把 embedding 向量和原文元数据放在同一个数据库中，并使用余弦距离等运算检索语义相近的文本。

## 一次检索的流程

资料会先被解析成带标题、页码和来源路径的结构化块，然后按 token 上限切成 chunk。每个 chunk 生成 embedding，写入 PostgreSQL。提问时同时执行向量检索和关键词检索，再用 Reciprocal Rank Fusion（RRF）合并排名。最终上下文使用 `[DOC-n]` 标签，让模型能够返回可核对的引用。

## 本地 Agent 的调用方式

Agent 可以调用 `POST /v1/retrieval/search` 只获取证据，也可以调用 `POST /v1/qa/ask` 直接获取带引用的回答。数据库只是可重建索引，`knowledge/` 下的 Markdown 和 PDF 才是原始资料。
