# Vector Search Notes

These notes describe PostgreSQL vector retrieval for a local knowledge base.

## pgvector

The pgvector extension stores embedding vectors in PostgreSQL and exposes
cosine distance, inner product, and L2 distance operators. A vector index can
make nearest-neighbor searches practical over many chunks.

## Embeddings

An embedding model maps a passage to a numeric vector. Similar passages have
nearby vectors, so the retriever compares a question vector with stored
embeddings before assembling citation context.

## Vector Search

A vector search retrieves the closest chunks, while keyword and phrase search
can add exact technical identifiers such as pgvector and RRF.

The database can also store ordinary metadata such as an ingestion timestamp.
