# Vector Databases

Vector databases are specialized storage systems designed for high-dimensional vector data.
They enable similarity search, which is fundamental to RAG systems.

## Popular Vector Databases

- **ChromaDB** - Lightweight, in-process vector store perfect for prototyping and small-to-medium workloads
- **Pinecone** - Managed cloud vector database with automatic scaling
- **Weaviate** - Open-source vector database with hybrid search capabilities
- **Qdrant** - High-performance vector similarity search engine

## How Vector Search Works

1. Documents are converted to embeddings (dense vectors) using an embedding model
2. These vectors are stored in the database with associated metadata
3. At query time, the question is also embedded into a vector
4. The database finds the most similar vectors using distance metrics (cosine similarity, euclidean distance)
5. The corresponding documents are returned as context for the LLM
