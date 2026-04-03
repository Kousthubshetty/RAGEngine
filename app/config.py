from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    model_name: str = "claude-sonnet-4-20250514"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    chroma_persist_dir: str = "./chroma_db"
    knowledge_dir: str = "./knowledge"
    embedding_model: str = "all-MiniLM-L6-v2"
    rate_limit_ask: str = "10/minute"
    rate_limit_refresh: str = "2/minute"
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
