"""
Configuration module for Auro-PAI Platform Backend
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # Basic application settings
    APP_NAME: str = "Auro-PAI Platform Backend"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080", "http://localhost:8001"]
    
    # LLM Settings
    LLAMACPP_SERVER_URL: str = "http://10.0.0.206:8000"
    LLAMACPP_MODEL_NAME: str = "mixtral-8x7b-instruct"
    LLAMACPP_TIMEOUT: int = 300
    
    # Alternative LLM providers
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4"
    
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-pro"
    
    # RAG Settings
    CHROMADB_HOST: str = "localhost"
    CHROMADB_PORT: int = 8002
    CHROMADB_COLLECTION_NAME: str = "auro_pai_knowledge"
    
    # Vector embedding settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Tool Settings
    WEB_SEARCH_ENABLED: bool = True
    WEB_SEARCH_API_KEY: Optional[str] = None  # For services like Bing, Google
    WEB_SEARCH_ENGINE: str = "duckduckgo"  # duckduckgo, bing, google
    
    URL_FETCH_TIMEOUT: int = 30
    URL_FETCH_MAX_SIZE: int = 1024 * 1024  # 1MB
    
    # Context Management
    MAX_CONTEXT_LENGTH: int = 8192
    MAX_SESSIONS: int = 100
    SESSION_TIMEOUT: int = 3600  # 1 hour in seconds
    
    # File processing
    SUPPORTED_FILE_TYPES: List[str] = [
        ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp",
        ".md", ".txt", ".json", ".yaml", ".yml", ".xml", ".html",
        ".css", ".sql", ".sh", ".bat", ".ps1"
    ]
    
    # Security
    API_KEY: Optional[str] = None
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    # Database (if needed for persistence)
    DATABASE_URL: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
