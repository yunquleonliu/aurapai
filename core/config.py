"""
Configuration module for Auro-PAI Platform Backend
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # Basic application settings
    APP_NAME: str = "Aura-PAI Platform Backend"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    RELOAD: bool = True
    API_V1_STR: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080", "http://localhost:8001"]
    
    # LLM Settings
    LLAMACPP_SERVER_URL: str = "http://10.0.0.206:8080"
    LLAMACPP_MODEL_NAME: str = "mixtral-8x7b-instruct"
    LLAMACPP_TIMEOUT: int = 300
    
    # Agent Settings
    AGENT_MODE: str = "ReAct"  # Options: "Plan-and-Execute", "ReAct"
    
    # Alternative LLM providers
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4"
    
    # Gemini API and model settings (must match .env if set there)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GEMINI_IMAGE_MODEL: str = "gemini-1.5-flash"

    # Image Generation Settings
    HUGGINGFACE_API_KEY: Optional[str] = None
    IMAGE_GENERATION_ENABLED: bool = True
    
    # RAG Settings
    CHROMADB_HOST: str = "localhost"
    CHROMADB_PORT: int = 8002
    CHROMADB_COLLECTION_NAME: str = "auro_pai_knowledge"
    CHROMADB_PATH: str = "chroma_data"
    
    # Vector embedding settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Tool Settings
    WEB_SEARCH_ENABLED: bool = True
    WEB_SEARCH_ENGINE: str = "duckduckgo"
    WEB_SEARCH_MAX_RESULTS: int = 10
    URL_FETCH_TIMEOUT: int = 30
    URL_FETCH_MAX_SIZE: int = 1048576
    
    # Context Management
    MAX_CONTEXT_LENGTH: int = 8192
    MAX_SESSIONS: int = 100
    SESSION_TIMEOUT: int = 3600
    SESSION_CLEANUP_INTERVAL: int = 600  # 10 minutes
 
    GOOGLE_API_KEY: Optional[str] = "AIzaSyC4MYPb_CO1BRnieP_ttlPfBi9p3L7bWD8"
    GOOGLE_CSE_ID: Optional[str] = "b1506a65a0a54442b"
   
    # File processing
    SUPPORTED_FILE_TYPES: List[str] = [
        ".pdf", ".md", ".txt", ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp",
        ".json", ".yaml", ".yml", ".xml", ".html",
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
