from pydantic import field_validator
from pydantic_settings import BaseSettings
from typing import List, Any
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Zero-Drift Image Generation"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # API Keys
    NANO_BANANA_API_KEY: str = os.getenv("NANO_BANANA_API_KEY", "")
    NANO_BANANA_BASE_URL: str = os.getenv("NANO_BANANA_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    GEMINI_MODEL_VERSION: str = os.getenv("GEMINI_MODEL_VERSION", "nano-banana-pro-preview")
    
    # Storage
    UPLOAD_DIR: str = "uploads"
    IMAGES_DIR: str = "uploads/images"
    EXCEL_DIR: str = "uploads/excel"
    EXPORT_DIR: str = "exports"
    
    # Security
    ALLOWED_EXTENSIONS: Any = ["png", "jpg", "jpeg", "xlsx", "xls"]
    
    @field_validator("ALLOWED_EXTENSIONS", mode="before")
    @classmethod
    def assemble_extensions(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
