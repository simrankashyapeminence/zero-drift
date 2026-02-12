from fastapi import FastAPI
from app.api.endpoints import upload, processing
from app.core.config import settings
from app.core.logging_config import setup_logging
from loguru import logger
from app.core.middleware import log_request_middleware, setup_exception_handlers
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware

# Initialize Logging
setup_logging()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-grade image generation service using Nano Banana API with Zero-Drift technology."
)

# Add Middleware
app.add_middleware(BaseHTTPMiddleware, dispatch=log_request_middleware)
setup_exception_handlers(app)

# Add CORS last so it runs first (outermost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "app": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online"
    }

# Include Routers
# app.include_router(upload.router, prefix=f"{settings.API_V1_STR}/upload", tags=["Upload"])
app.include_router(processing.router, prefix=f"{settings.API_V1_STR}/processing", tags=["Processing"])

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.PROJECT_NAME}...")
    # Ensure necessary folders exist
    import os
    for folder in [settings.UPLOAD_DIR, settings.IMAGES_DIR, settings.EXCEL_DIR, settings.EXPORT_DIR]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            logger.debug(f"Created folder: {folder}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
