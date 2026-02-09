from fastapi import Request, status
from fastapi.responses import JSONResponse
from loguru import logger
import time
import uuid

async def log_request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    # Add request id to the loguru context
    with logger.contextualize(request_id=request_id):
        start_time = time.time()
        
        # Log request details
        logger.info(f"Incoming request: {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            
            process_time = (time.time() - start_time) * 1000
            formatted_process_time = "{0:.2f}".format(process_time)
            
            logger.info(f"Completed request: {request.method} {request.url.path} - Status: {response.status_code} - Duration: {formatted_process_time}ms")
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            return response
            
        except Exception as e:
            logger.exception(f"Unhandled exception occurred: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "An internal server error occurred.", "request_id": request_id}
            )

def setup_exception_handlers(app):
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Global Exception: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "message": str(exc)}
        )
