from fastapi import APIRouter, BackgroundTasks, HTTPException, File, UploadFile
from app.services.excel_service import ExcelService
from app.services.nano_banana_service import NanoBananaService
from app.core.config import settings
from app.models.metadata import BatchProcessRequest
from loguru import logger
import os
import time
import shutil
from typing import List

router = APIRouter()
nano_service = NanoBananaService()

# In-memory job store (for production, use Redis or a DB)
jobs = {}

# @router.post("/process")
# async def start_processing(request: BatchProcessRequest, background_tasks: BackgroundTasks):
#     excel_path = os.path.join(settings.EXCEL_DIR, request.excel_filename)
#     job_id = f"job_{int(time.time())}"
#     
#     if not os.path.exists(excel_path):
#         raise HTTPException(status_code=404, detail="Excel file not found")
#     
#     try:
#         # 1. Parse Excel
#         metadata_list = ExcelService.parse_metadata(excel_path)
#         
#         # 2. Map Images
#         mapped_metadata = ExcelService.map_images_to_metadata(metadata_list, request.image_filenames)
#         
#         # Initial Job State
#         jobs[job_id] = {
#             "status": "running",
#             "total_items": len(mapped_metadata),
#             "completed_items": 0,
#             "failed_items": 0,
#             "results": []
#         }
# 
#         async def run_and_track():
#             try:
#                 results = await nano_service.batch_process(mapped_metadata)
#                 jobs[job_id]["status"] = "completed"
#                 jobs[job_id]["results"] = results
#                 jobs[job_id]["completed_items"] = len(results)
#                 jobs[job_id]["failed_items"] = len(mapped_metadata) - len(results)
#                 logger.info(f"Job {job_id} finished successfully.")
#             except Exception as e:
#                 jobs[job_id]["status"] = "failed"
#                 jobs[job_id]["error"] = str(e)
#                 logger.error(f"Job {job_id} failed: {e}")
# 
#         # 3. Queue Generation
#         background_tasks.add_task(run_and_track)
#         
#         logger.info(f"Processing started for job {job_id}.")
#         return {
#             "job_id": job_id,
#             "status": "processing_started",
#             "items_count": len(mapped_metadata)
#         }
#         
#     except Exception as e:
#         logger.error(f"Processing failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def unified_upload_and_process(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    excel_file: UploadFile = File(...)
):
    """
    Unified endpoint: Uploads Images + Excel, maps them, and starts generation.
    """
    job_id = f"job_unified_{int(time.time())}"
    logger.info(f"📥 Received unified request. Job: {job_id}")

    # 1. Save Excel
    excel_path = os.path.join(settings.EXCEL_DIR, excel_file.filename)
    with open(excel_path, "wb") as buffer:
        shutil.copyfileobj(excel_file.file, buffer)
    
    # 2. Save Images
    saved_images = []
    for img in images:
        img_path = os.path.join(settings.IMAGES_DIR, img.filename)
        with open(img_path, "wb") as buffer:
            shutil.copyfileobj(img.file, buffer)
        saved_images.append(img.filename)
    
    logger.info(f"✅ Files saved: 1 Excel, {len(saved_images)} Images.")

    try:
        # 3. Parse and Map
        metadata_list = ExcelService.parse_metadata(excel_path)
        mapped_metadata = ExcelService.map_images_to_metadata(metadata_list, saved_images)
        
        if not mapped_metadata:
            return {"job_id": job_id, "status": "failed", "detail": "No images matched the excel data."}

        # 4. Job State
        jobs[job_id] = {
            "status": "running",
            "total_items": len(mapped_metadata),
            "completed_items": 0,
            "failed_items": 0,
            "results": []
        }

        async def run_and_track():
            try:
                results = await nano_service.batch_process(mapped_metadata)
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["results"] = results
                jobs[job_id]["completed_items"] = len(results)
                jobs[job_id]["failed_items"] = len(mapped_metadata) - len(results)
            except Exception as e:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = str(e)

        background_tasks.add_task(run_and_track)
        
        return {
            "job_id": job_id,
            "status": "processing_started",
            "items_count": len(mapped_metadata),
            "mapped_products": [p.product_code for p in mapped_metadata]
        }

    except Exception as e:
        logger.error(f"Unified process failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return jobs[job_id]
