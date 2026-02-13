from fastapi import APIRouter, BackgroundTasks, HTTPException, File, UploadFile
from app.services.excel_service import ExcelService
from app.services.nano_banana_service import NanoBananaService
from app.core.config import settings
from app.models.metadata import BatchProcessRequest
from loguru import logger
import os
import time
import shutil
from typing import List, Any

router = APIRouter()
nano_service = NanoBananaService()

# In-memory job store
jobs = {}

# Ensure directories exist
os.makedirs(settings.EXCEL_DIR, exist_ok=True)
os.makedirs(settings.IMAGES_DIR, exist_ok=True)
os.makedirs(settings.EXPORT_DIR, exist_ok=True)


@router.post("/upload")
async def unified_upload_and_process(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    excel_file: UploadFile = File(...)
):
    """
    Upload Excel + Images, auto-detect single or dual generation,
    and start background processing.
    """

    job_id = f"job_{int(time.time())}"

    try:
        if not images:
            raise HTTPException(status_code=400, detail="No valid images uploaded.")

        # ---------------------------------------------------------
        # 1.Validate Excel
        # ---------------------------------------------------------
        if not excel_file.filename.endswith(".xlsx"):
            raise HTTPException(status_code=400, detail="Only .xlsx files are allowed")

        unique_prefix = str(int(time.time()))

        excel_filename = f"{unique_prefix}_{excel_file.filename}"
        excel_path = os.path.join(settings.EXCEL_DIR, excel_filename)

        with open(excel_path, "wb") as buffer:
            shutil.copyfileobj(excel_file.file, buffer)

        # ---------------------------------------------------------
        # 2️. Validate and Save Images
        # ---------------------------------------------------------
        allowed_types = ["image/jpeg", "image/png"]
        saved_images = []

        for img in images:
            if img.content_type not in allowed_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image type: {img.filename}"
                )

            img_filename = f"{unique_prefix}_{img.filename}"
            img_path = os.path.join(settings.IMAGES_DIR, img_filename)

            with open(img_path, "wb") as buffer:
                shutil.copyfileobj(img.file, buffer)

            saved_images.append(img_filename)

        # ---------------------------------------------------------
        # 3️.Parse Excel + Map Images
        # ---------------------------------------------------------
        metadata_list = ExcelService.parse_metadata(excel_path)
        mapped_metadata = ExcelService.map_images_to_metadata(
            metadata_list,
            saved_images
        )

        if not mapped_metadata:
            return {
                "job_id": job_id,
                "status": "failed",
                "detail": "No images matched Excel data"
            }

        # ---------------------------------------------------------
        # 4️. Initialize Job State
        # ---------------------------------------------------------
        unique_product_codes = list(set(p.product_code for p in mapped_metadata))
        num_unique = len(unique_product_codes)
        
        # Calculate expected tasks based on unique product count
        if num_unique == 1:
            total_tasks = 1  # Single product → 1 output
        elif num_unique == 2:
            total_tasks = 1  # Dual outfit → 1 combined output
        else:
            total_tasks = num_unique  # Multiple → 1 output per product

        jobs[job_id] = {
            "status": "running",
            "total_items": total_tasks,
            "completed_items": 0,
            "failed_items": 0,
            "results": []
        }

        # ---------------------------------------------------------
        # 5️. Background Processing Logic
        #    Delegates entirely to nano_service.batch_process()
        #    which handles single/dual/multi product routing.
        # ---------------------------------------------------------
        async def run_and_track():
            try:
                logger.info(f"🚀 Job {job_id} started | {num_unique} unique product(s) from {len(mapped_metadata)} image(s)")

                results = await nano_service.batch_process(mapped_metadata)

                # Update job state
                jobs[job_id]["completed_items"] = len(results)
                jobs[job_id]["results"] = results
                jobs[job_id]["failed_items"] = total_tasks - len(results)
                jobs[job_id]["status"] = "completed"

            except Exception as e:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = str(e)

        # Add to background
        background_tasks.add_task(run_and_track)

        return {
            "job_id": job_id,
            "status": "processing_started",
            "items_count": total_tasks,
            "mapped_products": unique_product_codes
        }

    except Exception as e:
        logger.error(f"Unified process failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return jobs[job_id]
