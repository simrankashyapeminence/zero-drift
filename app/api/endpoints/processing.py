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
        total_tasks = len(unique_product_codes)
        
        if len(mapped_metadata) == 2:
            # Dual mode always results in 1 task
            total_tasks = 1

        jobs[job_id] = {
            "status": "running",
            "total_items": total_tasks,
            "completed_items": 0,
            "failed_items": 0,
            "results": []
        }

        # ---------------------------------------------------------
        # 5️. Background Processing Logic
        # ---------------------------------------------------------
        async def run_and_track():
            try:
                logger.info(f"🚀 Job {job_id} started")

                results = []

                # 🔹 Dual Product Mode
                if len(mapped_metadata) == 2:
                    p1, p2 = mapped_metadata[0], mapped_metadata[1]
                    name1, name2 = p1.product_name.lower(), p2.product_name.lower()
                    
                    top_keywords = ["rashguard", "shirt", "top", "hoodie", "jacket", "bra", "tank"]
                    bottom_keywords = ["leggings", "pants", "shorts", "tights", "trousers"]
                    
                    upper, lower = (p1, p2) if any(k in name1 for k in top_keywords) else (p2, p1)
                    
                    logger.info(f"👕 Dual Generation Mode | {upper.product_code} + {lower.product_code}")
                    
                    res_path = await nano_service.generate_dual_tryon(upper, lower, variation_index=0)
                    results.append(res_path)
                    
                    # Update Progress
                    jobs[job_id]["completed_items"] = 1
                    jobs[job_id]["results"] = results

                # 🔹 Single Product Mode
                else:
                    logger.info(f"🧍 Single Product Generation Mode | {total_tasks} products")
                    results = []
                    
                    # Group by product code
                    grouped = {}
                    for p in mapped_metadata:
                        if p.product_code not in grouped:
                            grouped[p.product_code] = []
                        grouped[p.product_code].append(p)
                    
                    for i, (code, group) in enumerate(grouped.items()):
                        image_paths = [os.path.join(settings.IMAGES_DIR, p.image_filename) for p in group]
                        logger.info(f"🔄 Processing {code} ({i+1}/{total_tasks})")
                        
                        try:
                            res_path = await nano_service.generate_tryon_image(group[0], image_paths, variation_index=0)
                            results.append(res_path)
                            
                            # IMMEDIATE PROGRESS UPDATE
                            jobs[job_id]["completed_items"] = len(results)
                            jobs[job_id]["results"] = results
                        except Exception as e:
                            logger.error(f"❌ Failed {code}: {e}")
                            jobs[job_id]["failed_items"] += 1

                # -------------------------------------------------
                # 6️.Update Job State
                # -------------------------------------------------
                jobs[job_id]["status"] = "completed"
                # Results and completed_items are already updated in the loop

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
