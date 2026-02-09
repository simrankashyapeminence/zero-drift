from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
import shutil
from app.core.config import settings
from loguru import logger

router = APIRouter()

# @router.post("/excel")
# async def upload_excel(file: UploadFile = File(...)):
#     if not file.filename.endswith(tuple(settings.ALLOWED_EXTENSIONS)):
#         raise HTTPException(status_code=400, detail="File type not allowed")
#     
#     file_path = os.path.join(settings.EXCEL_DIR, file.filename)
#     try:
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#         logger.info(f"Excel file uploaded: {file.filename}")
#         return {"filename": file.filename, "status": "uploaded"}
#     except Exception as e:
#         logger.error(f"Excel upload failed: {str(e)}")
#         raise HTTPException(status_code=500, detail="Internal server error")
# 
# @router.post("/images")
# async def upload_images(files: List[UploadFile] = File(...)):
#     uploaded_files = []
#     for file in files:
#         if not file.filename.lower().endswith(tuple(settings.ALLOWED_EXTENSIONS)):
#             logger.warning(f"File {file.filename} color filtered: extension not allowed.")
#             continue
#             
#         file_path = os.path.join(settings.IMAGES_DIR, file.filename)
#         try:
#             with open(file_path, "wb") as buffer:
#                 shutil.copyfileobj(file.file, buffer)
#             uploaded_files.append(file.filename)
#         except Exception as e:
#             logger.error(f"Failed to upload {file.filename}: {str(e)}")
#             
#     logger.info(f"Uploaded {len(uploaded_files)} images.")
#     return {"uploaded_count": len(uploaded_files), "filenames": uploaded_files}
