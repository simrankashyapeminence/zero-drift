from pydantic import BaseModel
from typing import Optional, List

class ProductMetadata(BaseModel):
    product_code: str
    product_name: str
    product_type: str
    image_filename: Optional[str] = None
    status: str = "pending"
    result_url: Optional[str] = None

class BatchProcessRequest(BaseModel):
    excel_filename: str
    image_filenames: List[str]
