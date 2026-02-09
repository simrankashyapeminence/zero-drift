import pandas as pd
from loguru import logger
from typing import List, Dict
from app.models.metadata import ProductMetadata
import os

class ExcelService:
    @staticmethod
    def parse_metadata(file_path: str) -> List[ProductMetadata]:
        try:
            logger.info(f"Parsing Excel file: {file_path}")
            df = pd.read_excel(file_path)
            
            # Normalize column names to lowercase and remove spaces
            df.columns = [str(col).lower().strip().replace(' ', '_') for col in df.columns]
            
            # Mapping logic: look for product_code, product_name, type_of_product
            # Adjust mapping based on user description: "Product code, product name, type of product"
            column_mapping = {
                'product_code': ['product_code', 'code', 'p_code', 'sku'],
                'product_name': ['product_name', 'name', 'title'],
                'product_type': ['product_type', 'type', 'category', 'type_of_product']
            }
            
            results = []
            for index, row in df.iterrows():
                metadata = {}
                row_values = [str(v).strip().upper() for v in row.values]
                
                # Primary mapping for structured fields
                for key, variations in column_mapping.items():
                    for var in variations:
                        if var in df.columns:
                            metadata[key] = str(row[var])
                            break
                    if key not in metadata:
                        # Fallback: if we can't find a column, use first column for code if empty
                        metadata[key] = "N/A"
                
                # Keep the full processed row for "search anywhere" matching
                metadata['raw_row_values'] = row_values
                results.append(metadata)
                
            logger.success(f"Successfully parsed {len(results)} rows. Columns found: {list(df.columns)}")
            return results
        except Exception as e:
            logger.error(f"Error parsing Excel file: {str(e)}")
            raise e

    @staticmethod
    def map_images_to_metadata(metadata_list: List[Dict], image_files: List[str]) -> List[ProductMetadata]:
        """
        Search EVERY cell in EVERY row for the image filename.
        If found, intelligently extract name and type from surrounding cells 
        if headers are missing.
        """
        logger.info(f"🔍 [DEEP SEARCH] Starting mapping for {len(image_files)} images.")
        mapped_results = []
        
        for img_filename in image_files:
            # Normalize image name: CS6-EUW.jpg -> CS6-EUW
            clean_img_name = os.path.splitext(img_filename)[0].strip().upper()
            match_found = False
            
            for row_dict in metadata_list:
                row_values = row_dict['raw_row_values'] # These are already uppercase stripped strings
                
                if clean_img_name in row_values:
                    # We found the row! Now let's extract details.
                    # 1. Try to use detected headers first
                    p_code = row_dict.get('product_code')
                    p_name = row_dict.get('product_name')
                    p_type = row_dict.get('product_type')

                    # 2. If headers failed (returned N/A), use position-based logic
                    # Usually: [Code] [Name] [Type] [Extra...]
                    if p_name == "N/A" or not p_name:
                        # Find where the code was in the row
                        try:
                            code_idx = row_values.index(clean_img_name)
                            # Take the next cell for name if it exists
                            if code_idx + 1 < len(row_values):
                                p_name = row_values[code_idx + 1]
                            # Take the cell after that for type
                            if code_idx + 2 < len(row_values):
                                p_type = row_values[code_idx + 2]
                        except ValueError:
                            pass

                    # Final cleanup of extracted values
                    final_code = p_code if p_code != "N/A" else clean_img_name
                    final_name = p_name if p_name != "N/A" else "Fashion Item"
                    final_type = p_type if p_type != "N/A" else "Garment"

                    meta_obj = ProductMetadata(
                        product_code=final_code,
                        product_name=final_name,
                        product_type=final_type,
                        image_filename=img_filename
                    )

                    mapped_results.append(meta_obj)
                    logger.success(f"🎯 Match & Extract: '{img_filename}' -> '{final_name}' ({final_type})")
                    match_found = True
                    break 
            
            if not match_found:
                logger.warning(f"🚫 No Match: '{clean_img_name}' not found anywhere in the Excel data.")

        return mapped_results
