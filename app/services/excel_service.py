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
            
            # Normalize column names 
            df.columns = [str(col).lower().strip().replace(' ', '_') for col in df.columns]
            
            column_mapping = {
                'product_code': ['product_code', 'code', 'p_code', 'sku', 'nazwa_bazowa', 'nr_artykułu', 'id'],
                'product_name': ['product_name', 'name', 'title', 'nazwa_handlowa', 'opis'],
                'product_type': ['product_type', 'type', 'category', 'type_of_product', 'rodzaj'],
                'gender': ['płeć', 'gender', 'sex'],
                'sport': ['sport_dominujący', 'sport', 'discipline']
            }
            
            results = []
            for index, row in df.iterrows():
                metadata = {}
                row_values = [str(v).strip().upper() for v in row.values]
                
                for key, variations in column_mapping.items():
                    for var in variations:
                        if var in df.columns:
                            metadata[key] = str(row[var])
                            break
                    if key not in metadata:
                        metadata[key] = "N/A"
                
                metadata['raw_row_values'] = row_values
                results.append(metadata)
                
            logger.success(f"Successfully parsed {len(results)} rows.")
            # Debug: Log first 3 rows to see data format
            for i, res in enumerate(results[:3]):
                logger.debug(f"Row {i} Data: Code={res['product_code']} | Sport={res['sport']} | Gender={res['gender']} | Values={res['raw_row_values']}")
                
            return results
        except Exception as e:
            logger.error(f"Error parsing Excel file: {str(e)}")
            raise e

    @staticmethod
    def map_images_to_metadata(metadata_list: List[Dict], image_files: List[str]) -> List[ProductMetadata]:
        import re
        def deep_clean(text: str) -> str:
            return re.sub(r'[^A-Z0-9]', '', str(text).upper())

        logger.info(f"🔍 [MATCHING] Processing {len(image_files)} images against {len(metadata_list)} metadata rows.")
        mapped_results = []
        
        for img_filename in image_files:
            # "RSO3-1.png" -> "RSO31"
            img_base = os.path.splitext(img_filename)[0]
            clean_img_name = deep_clean(img_base)
            match_found = False
            
            for row_dict in metadata_list:
                p_code = row_dict.get('product_code', '')
                p_name = row_dict.get('product_name', '')
                all_vals = [p_code, p_name] + row_dict['raw_row_values']
                
                for raw_val in all_vals:
                    if not raw_val or str(raw_val).upper() == "N/A":
                        continue
                        
                    clean_val = deep_clean(raw_val)
                    if len(clean_val) < 2: # Allow shorter codes like "K1"
                        continue
                    
                    # Match logic:
                    # 1. Exact clean match
                    # 2. Excel code is at the START of the filename (e.g. "RSO3" in "RSO3-1")
                    # 3. Filename is part of Excel code (e.g. "RSO3" in "RSO3_BLACK")
                    if clean_val == clean_img_name or \
                       clean_img_name.startswith(clean_val) or \
                       clean_val.startswith(clean_img_name):
                        
                        final_name = str(row_dict.get('product_name', 'Fashion Item'))
                        final_type = str(row_dict.get('product_type', 'Garment'))
                        final_gender = str(row_dict.get('gender', 'N/A'))
                        final_sport = str(row_dict.get('sport', 'N/A'))
                        
                        meta_obj = ProductMetadata(
                            product_code=str(p_code) if p_code != "N/A" else clean_val,
                            product_name=final_name if final_name != "N/A" else "Product",
                            product_type=final_type if final_type != "N/A" else "Fashion",
                            gender=final_gender,
                            sport=final_sport,
                            image_filename=img_filename
                        )
                        mapped_results.append(meta_obj)
                        logger.success(f"🎯 MATCH FOUND: '{img_filename}' matches '{raw_val}'")
                        match_found = True
                        break
                
                if match_found:
                    break
            
            if not match_found:
                # Try splitting by common delimiters and check prefixes
                parts = re.split(r'[-_ ]+', img_base)
                if parts:
                    prefix = deep_clean(parts[0])
                    if len(prefix) >= 2:
                        for row_dict in metadata_list:
                            for raw_val in [row_dict.get('product_code', '')] + row_dict['raw_row_values']:
                                if deep_clean(raw_val) == prefix:
                                    mapped_results.append(ProductMetadata(
                                        product_code=str(raw_val),
                                        product_name=str(row_dict.get('product_name', 'Fashion Item')),
                                        product_type=str(row_dict.get('product_type', 'Garment')),
                                        gender=str(row_dict.get('gender', 'N/A')),
                                        sport=str(row_dict.get('sport', 'N/A')),
                                        image_filename=img_filename
                                    ))
                                    logger.success(f"🎯 PREFIX MATCH: '{img_filename}' matched by prefix '{prefix}'")
                                    match_found = True
                                    break
                            if match_found: break

            if not match_found:
                logger.warning(f"🚫 NO MATCH for '{img_filename}'")

        return mapped_results
