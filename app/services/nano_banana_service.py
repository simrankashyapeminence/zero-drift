import httpx
import base64
import asyncio
import time
from loguru import logger
from app.core.config import settings
from app.models.metadata import ProductMetadata
import os


class NanoBananaService:
    def __init__(self):
        self.api_key = settings.NANO_BANANA_API_KEY
        self.base_url = settings.NANO_BANANA_BASE_URL

        self.headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        masked = (
            f"{self.api_key[:6]}...{self.api_key[-4:]}"
            if self.api_key
            else "MISSING"
        )
        logger.info(f"🔑 Gemini service initialized | Key: {masked}")
        logger.info(f"🌐 Base URL: {self.base_url}")

    async def generate_tryon_image(self, product: ProductMetadata, image_paths: list[str], variation_index: int = 0) -> str:
        """
        Calls the Nano Banana Pro (Gemini) API for virtual try-on using one or more reference images.
        """
        max_retries = 3
        retry_delay = 2 

        # Dynamic Pose Logic with Close-ups
        poses = [
            "dynamic mid-action strike or high-energy movement",
            "macro close-up shot focusing on the chest logo and neckline detail",
            "intense athletic focus in a powerful stance",
            "close-up zoom on the fabric texture and shoulder stitching"
        ]
        selected_pose = poses[variation_index % len(poses)]
        is_close_up = "close-up" in selected_pose.lower() or "macro" in selected_pose.lower()

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}: Generating image (Var {variation_index + 1}) for {product.product_code}")
                
                parts = []
                # 1. Add Prompt
                prompt = (
                    f"Task: High-Fidelity Athlete Catalog [Variation {variation_index + 1}]. PRODUCT: {product.product_name} for {product.sport}. "
                    
                    "\n[STRICT LOCK - MODEL IDENTITY]: "
                    f"You MUST use the EXACT same {product.gender} model for this product. "
                    "The face, facial features, hair, skin tone, and body build must be identical. "
                    
                    "\n[CAMERA & FRAMING]: "
                    + ("Macro details shot. " if is_close_up else "Standard shot. ") + 
                    "STRICT: Generate ONE SINGLE model. "
                    "CRITICAL FACE VISIBILITY: The model's COMPLETE FACE, HEAD, and HAIR must be 100% visible with significant headroom. "
                    "NEVER crop the eyes, mouth, face, or hair. The entire face must be shown clearly and looking towards the camera or slightly away. "
                    "UNBROKEN VIEW: Do not cut off the head at the top of the frame. Even in close-ups, the face is the most important for realism."
 
                    f"\n[DYNAMIC SPORT ACTION - UNIQUE POSE]: Pose: {selected_pose}. "
                    "The model must be captured in a singular, powerful athletic moment. NO multi-view collages."

                    "\n[ZERO-DRIFT GARMENT FIDELITY - ABSOULTE REQUIREMENT]: "
                    "The garment must be a 100% EXACT CLONE of the source image(s). "
                    "1. BRAND LOGO INTEGRITY (SMMASH): The brand logo 'SMMASH' MUST be 100% perfectly preserved in EVERY frame, including wide or zoomed-out shots. Even from a distance, the logo must remain sharp, crisp, and perfectly legible. Every letter, font weight, and spacing of 'SMMASH' must be professional and identical to the reference. NO simplification or blurring of the logo in wide shots. "
                    "2. MACRO STITCHING & SEAMS: Every stitch line, seam, and hem must follow the exact geometry of the original product. "
                    "3. PRODUCT SHAPE & FIT: Maintain the exact silhouette, neckline, sleeve length, and fabric drape as seen in the references. "
                    "4. COLOR & TEXTURE: Use the exact HEX colors and fabric weave (e.g., micro-mesh, ribbed, or dry-fit textures)."

                    "\nQuality: Professional 8K photography, DSLR sharp focus, macro-detailed fabric textures, high-clarity branding."
                )
                parts.append({"text": prompt})

                # 2. Add Images
                from app.utils.image_processor import ImageProcessor
                for img_path in image_paths:
                    if not os.path.exists(img_path):
                        logger.error(f"Image not found: {img_path}")
                        continue
                    
                    # Using 4K for maximum logo and stitching detail
                    final_image_path = ImageProcessor.optimize_for_api(img_path, max_size=(4096, 4096))
                    with open(final_image_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode("utf-8")

                    mime_type = "image/png" if final_image_path.lower().endswith(".png") else "image/jpeg"
                    parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": img_data
                        }
                    })

                payload = {
                    "contents": [{"parts": parts}],
                    "generationConfig": {
                        "temperature": 0.3, # Slightly increased for pose variety
                        "topP": 0.99,
                        "topK": 100,
                        "maxOutputTokens": 32768,
                    }
                }

                endpoint = f"{self.base_url}/models/{settings.GEMINI_MODEL_VERSION}:generateContent"
                
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(endpoint, headers=self.headers, json=payload)
                    
                    if response.status_code == 429:
                        logger.warning(f"Rate limited. Retrying...")
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue

                    if response.status_code != 200:
                        logger.error(f"API Error ({response.status_code}): {response.text}")
                        raise ValueError(f"API failed for {product.product_code}")

                    result_data = response.json()
                    
                    # Log Usage
                    usage = result_data.get("usageMetadata", {})
                    logger.info(f"📊 Tokens used: {usage.get('totalTokenCount', 0)}")

                    output_image_b64 = None
                    try:
                        candidate = result_data["candidates"][0]
                        content = candidate.get("content", {})
                        parts_resp = content.get("parts", [])
                        
                        text_response = ""
                        for part in parts_resp:
                            if "inline_data" in part or "inlineData" in part:
                                output_image_b64 = part.get("inline_data", part.get("inlineData"))["data"]
                                break
                            elif "text" in part:
                                text_response += part["text"]
                                
                        if not output_image_b64:
                            finish_reason = candidate.get("finishReason")
                            safety_ratings = candidate.get("safetyRatings", [])
                            logger.error(f"❌ No image content in part. Finish Reason: {finish_reason} | Safety: {safety_ratings}")
                            if text_response:
                                logger.warning(f"⚠️ Model returned text instead of image: {text_response[:500]}")
                            else:
                                logger.error(f"Full response for debugging: {result_data}")
                            raise ValueError("No image data in response.")
                            
                    except Exception as e:
                        if "No image data" in str(e):
                            raise e
                        logger.error(f"❌ Extraction failed: {str(e)}")
                        logger.debug(f"Full response data: {result_data}")
                        raise e

                    output_filename = f"res_{product.product_code}_var{variation_index}_{int(time.time())}.png"
                    output_path = os.path.join(settings.EXPORT_DIR, output_filename)
                    
                    with open(output_path, "wb") as f:
                        f.write(base64.b64decode(output_image_b64))

                    logger.success(f"Generated variation {variation_index + 1}: {output_filename}")
                    return output_path

            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(retry_delay * (attempt + 1))

    async def generate_outfit_image(self, products: list[ProductMetadata], variation_index: int = 0) -> str:
        """
        Calls Gemini with multiple reference images to generate a single cohesive outfit in a specific pose.
        """
        max_retries = 3
        retry_delay = 5

        # Dynamic Pose Logic with Close-ups
        poses = [
            "full-body heroic pose showing the complete outfit silhouette",
            "waist-up medium-close shot showing the top branding and stitching",
            "dynamic athletic action movement",
            "close-up focusing on the mid-section showing fabric integration"
        ]
        selected_pose = poses[variation_index % len(poses)]
        is_close_up = "close-up" in selected_pose.lower() or "medium-close" in selected_pose.lower()

        # Group products to describe them in the prompt
        upper_keywords = ["GÓRA", "KOSZULKA", "SHIRT", "SWEATSHIRT", "JACKET", "TOP"]
        lower_keywords = ["DÓŁ", "SPODNIE", "PANTS", "LEGGINSY", "LEGGINGS", "SHORTS", "SUKIENKA"]

        upper_wear = [p for p in products if any(k in str(p.product_type).upper() or k in str(p.product_name).upper() for k in upper_keywords)]
        lower_wear = [p for p in products if any(k in str(p.product_type).upper() or k in str(p.product_name).upper() for k in lower_keywords)]
        
        if not upper_wear or not lower_wear:
            # Fallback for 2 unique products
            unique_codes = list(set([p.product_code for p in products]))
            if len(unique_codes) == 2:
                upper_wear = [p for p in products if p.product_code == unique_codes[0]]
                lower_wear = [p for p in products if p.product_code == unique_codes[1]]
            else:
                mid = len(products) // 2
                upper_wear = products[:mid]
                lower_wear = products[mid:]

        sport = products[0].sport
        gender = products[0].gender

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}: Generating combined outfit image (Var {variation_index + 1})")
                
                parts = []
                prompt = (
                    f"Task: High-Fidelity Athlete Outfit Generation [Variation {variation_index + 1}]. PRODUCT SET: Combined {sport} outfit for {gender}. "
                    "\n[COMPOSITION]: "
                    f"You are provided with {len(products)} total reference images to create a SINGLE result. "
                    f"- {len(upper_wear)} image(s) show the UPPER WEAR. "
                    f"- {len(lower_wear)} image(s) show the LOWER WEAR. "
                    
                    "\n[STRICT LOCK - MODEL IDENTITY]: "
                    f"Use a professional athletic {gender} model. Face and body build must be photorealistic. "
                    
                    "\n[CAMERA & FRAMING]: "
                    + ("Upper-body focus. " if is_close_up else "Full-body shot. ") +
                    "STRICT: Generate ONE SINGLE model. "
                    "MODEL FACE INTEGRITY: The model's FULL UNBROKEN FACE, HEAD, AND HAIR must be visible in every single frame. "
                    "NEVER crop or cut off the model's head or face. Significant headroom is REQUIRED. "
                    "The face must be looking forward or in a natural athletic direction, but always fully composed within the frame."
                    
                    f"\n[DYNAMIC SPORT ACTION - UNIQUE POSE]: Pose: {selected_pose}. "
                    f"Show the model in a professional, authentic {sport} pose or action. NO split-screens."
                    
                    "\n[ZERO-DRIFT GARMENT FIDELITY - ABSOLUTE REQUIREMENT]: "
                    "The model must be wearing BOTH the provided upper wear and lower wear. "
                    "1. BRAND LOGO INTEGRITY (SMMASH): The logo 'SMMASH' on both top and bottom must be output with 100% accuracy, especially in full-body/zoomed-out views. Every character in 'SMMASH' must be crisp, perfectly legible, and identical to the source from any distance. Zero tolerance for ruined text or logo simplification in wide shots. "
                    "2. MACRO STITCHING & GEOMETRY: Maintain every stitch line, seam pattern, and hem design for BOTH products. "
                    "3. PRODUCT SHAPE & FIT: Ensure the exact silhouette, neckline, and fit for both the upper and lower garments."
                    
                    "\nQuality: Professional 8K photography, DSLR sharp focus, macro-detailed textures, high-clarity branding."
                )
                parts.append({"text": prompt})

                from app.utils.image_processor import ImageProcessor
                for p in products:
                    image_path = os.path.join(settings.IMAGES_DIR, p.image_filename)
                    opt_path = ImageProcessor.optimize_for_api(image_path, max_size=(4096, 4096))
                    with open(opt_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode("utf-8")
                    mime_type = "image/png" if opt_path.lower().endswith(".png") else "image/jpeg"
                    parts.append({"inline_data": {"mime_type": mime_type, "data": img_data}})

                payload = {
                    "contents": [{"parts": parts}],
                    "generationConfig": {"temperature": 0.3, "topP": 0.99, "topK": 100, "maxOutputTokens": 32768}
                }

                endpoint = f"{self.base_url}/models/{settings.GEMINI_MODEL_VERSION}:generateContent"
                async with httpx.AsyncClient(timeout=180.0) as client:
                    response = await client.post(endpoint, headers=self.headers, json=payload)
                    if response.status_code != 200: raise ValueError(f"API Error ({response.status_code}): {response.text}")

                    result_data = response.json()
                    candidate = result_data["candidates"][0]
                    content = candidate.get("content", {})
                    output_image_b64 = None
                    for part in content.get("parts", []):
                        if "inline_data" in part or "inlineData" in part:
                            output_image_b64 = part.get("inline_data", part.get("inlineData"))["data"]
                            break
                    
                    if not output_image_b64: raise ValueError("No image data in response.")

                    output_filename = f"res_outfit_{int(time.time())}.png"
                    output_path = os.path.join(settings.EXPORT_DIR, output_filename)
                    with open(output_path, "wb") as f:
                        f.write(base64.b64decode(output_image_b64))

                    logger.success(f"Generated Outfit Image: {output_filename}")
                    return output_path

            except Exception as e:
                logger.error(f"Error on outfit attempt {attempt + 1}: {str(e)}")
                await asyncio.sleep(retry_delay)
        
        raise Exception("Failed to generate outfit image.")

    async def batch_process(self, products: list[ProductMetadata]):
        """
        Intelligently groups products by code and generates outputs with varied poses.
        - Generates 3 unique variations for each product or outfit.
        """
        if not products:
            logger.warning("No products provided.")
            return []

        # 1. Group by code
        grouped = {}
        for p in products:
            if p.product_code not in grouped:
                grouped[p.product_code] = []
            grouped[p.product_code].append(p)
        
        unique_codes = list(grouped.keys())
        num_products = len(unique_codes)
        target_variations = 4 # Generates 2 action/hero shots and 2 close-up shots
        
        results = []

        # Case 1: Exactly 2 different products -> Try to make an outfit
        if num_products == 2:
            logger.info(f"🎯 2 products detected. Generating {target_variations} unique outfit poses...")
            for v_idx in range(target_variations):
                try:
                    result_path = await self.generate_outfit_image(products, variation_index=v_idx)
                    results.append(result_path)
                    # Small throttle between variations
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"Outfit pose {v_idx+1} failed: {e}")
            return results
        
        # Case 2: Individual products (single or multiple unique codes)
        logger.info(f"🚀 Batching {num_products} unique products. Target: {target_variations} poses each.")
        for code, group in grouped.items():
            image_paths = [os.path.join(settings.IMAGES_DIR, p.image_filename) for p in group]
            for v_idx in range(target_variations):
                logger.info(f"🔄 Generating {code} | Pose Variation {v_idx+1}/{target_variations}")
                try:
                    result_path = await self.generate_tryon_image(group[0], image_paths, variation_index=v_idx)
                    results.append(result_path)
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"Failed Variation {v_idx+1} for {code}: {e}")

        return results

