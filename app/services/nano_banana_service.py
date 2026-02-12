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
                    f"Task: High-Fidelity Athlete Catalog [View {variation_index + 1}]. PRODUCT: {product.product_name} for {product.sport}. "
                    
                    + (f"\n[ACTION & POSE]: {product.pose} " if product.pose != "N/A" else f"\n[DYNAMIC SPORT ACTION]: Pose: {selected_pose}. ") +
                    (f"\n[ENVIRONMENT]: {product.environment} " if product.environment != "N/A" else "") +
                    
                    "\n[STRICT REQUIREMENT - MODEL IDENTITY]: "
                    f"Use the EXACT same {product.gender} model. Face, facial features (including beard/facial hair structure), and hair must be IDENTICAL to the reference. No modifications allowed."
                    
                    "\n[CRITICAL - ZERO-DRIFT GARMENT REPLICATION]: "
                    "The garment must be a 100% pixel-perfect photographic clone of the source image. "
                    "1. LOGO & TEXT LOCK: The brand name 'SMMASH' must be IDENTICAL to the source. No character changes, no font changes, no spacing changes. If the logo is sharp in the source, it MUST be sharp in the output. "
                    "2. SURFACE PURITY: If a surface (sleeves, back, chest) is blank in the source, it MUST remain blank. DO NOT add any extra logos. "
                    "3. BRANDING COORDINATES: All branding must stay in its EXACT original position. No relocation. "
                    "4. NO INTERPRETATION: Do not 'improve' or 're-draw' the logo. Mirror the pixels exactly. "

                    "\n[CAMERA & FRAMING]: "
                    + ("Macro details shot. " if is_close_up else "Standard shot. ") + 
                    "Generate ONE SINGLE model. Full face and head must be visible. Significant headroom required. "

                    f"\n[DYNAMIC SPORT ACTION]: Pose: {selected_pose}. "
                    "Capture a singular professional movement. No design changes. No collages."
                    
                    "\nQuality: Professional 8K photography, DSLR ultra-sharp focus. ZERO AI HALLUCINATION TOLERANCE."
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
                        "temperature": 0.1, 
                        "topP": 0.1, # Drastically reduced to prevent creativity
                        "topK": 10,  # Drastically reduced
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
                    f"Task: High-Fidelity Athlete Outfit View [View {variation_index + 1}]. PRODUCT SET: Combined {sport} outfit for {gender}. "
                    "You are provided with MULTIPLE reference images (different angles/lighting) to maximize replication accuracy. "
                    "STUDY ALL reference images closely to ensure every detail is captured. "
                    "Result must show BOTH Top and Bottom garments on one single model. "
                    
                    + (f"\n[ACTION & POSE]: {products[0].pose} " if products[0].pose != "N/A" else f"\n[DYNAMIC SPORT ACTION]: Pose: {selected_pose}. ") +
                    (f"\n[ENVIRONMENT]: {products[0].environment} " if products[0].environment != "N/A" else "") +

                    "\n[STRICT REQUIREMENT - MODEL IDENTITY]: "
                    f"Professional athletic {gender} model. Face, beard structure, and body must be IDENTICAL to the references. No cropping. "
                    
                    "\n[CRITICAL - ZERO-DRIFT GARMENT REPLICATION]: "
                    "Each garment must be a 100% pixel-perfect photographic clone of its respective reference set. "
                    "1. LOGO & TEXT LOCK: The 'SMMASH' logo must be output with 100% accuracy. Zero tolerance for character distortion or font shifts. "
                    "2. SURFACE PURITY: Do NOT add any extra logos, patterns, or graphics. If an area is blank in the reference, it MUST be blank in the output. "
                    "3. BRANDING LOCK: All existing branding must stay in the EXACT original positions. No relocation. "
                    "4. SEPARATION: Maintain clear distinction between Top and Bottom. Do not merge patterns. "
                    
                    "\n[CAMERA & FRAMING]: "
                    + ("Upper-body focus. " if is_close_up else "Full-body shot. ") + 
                    "Generate ONE SINGLE model. Head/face must be 100% visible with headroom. "
                    
                    f"\n[DYNAMIC SPORT ACTION]: Pose: {selected_pose}. "
                    
                    "\nQuality: Professional 8K DSLR photography, ultra-sharp focus. NO RE-DESIGNING."
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
                    "generationConfig": {"temperature": 0.1, "topP": 0.1, "topK": 10, "maxOutputTokens": 32768}
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
            logger.warning("No products provided for processing.")
            return []

        # 1. Group by code
        grouped = {}
        for p in products:
            if p.product_code not in grouped:
                grouped[p.product_code] = []
            grouped[p.product_code].append(p)
        
        unique_codes = list(grouped.keys())
        num_products = len(unique_codes)
        target_variations = 1 # Changed to 1: Use all uploaded reference images to generate one MASTERPIECE per product/outfit
        
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

    # ---------------------------------------------------------
    # Utility: Encode image safely (Added from drift branch)
    # ---------------------------------------------------------
    def encode_image(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing image: {image_path}")

        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        mime_type = "image/jpeg"
        if image_path.lower().endswith(".png"):
            mime_type = "image/png"

        return img_b64, mime_type

    # ---------------------------------------------------------
    # Dual Garment Try-On (Added from drift branch)
    # ---------------------------------------------------------
    async def generate_dual_tryon(
        self,
        upper_product: ProductMetadata,
        lower_product: ProductMetadata,
        variation_index: int = 0
    ) -> str:

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"🎯 Attempt {attempt+1} | Generating dual try-on: "
                    f"{upper_product.product_code} + {lower_product.product_code}"
                )

                # -------------------------------------------------
                # Encode both garments
                # -------------------------------------------------
                upper_path = os.path.join(settings.IMAGES_DIR, upper_product.image_filename)
                lower_path = os.path.join(settings.IMAGES_DIR, lower_product.image_filename)

                upper_b64, upper_mime = self.encode_image(upper_path)
                lower_b64, lower_mime = self.encode_image(lower_path)

                prompt = f"""
Task: Professional Athlete Catalog Photo [View {variation_index + 1}].

You are provided with MULTIPLE reference images for the garments to maximize accuracy. 
Use all angles provided to produce one singular perfect output.

{f'[ACTION & POSE]: {upper_product.pose}' if upper_product.pose != 'N/A' else ''}
{f'[ENVIRONMENT]: {upper_product.environment}' if upper_product.environment != 'N/A' else ''}

[STRICT REQUIREMENT - MODEL IDENTITY]:
Use the IDENTICAL {upper_product.gender} martial arts model from the reference. No changes to face, facial hair structure (beard), or hair style. Face must be fully visible.

[CRITICAL - ZERO-DRIFT GARMENT REPLICATION]:
The resulting image must be an EXACT photographic clone of the references.

1. LOGO & TEXT LOCK: Replicate the 'SMMASH' text and logo with 100% font/spacing accuracy. No hallucinations.
2. SURFACE PURITY: If a surface is blank in the reference, it MUST remain blank. Do NOT add any extra logos.
3. BRANDING LOCK: All existing branding must remain in the EXACT original position. 
4. SEPARATION: Top stays on top. Bottom stays on bottom. No pattern bleeding.

CRITICAL RULES:
- Model must wear BOTH garments simultaneously.
- No design simplification or changes.
- Professional DSLR quality, ultra-sharp focus.
"""

                payload = {
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt},

                                # Upper garment reference
                                {
                                    "inline_data": {
                                        "mime_type": upper_mime,
                                        "data": upper_b64
                                    }
                                },

                                # Lower garment reference
                                {
                                    "inline_data": {
                                        "mime_type": lower_mime,
                                        "data": lower_b64
                                    }
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.1, 
                        "topP": 0.1, 
                        "topK": 10,
                        "maxOutputTokens": 32768,
                    }
                }

                endpoint = f"{self.base_url}/models/{settings.GEMINI_MODEL_VERSION}:generateContent"

                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        endpoint,
                        headers=self.headers,
                        json=payload
                    )

                # -------------------------------------------------
                # Rate limit handling
                # -------------------------------------------------
                if response.status_code == 429:
                    logger.warning("⏳ Rate limited. Retrying...")
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue

                if response.status_code != 200:
                    logger.error(f"❌ API Error {response.status_code}: {response.text}")
                    raise ValueError("Gemini API failed.")

                result_data = response.json()

                # -------------------------------------------------
                # Log token usage
                # -------------------------------------------------
                usage = result_data.get("usageMetadata", {})
                logger.info(f"📊 Tokens used: {usage.get('totalTokenCount', 0)}")

                # -------------------------------------------------
                # Safe image extraction (FIXES YOUR ERROR)
                # -------------------------------------------------
                output_image_b64 = None

                candidates = result_data.get("candidates", [])
                for candidate in candidates:
                    parts = candidate.get("content", {}).get("parts", [])
                    for part in parts:
                        if "inline_data" in part:
                            output_image_b64 = part["inline_data"]["data"]
                            break
                        if "inlineData" in part:
                            output_image_b64 = part["inlineData"]["data"]
                            break
                        if "text" in part:
                            logger.warning(f"⚠ Model returned text: {part['text']}")
                    if output_image_b64:
                        break

                if not output_image_b64:
                    raise ValueError("No image data in response.")

                # -------------------------------------------------
                # Save output
                # -------------------------------------------------
                output_filename = (
                    f"dual_{upper_product.product_code}_"
                    f"{lower_product.product_code}_"
                    f"var{variation_index}_"
                    f"{int(time.time())}.png"
                )

                output_path = os.path.join(settings.EXPORT_DIR, output_filename)

                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(output_image_b64))

                logger.success(f"✅ Generated: {output_filename}")
                return output_path

            except Exception as e:
                logger.error(f"❌ Attempt {attempt+1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(retry_delay * (attempt + 1))

    # ---------------------------------------------------------
    # Batch Example (Pairs of Products) (Added from drift branch)
    # ---------------------------------------------------------
    async def batch_dual_process(self, product_pairs: list[tuple[ProductMetadata, ProductMetadata]]):

        results = []

        for i, (upper, lower) in enumerate(product_pairs):
            try:
                result = await self.generate_dual_tryon(
                    upper_product=upper,
                    lower_product=lower,
                    variation_index=i
                )
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Failed pair {i+1}: {e}")

        logger.info(f"🏁 Batch complete | Generated {len(results)} images.")
        return results
