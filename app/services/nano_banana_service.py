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
        
        # Ensure export directory exists
        if not os.path.exists(settings.EXPORT_DIR):
            try:
                os.makedirs(settings.EXPORT_DIR, exist_ok=True)
                logger.info(f"📁 Created export directory: {settings.EXPORT_DIR}")
            except Exception as e:
                logger.error(f"❌ Failed to create export directory: {e}")

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

        # Build pose and environment purely from metadata
        if product.pose != "N/A":
            pose_text = f"[POSE]: {product.pose}. The model must be ALONE — no other person in the image."
        else:
            pose_text = f"[POSE]: The model is a {product.sport} athlete in a mid-action pose that clearly shows they practice {product.sport}. Not a stiff formal pose — the model should look athletic and in motion, like a real {product.sport} athlete during light training or warm-up. Keep it natural and moderate — not too extreme, not too static. The model must be ALONE — no other person, no opponent, no physical contact."

        if product.environment != "N/A":
            env_text = f"[ENVIRONMENT]: {product.environment}."
        else:
            env_text = f"[ENVIRONMENT]: Choose a professional environment that naturally fits {product.sport}. Clean, well-lit, realistic."

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}: Generating image for {product.product_code}")
                
                parts = []
                prompt = (
                    # ── GARMENT FIRST — THIS IS THE #1 PRIORITY ──
                    "[ZERO-DRIFT PRODUCT REPLICATION — THIS IS THE MOST IMPORTANT INSTRUCTION]: "
                    "You are given a reference photo of a real product. "
                    "Your output image MUST contain this EXACT SAME product — ZERO changes allowed. "
                    "The product in the output must be a pixel-perfect copy of the reference: "
                    "same design, same colors, same patterns, same graphics, same logos, same text, same everything. "
                    "If the product has a logo or brand name text, it must appear EXACTLY as in the reference — "
                    "same letters, same font, same size, same position, same color. "
                    "\nCRITICAL TEXT RULES: "
                    "0. LOGO AND THE TEXT SHOULD BE SAME ON THE GENERATED IAMGE AS IN THE REFERENCE IMAGE. "
                    "1. DO NOT 'read' the text. Treat any text on the garment as abstract geometric SHAPES and SYMBOLS. "
                    "2. Trace these shapes pixel-for-pixel. Do not try to spell-check, correct, or re-type them. "
                    "3. If the text looks like 'SMMASH' or any other brand, copy the exact curves and lines of the letters. "
                    "4. DO NOT redraw, retype, reinterpret, resize, relocate, blur, distort, or modify ANY element on the product. "
                    "5. Even if the text is small or distant, it must remain SHARP and IDENTICAL to the reference. "
                    "If an area is plain/blank, keep it plain/blank. "
                    "The product must look like someone took the EXACT item from the reference photo, put it on a model, and photographed it. "
                    "ZERO DRIFT from the reference. "

                    f"\n\n[SCENE SETUP]: "
                    f"Professional catalog photo for {product.sport}. {product.gender} model. "
                    f"\n{pose_text}"
                    f"\n{env_text}"

                    f"\n\n[MODEL]: "
                    f"ONE single {product.gender} athletic model, ALONE. "
                    "Full head, face, and hair completely visible — NEVER crop the top of the head. "
                    "Leave generous headroom above the head. "
                    "No other people. No fighting. No grappling. "

                    "\n\n[BODY & MODESTY]: "
                    "Clothing fits naturally, relaxed look. "
                    "NO visible outline of private body parts. "
                    "Garment drapes naturally. Professional catalog standard. "

                    "\n\n[ANATOMY]: "
                    "Correct human anatomy — 2 arms, 2 legs, 5 fingers per hand. No extra or distorted limbs. "

                    "\n\n[FRAMING]: "
                    "Full-body shot, head to toe. Entire head in frame with space above. "
                    "ONE model. No collages. No grids. "

                    "\n\n[QUALITY]: Ultra-high resolution, 8K DSLR photography, razor-sharp focus on product details and logos."
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
                        "temperature": 0.0,
                        "topP": 0.1,
                        "topK": 5,
                        "maxOutputTokens": 32768,
                        "responseModalities": ["TEXT", "IMAGE"],
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
                    
                    # ── Logo Refinement Pass ──
                    # Send generated image + reference back to fix only the logo/text
                    try:
                        refined_path = await self._refine_logo(output_path, image_paths)
                        if refined_path:
                            return refined_path
                    except Exception as logo_err:
                        logger.warning(f"⚠️ Logo refinement failed, using original: {logo_err}")
                    
                    return output_path

            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(retry_delay * (attempt + 1))

    async def _refine_logo(self, generated_image_path: str, reference_image_paths: list[str]) -> str | None:
        """
        Second-pass: sends the generated image + original references to Gemini
        with a focused prompt to fix ONLY the logo/text area.
        Returns the refined image path, or None if refinement fails.
        """
        logger.info("🔧 Starting logo refinement pass...")
        
        from app.utils.image_processor import ImageProcessor
        
        parts = []
        
        # Focused edit prompt — ONLY fix the logo
        # Focused edit prompt — ONLY fix the logo
        parts.append({"text": (
            "TASK: PIXEL-PERFECT REPAIR of Garment Graphics & Text. "
            "\nYou are given: "
            "1) REFERENCE PRODUCT PHOTOS (first images) — The SOURCE OF TRUTH. Contains the CORRECT logo, text, graphics. "
            "2) A GENERATED PHOTO (last image) — Needs repair. The text/logos on the garment may be distorted/hallucinated. "
            "\nYour job: "
            "Take the GENERATED PHOTO and overwrite the garment's graphics/text with a PIXEL-PERFECT CLONE from the REFERENCE PHOTOS. "
            "\nCRITICAL RULES FOR TEXT: "
            "1. DO NOT READ THE TEXT. Treat it as a pattern of shapes or foreign symbols. "
            "2. DO NOT SPELL-CHECK or RE-TYPE. If the reference says 'SMMASH', trace those exact shapes. "
            "3. ERASE any hallucinated or distorted text on the generated image and REPLACE it with the exact shapes from the reference. "
            "4. The final result must look like a PHOTOCOPY of the reference design applied to the model's clothing. "
            "5. NO BLUR. NO DISTORTION. Sharp edges only. "
            "\nGENERAL RULES: "
            "- DO NOT touch the model's face, skin, hair, pose, or background. "
            "- ONLY edit the pixels on the fabric surface to match the reference design. "
            "- Output the full, high-quality image with the corrected garment. "
            "\nORDER: Reference Images (FIRST) -> Generated Image (LAST)."
        )})
        
        # Add reference images first
        for ref_path in reference_image_paths:
            if not os.path.exists(ref_path):
                continue
            opt_path = ImageProcessor.optimize_for_api(ref_path, max_size=(4096, 4096))
            with open(opt_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            mime_type = "image/png" if opt_path.lower().endswith(".png") else "image/jpeg"
            parts.append({"inline_data": {"mime_type": mime_type, "data": img_data}})
        
        # Add the generated image last
        with open(generated_image_path, "rb") as f:
            gen_data = base64.b64encode(f.read()).decode("utf-8")
        gen_mime = "image/png" if generated_image_path.lower().endswith(".png") else "image/jpeg"
        parts.append({"inline_data": {"mime_type": gen_mime, "data": gen_data}})
        
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.0,
                "topP": 0.05,
                "topK": 5,
                "maxOutputTokens": 32768,
                "responseModalities": ["TEXT", "IMAGE"],
            }
        }
        
        endpoint = f"{self.base_url}/models/{settings.GEMINI_MODEL_VERSION}:generateContent"
        
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(endpoint, headers=self.headers, json=payload)
            
            if response.status_code != 200:
                logger.warning(f"Logo refinement API error ({response.status_code})")
                return None
            
            result_data = response.json()
            
            # Extract refined image
            refined_b64 = None
            try:
                candidate = result_data["candidates"][0]
                content = candidate.get("content", {})
                for part in content.get("parts", []):
                    if "inline_data" in part or "inlineData" in part:
                        refined_b64 = part.get("inline_data", part.get("inlineData"))["data"]
                        break
            except Exception:
                pass
            
            if not refined_b64:
                logger.warning("Logo refinement returned no image")
                return None
            
            # Save refined image (overwrite the original)
            refined_path = generated_image_path.replace(".png", "_refined.png")
            with open(refined_path, "wb") as f:
                f.write(base64.b64decode(refined_b64))
            
            logger.success(f"✅ Logo refined: {refined_path}")
            return refined_path

    async def generate_outfit_image(self, products: list[ProductMetadata], variation_index: int = 0) -> str:
        """
        Calls Gemini with multiple reference images to generate a single cohesive outfit in a specific pose.
        Uses Zero-Drift logic.
        """
        max_retries = 3
        retry_delay = 2

        # Group products to describe them in the prompt - simplified logic
        sport = products[0].sport
        gender = products[0].gender

        # Build pose and environment from metadata
        if products[0].pose != "N/A":
            pose_text = f"[POSE]: {products[0].pose}. The model must be ALONE — no other person."
        else:
            pose_text = f"[POSE]: The model is a {sport} athlete in a mid-action pose that clearly shows they practice {sport}. Not a stiff formal pose — the model should look athletic and in motion, like a real {sport} athlete during light training or warm-up. Keep it natural and moderate — not too extreme, not too static. The model must be ALONE."

        if products[0].environment != "N/A":
            env_text = f"[ENVIRONMENT]: {products[0].environment}."
        else:
            env_text = f"[ENVIRONMENT]: Choose a professional environment that naturally fits {sport}. Clean, well-lit, realistic."

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}: Generating combined outfit image (Var {variation_index + 1})")
                
                parts = []
                prompt = (
                    f"[ZERO-DRIFT PRODUCT REPLICATION — THIS IS THE MOST IMPORTANT INSTRUCTION]: "
                    "You are given reference photos of TWO real products (upper and lower garments). "
                    "Your output image MUST contain these EXACT SAME products — ZERO changes allowed. "
                    "Each product must be a pixel-perfect copy of its reference: "
                    "same design, same colors, same patterns, same graphics, same logos, same text, same everything. "
                    "If a product has a logo or brand name text, it must appear EXACTLY as in the reference — "
                    "same letters, same font, same size, same position, same color. "
                    "\nCRITICAL TEXT RULES: "
                    "0. LOGO AND THE TEXT SHOULD BE SAME ON THE GENERATED IAMGE AS IN THE REFERENCE IMAGE. "
                    "1. DO NOT 'read' the text. Treat any text on the garment as abstract geometric SHAPES and SYMBOLS. "
                    "2. Trace these shapes pixel-for-pixel. Do not try to spell-check, correct, or re-type them. "
                    "3. If the text looks like 'SMMASH' or any other brand, copy the exact curves and lines of the letters. "
                    "4. DO NOT redraw, retype, reinterpret, resize, relocate, blur, distort, or modify ANY element on the product. "
                    "5. Even if the text is small or distant, it must remain SHARP and IDENTICAL to the reference. "
                    "If an area is plain/blank, keep it plain/blank. "
                    "Upper garment stays on top. Lower garment stays on bottom. No pattern bleeding. "
                    "ZERO DRIFT from the references. "
                    
                    f"\n\n[SCENE SETUP]: "
                    f"Professional catalog photo — complete {sport} outfit for {gender}. "
                    "ONE SINGLE model wearing BOTH garments as a complete outfit. "
                    "ABSOLUTELY NO SECOND PERSON. NO BACKGROUND MODELS. NO CROWD. JUST ONE PERSON. "
                    f"\n{pose_text}"
                    f"\n{env_text}"

                    f"\n\n[MODEL]: "
                    f"ONE single {gender} athletic model, ALONE. "
                    "The model is the SOLE subject. No training partners. No opponents. "
                    "Full head, face, and hair completely visible — NEVER crop the top of the head. "
                    "Leave generous headroom. "

                    "\n\n[BODY & MODESTY]: "
                    "Clothing fits naturally. NO visible outline of private body parts. "
                    "Professional catalog standard. "

                    "\n\n[ANATOMY]: "
                    "Correct anatomy — 2 arms, 2 legs, 5 fingers per hand. No extra limbs. "

                    "\n\n[FRAMING]: "
                    "Full-body shot, head to toe. ONE model. No collages. No grids. "

                    "\n\n[QUALITY]: Ultra-high resolution, 8K DSLR, razor-sharp focus on product details and logos."
                )
                parts.append({"text": prompt})

                from app.utils.image_processor import ImageProcessor
                for p in products:
                    image_path = os.path.join(settings.IMAGES_DIR, p.image_filename)
                    if os.path.exists(image_path):
                        opt_path = ImageProcessor.optimize_for_api(image_path, max_size=(4096, 4096))
                        with open(opt_path, "rb") as f:
                            img_data = base64.b64encode(f.read()).decode("utf-8")
                        mime_type = "image/png" if opt_path.lower().endswith(".png") else "image/jpeg"
                        parts.append({"inline_data": {"mime_type": mime_type, "data": img_data}})

                payload = {
                    "contents": [{"parts": parts}],
                    "generationConfig": {
                        "temperature": 0.0,
                        "topP": 0.1,
                        "topK": 5,
                        "maxOutputTokens": 32768,
                        "responseModalities": ["TEXT", "IMAGE"],
                    }
                }

                endpoint = f"{self.base_url}/models/{settings.GEMINI_MODEL_VERSION}:generateContent"
                async with httpx.AsyncClient(timeout=180.0) as client:
                    response = await client.post(endpoint, headers=self.headers, json=payload)
                    if response.status_code == 429:
                        logger.warning("Rate limit hit. Waiting...")
                        await asyncio.sleep(5)
                        continue
                    
                    if response.status_code != 200: 
                        raise ValueError(f"API Error ({response.status_code}): {response.text[:200]}")

                    result_data = response.json()
                    
                    # Extract image
                    output_image_b64 = None
                    try:
                        candidates = result_data.get("candidates", [])
                        if candidates:
                            for part in candidates[0].get("content", {}).get("parts", []):
                                if "inline_data" in part or "inlineData" in part:
                                    output_image_b64 = part.get("inline_data", part.get("inlineData"))["data"]
                                    break
                    except:
                        pass
                    
                    if not output_image_b64:
                         raise ValueError("No image data in response.")

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
        target_variations = 1  # Generating ONLY 1 high-accuracy image as per user request
        
        results = []

        # Check if all products belong to the EXACT same sport/category
        unique_sports = set(p.sport for p in products)
        is_consistent_outfit = (len(unique_sports) == 1)

        # Case 1: Multiple products for the SAME sport -> Combined Outfit
        if num_products >= 2 and is_consistent_outfit:
            sport_name = list(unique_sports)[0]
            logger.info(f"🎯 Multi-product outfit detected for {sport_name}. Generating {target_variations} unique combined poses...")
            
            for v_idx in range(target_variations):
                try:
                    result_path = await self.generate_outfit_image(products, variation_index=v_idx)
                    results.append(result_path)
                    # Small throttle between variations
                    await asyncio.sleep(2)
                except Exception as e:
                    logger.error(f"Outfit pose {v_idx+1} failed: {e}")
            return results
        
        # Case 2: Different sports OR individual products -> Process separately
        if num_products >= 2:
            logger.info(f"🔀 Mixed sports detected ({unique_sports}). Processing {num_products} products individually.")
        else:
            logger.info(f"🚀 Processing single product. Target: {target_variations} poses.")

        for code, group in grouped.items():
            # Collect all images for this product code
            image_paths = [os.path.join(settings.IMAGES_DIR, p.image_filename) for p in group]
            # Use the metadata from the first item in the group (they share the code)
            product_meta = group[0]
            
            for v_idx in range(target_variations):
                logger.info(f"🔄 Generating {code} ({product_meta.sport}) | Pose Variation {v_idx+1}/{target_variations}")
                try:
                    # Pass the specific product metadata which includes the correct sport/pose/env
                    result_path = await self.generate_tryon_image(product_meta, image_paths, variation_index=v_idx)
                    results.append(result_path)
                    await asyncio.sleep(2)
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
