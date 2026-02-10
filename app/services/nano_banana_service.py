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

    async def generate_tryon_image(self, product: ProductMetadata, image_path: str, variation_index: int = 0) -> str:
        """
        Calls the Nano Banana Pro (Gemini) API for virtual try-on with strict consistency and brand integrity.
        """
        max_retries = 3
        retry_delay = 2 

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}: Generating image for {product.product_code}")
                
                if not os.path.exists(image_path):
                    logger.error(f"Image not found: {image_path}")
                    raise FileNotFoundError(f"Missing image: {image_path}")

                from app.utils.image_processor import ImageProcessor
                
                # Support up to 4K resolution
                final_image_path = ImageProcessor.optimize_for_api(image_path, max_size=(4096, 4096))
                
                with open(final_image_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")

                # Detect mime type
                mime_type = "image/jpeg"
                if final_image_path.lower().endswith(".png"):
                    mime_type = "image/png"

                # ABSOLUTE CONSISTENCY PROMPT: STRICT LOCK on Model, Scene, and Garment Fidelity.
                prompt = (
                    f"Task: High-Fidelity Athlete Catalog [15-Image Series]. PRODUCT: {product.product_name} for {product.sport}. "
                    
                    "\n[STRICT LOCK - MODEL IDENTITY]: "
                    f"You MUST use the EXACT same {product.gender} model from Variation 1 in this image. "
                    "The face, facial features, hair, skin tone, and body build must be identical. "
                    
                    "\n[STRICT LOCK - ENVIRONMENT]: "
                    f"The background (the {product.sport} location) must be 100% identical to all other images in this series. "
                    "Fixed lighting, fixed decor. Only the model's action and camera angle vary. "
                    
                    "\n[CAMERA & FRAMING]: "
                    "Wide or medium shot. The model's FULL HEAD, HAIR, AND FACE must be completely visible with clear space above the head (headroom). "
                    "CRITICAL: NEVER crop or cutoff the model's head, face, or hair. The model must be perfectly centered within the frame. "

                    f"\n[DYNAMIC SPORT ACTION]: This image ({variation_index + 1}/15) MUST show the model ACTUALLY PERFORMING {product.sport}. "
                    "Show dynamic, authentic athletic movement (e.g., mid-action strike, sprint, or pose). The entire action and the model's full upper body must fit comfortably within the frame. "

                    "\n[ZERO-DRIFT GARMENT FIDELITY - CRITICAL]: "
                    "The garment must be a 100% EXACT CLONE of the source image. "
                    "1. COLOR PERFECTION: Preserve the exact color, shade, and vibrancy. DO NOT change the hue or saturation. "
                    "2. STYLE & CUT: Necklines, sleeves, seams, and fit must be 100% identical. NO changes to the garment's design. "
                    "3. BRANDING: Logos and text must be 100% IDENTICAL in size, position, and font. No 'hallucinated' text. "
                    "4. TEXTURE: Maintain exact fabric weave and material properties (e.g., compression fit, dry-fit texture). "

                    "\nQuality: Professional 8K photography, DSLR sharp focus, ultra-detailed fabric textures."
                )

                payload = {
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt},
                                {
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": img_data
                                    }
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.2,  # Lowered for maximum fidelity to input
                        "topP": 0.99,
                        "topK": 100,
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
                        parts = content.get("parts", [])
                        
                        for part in parts:
                            if "inline_data" in part:
                                output_image_b64 = part["inline_data"]["data"]
                                break
                            elif "inlineData" in part:
                                output_image_b64 = part["inlineData"]["data"]
                                break
                        
                        if not output_image_b64:
                            raise ValueError("No image data in response.")
                            
                    except Exception as e:
                        logger.error(f"❌ Extraction failed: {str(e)}")
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

    async def batch_process(self, products: list[ProductMetadata]):
        """
        Takes uploaded products and generates exactly 15 high-quality variations.
        """
        num_uploads = len(products)
        target_total = 15
        
        if num_uploads == 0:
            logger.warning("No products provided for processing.")
            return []

        logger.info(f"🚀 Starting Batch: {num_uploads} uploads -> 15 variations")
        
        results = []
        for i in range(target_total):
            product = products[i % num_uploads]
            
            if not product.image_filename:
                continue

            image_path = os.path.join(settings.IMAGES_DIR, product.image_filename)

            logger.info(f"🔄 Generation {i+1}/{target_total} | {product.product_code} ({product.sport})")

            try:
                result_path = await self.generate_tryon_image(product, image_path, variation_index=i)
                results.append(result_path)
            except Exception as e:
                logger.error(f"❌ Failed generation {i+1}: {e}")

        logger.info(f"🏁 Batch finished | Generated {len(results)}/{target_total} images.")
        return results
