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

    async def generate_tryon_image(self, product: ProductMetadata, image_path: str) -> str:
        """
        Calls the Nano Banana Pro (Gemini) API for virtual try-on.
        """
        max_retries = 3
        retry_delay = 2 

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}: Starting Nano Banana Pro generation for {product.product_code}")
                
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

                # Enhanced 4K High-Resolution Prompt
                prompt = (
                    f"Task: Virtual Try-On. Product: {product.product_name}. Type: {product.product_type}. "
                    "Instruction: Generate a professional model wearing this garment in 4K resolution. "
                    "QUALITY: Professional studio photography, high fidelity, 4k ultra-detailed, photorealistic, 8k textures. "
                    "ZERO-DRIFT REQUIREMENT: Keep the garment EXACTLY as provided. "
                    "Maintain every single seam, stitch, fold, texture, and logo perfectly."
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
                        "temperature": 0.5,
                        "topP": 0.99,
                        "topK": 100,
                        "maxOutputTokens": 4096,  # 4K detail tokens
                    }
                }

                # Construct the Official Endpoint
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
                    
                    # Robust extraction: loop parts to find image data
                    output_image_b64 = None
                    try:
                        candidate = result_data["candidates"][0]
                        
                        # Check for safety or other finish reasons
                        if candidate.get("finishReason") != "STOP":
                            logger.warning(f"Warning: Finish reason is {candidate.get('finishReason')}")

                        content = candidate.get("content", {})
                        parts = content.get("parts", [])
                        
                        logger.info(f"🔍 AI Response has {len(parts)} parts.")
                        
                        for i, part in enumerate(parts):
                            keys = list(part.keys())
                            logger.info(f"  Part {i}: Keys {keys}")
                            
                            # Check for both snake_case and camelCase keys
                            if "inline_data" in part:
                                output_image_b64 = part["inline_data"]["data"]
                                logger.success(f"🎨 Found image in 'inline_data' (Part {i})")
                                break
                            elif "inlineData" in part:
                                output_image_b64 = part["inlineData"]["data"]
                                logger.success(f"🎨 Found image in 'inlineData' (Part {i})")
                                break
                            elif "image" in part:
                                output_image_b64 = part["image"]["data"]
                                logger.success(f"🎨 Found image in 'image' (Part {i})")
                                break
                            elif "text" in part:
                                logger.info(f"  Part {i} contains text: {part['text'][:100]}...")
                        
                        if not output_image_b64:
                            raise ValueError(f"No image data found in any of the {len(parts)} parts.")
                            
                    except (KeyError, IndexError, Exception) as e:
                        logger.error(f"❌ Extraction failed: {str(e)}")
                        raise e

                    output_filename = f"result_{product.product_code}_{int(time.time())}.png"
                    output_path = os.path.join(settings.EXPORT_DIR, output_filename)
                    
                    with open(output_path, "wb") as f:
                        f.write(base64.b64decode(output_image_b64))

                    img_size = os.path.getsize(output_path) / 1024
                    logger.success(f"Generated successfully: {output_filename} ({img_size:.2f} KB)")
                    return output_path

            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(retry_delay * (attempt + 1))

    async def batch_process(self, products: list[ProductMetadata]):
        total = len(products)
        logger.info(f"🚀 Starting NanoBanana batch | {total} items")

        results = []

        for idx, product in enumerate(products, start=1):
            if not product.image_filename:
                logger.warning(
                    f"⚠️ Skipping {idx}/{total} | No image for {product.product_code}"
                )
                continue

            image_path = os.path.join(
                settings.IMAGES_DIR, product.image_filename
            )

            logger.info(
                f"🔄 Processing {idx}/{total} | {product.product_code}"
            )

            try:
                start = time.time()
                result_path = await self.generate_tryon_image(
                    product, image_path
                )

                product.status = "completed"
                product.result_url = result_path
                results.append(result_path)

                logger.success(
                    f"✅ Completed {product.product_code} in {time.time() - start:.2f}s"
                )

            except Exception as e:
                product.status = "failed"
                logger.error(
                    f"❌ Failed {product.product_code}: {e}"
                )

        logger.info(
            f"🏁 Batch finished | Success {len(results)}/{total}"
        )
        return results
