from PIL import Image
import os
from loguru import logger

class ImageProcessor:
    @staticmethod
    def validate_image(image_path: str) -> bool:
        """Checks if the file is a valid image and within size limits."""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception as e:
            logger.error(f"Image validation failed for {image_path}: {e}")
            return False

    @staticmethod
    def optimize_for_api(image_path: str, max_size=(4096, 4096)) -> str:
        """Resizes image if too large while maintaining aspect ratio, preserving quality."""
        try:
            with Image.open(image_path) as img:
                if img.width > max_size[0] or img.height > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    optimized_path = image_path.replace(".", "_optimized.")
                    img.save(optimized_path, quality=95)
                    logger.info(f"Image optimized: {optimized_path}")
                    return optimized_path
            return image_path
        except Exception as e:
            logger.warning(f"Image optimization failed, using original: {e}")
            return image_path
