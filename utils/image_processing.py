# utils/image_processing.py
import os
import cv2
import torch
from app.config import RESULTS_DIR
from models.gan_model import preprocess_image, postprocess_image

import logging

# Set up logging
logger = logging.getLogger(__name__)

def generate_sketch_from_image(input_path, model, device):
    """Generates sketch using the trained model."""
    logger.info(f"Started generating sketch from image: {input_path}")

    try:
        # Log preprocessing step
        logger.info(f"Preprocessing image: {input_path}")
        img_tensor = preprocess_image(input_path, device)
        logger.info(f"Image preprocessed successfully")

        # Log model inference step
        logger.info("Generating sketch using the trained model...")
        with torch.no_grad():
            generated_sketch = model(img_tensor)
        logger.info("Sketch generated successfully")

        # Log postprocessing step
        logger.info("Postprocessing the generated sketch")
        sketch = postprocess_image(generated_sketch)
        logger.info("Postprocessing completed")

        return sketch

    except Exception as e:
        logger.error(f"Error during sketch generation from image {input_path}: {e}")
        raise  # Re-raise the exception so it can be caught in the calling code

def save_sketch_to_file(sketch, filename):
    """Saves the generated sketch to a file."""
    output_path = os.path.join(RESULTS_DIR, "gan_sketch_" + filename)
    cv2.imwrite(output_path, sketch)
    return output_path

from PIL import Image
