# utils/file_operations.py
import os
import shutil
from app.config import UPLOAD_DIR, RESULTS_DIR
import time

def save_uploaded_file(file, filename):
    """Saves uploaded file to the uploads directory."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path

def cleanup_old_files(directory, max_age_seconds=86400):
    """Deletes files older than `max_age_seconds` in the given directory."""
    current_time = time.time()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                os.remove(file_path)

from PIL import Image


def compress_image(img, quality):
    img.save('compressed_image.jpg', optimize=True, quality=quality)
    return Image.open('compressed_image.jpg')

def resized_img(img, size):
    return img.resize(size)