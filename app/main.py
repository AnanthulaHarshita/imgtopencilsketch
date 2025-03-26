import os
import time
import logging
from tkinter import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import torch
from fastapi import Body
from app.middleware import CustomHeaderMiddleware, RequestLoggingMiddleware
from app.config import UPLOAD_DIR, RESULTS_DIR, FRONTEND_DIR, CHECKPOINT_PATH
from models.gan_model import load_trained_model
from utils.file_operations import save_uploaded_file, cleanup_old_files,compress_image, resized_img
from utils.image_processing import generate_sketch_from_image, save_sketch_to_file
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Initialize FastAPI app
app = FastAPI()

# Add CORS Middleware here
origins = [
    "http://localhost:3000",  # Frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)
# Add middleware to the app
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(CustomHeaderMiddleware)

# Initialize model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_trained_model(CHECKPOINT_PATH)

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
os.makedirs(FRONTEND_DIR, exist_ok=True)

# Mount static directories for serving uploaded files
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
# Make sure to serve files from a directory (in this case 'uploads')
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Clean up old files on startup
cleanup_old_files(UPLOAD_DIR)
cleanup_old_files(RESULTS_DIR)

@app.get("/frontend/login.html")
async def read_login_html():
    return FileResponse("frontend/login.html")

@app.post("/upload_GAN/")
async def upload_file_GAN(file: UploadFile = File(...)):
    """Handles image uploads and saves the file to 'uploads/'."""
    logger.info("Received file upload request")
    if not file.filename.lower().endswith(("png", "jpg", "jpeg")):
        logger.error("Invalid file type")
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        # Save file to the uploads directory
        file_path = save_uploaded_file(file, file.filename)
        logger.info(f"File uploaded successfully to {file_path}")
        
        # Return the correct URL
        image_url = f"/uploads/{file.filename}"
        return JSONResponse(content={"message": "File uploaded successfully", "filename": file.filename, "image_url": image_url})
    except Exception as e:
        logger.error(f"Failed to save file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

""" Function to handle image uploading to the UPLOAD_DIR"""
@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    # Save the uploaded file
    file_path = save_uploaded_file(file, file.filename)

    # Open the image
    img = Image.open(file_path)

    # Resize the image
    resized_img = resized_img(img, (10,10))

    # Compress the image
    compressed_img = compress_image(resized_img, quality=90)

    # Save the resized and compressed image
    compressed_img.save('resized_and_compressed_image.jpg')

    return {"message": "Image uploaded and processed successfully"}


@app.post("/generate-sketch_GAN/")
async def generate_sketch_with_model_GAN(filename: str = Body(...)):
    logger.info(f"Received request to generate sketch for {filename}")
    
    input_path = os.path.join(UPLOAD_DIR, filename)
    logger.info(f"Checking if file exists at {input_path}")

    if not os.path.exists(input_path):
        logger.error(f"File not found: {input_path}")
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        logger.info(f"Generating sketch from {filename}")
        generated_sketch = generate_sketch_from_image(input_path, model, DEVICE)
        logger.info(f"Sketch generated successfully")
        
        output_path = save_sketch_to_file(generated_sketch, filename)
        filename_only = os.path.basename(output_path)
        logger.info(f"Saving generated sketch to {output_path}")
        
        return JSONResponse(content={"message": "Sketch generated successfully", "processed_filename": filename_only})
    except Exception as e:
        logger.error(f"Failed to generate sketch for {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate sketch: {str(e)}")

@app.get("/result_GAN/{filename}")
async def get_result_GAN(filename: str):
    """Fetches the processed sketch image from the results directory."""
    logger.info("Received result request")
    result_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(result_path):
        logger.error("Processed image not found")
        raise HTTPException(status_code=404, detail="Processed image not found")
    return FileResponse(result_path)

@app.get("/")
async def read_root():
    logger.info("Received root request")
    return {"message": "Hello, world!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
