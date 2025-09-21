# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from typing import List
# import uvicorn
# import numpy as np
# from PIL import Image
# import io
# import tensorflow as tf
# import pandas as pd
# import os
# from datetime import datetime
# import json

# # Initialize FastAPI app
# app = FastAPI(title="Cotton Leaf Disease Detection API")

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Mount static files directory
# os.makedirs("static/uploads", exist_ok=True)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Load the trained model (update path as needed)
# MODEL_PATH = "app/models/cotton_disease_model.h5"
# model = tf.keras.models.load_model(MODEL_PATH)

# # Class names (update based on your dataset)
# CLASS_NAMES = [
#     "Healthy", 
#     "Bacterial Blight", 
#     "Curly Virus", 
#     "Fusarium Wilt", 
#     "Powdery Mildew",
#     "Target Spot"
# ]

# # CSV file path for storing predictions
# CSV_FILE_PATH = "app/data/predictions.csv"

# # Ensure data directory exists
# os.makedirs("app/data", exist_ok=True)

# # Initialize CSV file if it doesn't exist
# if not os.path.exists(CSV_FILE_PATH):
#     df = pd.DataFrame(columns=["timestamp", "filename", "prediction", "confidence", "image_path"])
#     df.to_csv(CSV_FILE_PATH, index=False)

# @app.get("/")
# async def root():
#     return {"message": "Cotton Leaf Disease Detection API"}

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Read and preprocess the image
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
        
#         # Resize image to match model's expected sizing
#         image = image.resize((224, 224))
        
#         # Convert to array and normalize
#         img_array = np.array(image) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)
        
#         # Make prediction
#         predictions = model.predict(img_array)
#         predicted_class = np.argmax(predictions[0])
#         confidence = float(np.max(predictions[0]))
        
#         # Save the uploaded file
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"{timestamp}_{file.filename}"
#         file_path = f"static/uploads/{filename}"
        
#         with open(file_path, "wb") as f:
#             f.write(contents)
        
#         # Save prediction to CSV
#         new_prediction = {
#             "timestamp": datetime.now().isoformat(),
#             "filename": filename,
#             "prediction": CLASS_NAMES[predicted_class],
#             "confidence": confidence,
#             "image_path": file_path
#         }
        
#         df = pd.read_csv(CSV_FILE_PATH)
#         df = pd.concat([df, pd.DataFrame([new_prediction])], ignore_index=True)
#         df.to_csv(CSV_FILE_PATH, index=False)
        
#         return {
#             "class": CLASS_NAMES[predicted_class],
#             "confidence": confidence,
#             "filename": filename
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/predictions")
# async def get_predictions(limit: int = 10):
#     try:
#         df = pd.read_csv(CSV_FILE_PATH)
#         # Convert to list of dictionaries
#         predictions = df.tail(limit).to_dict(orient="records")
#         return predictions
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/stats")
# async def get_stats():
#     try:
#         df = pd.read_csv(CSV_FILE_PATH)
        
#         if df.empty:
#             return {
#                 "total_predictions": 0,
#                 "disease_distribution": {},
#                 "recent_accuracy": 0
#             }
        
#         total_predictions = len(df)
#         disease_distribution = df["prediction"].value_counts().to_dict()
        
#         # Calculate recent accuracy (assuming confidence > 0.7 is accurate)
#         recent_predictions = df.tail(10)
#         recent_accuracy = recent_predictions[recent_predictions["confidence"] > 0.7].shape[0] / len(recent_predictions) if len(recent_predictions) > 0 else 0
        
#         return {
#             "total_predictions": total_predictions,
#             "disease_distribution": disease_distribution,
#             "recent_accuracy": recent_accuracy
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

# Import routers
from app.routes.predictions import router as predictions_router

# Initialize FastAPI app
app = FastAPI(
    title="Cotton Leaf Disease Detection API",
    description="API for detecting diseases in cotton leaves using deep learning",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("app/data", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(predictions_router)

@app.get("/")
async def root():
    return {
        "message": "Cotton Leaf Disease Detection API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "predict": "/api/v1/predict",
            "predictions": "/api/v1/predictions",
            "stats": "/api/v1/stats"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running successfully"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )