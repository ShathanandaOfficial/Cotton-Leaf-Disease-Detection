from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import pandas as pd
import os
from datetime import datetime

router = APIRouter(prefix="/api/v1", tags=["predictions"])

# Load model (update path as needed)
model = tf.keras.models.load_model("app/models/cotton_disease_model.h5")
CLASS_NAMES = ["Healthy", "Bacterial Blight", "Curly Virus", "Fusarium Wilt", "Powdery Mildew", "Target Spot"]
CSV_FILE_PATH = "app/data/predictions.csv"

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))
        
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Save file and prediction
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = f"static/uploads/{filename}"
        
        os.makedirs("static/uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Save to CSV
        new_prediction = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "prediction": CLASS_NAMES[predicted_class],
            "confidence": confidence,
            "image_path": file_path
        }
        
        df = pd.read_csv(CSV_FILE_PATH) if os.path.exists(CSV_FILE_PATH) else pd.DataFrame(columns=["timestamp", "filename", "prediction", "confidence", "image_path"])
        df = pd.concat([df, pd.DataFrame([new_prediction])], ignore_index=True)
        df.to_csv(CSV_FILE_PATH, index=False)
        
        return {
            "class": CLASS_NAMES[predicted_class],
            "confidence": confidence,
            "filename": filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions")
async def get_predictions(limit: int = 10):
    try:
        if not os.path.exists(CSV_FILE_PATH):
            return []
            
        df = pd.read_csv(CSV_FILE_PATH)
        predictions = df.tail(limit).to_dict(orient="records")
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_stats():
    try:
        if not os.path.exists(CSV_FILE_PATH):
            return {
                "total_predictions": 0,
                "disease_distribution": {},
                "recent_accuracy": 0
            }
            
        df = pd.read_csv(CSV_FILE_PATH)
        total_predictions = len(df)
        disease_distribution = df["prediction"].value_counts().to_dict()
        
        recent_predictions = df.tail(10)
        recent_accuracy = recent_predictions[recent_predictions["confidence"] > 0.7].shape[0] / len(recent_predictions) if len(recent_predictions) > 0 else 0
        
        return {
            "total_predictions": total_predictions,
            "disease_distribution": disease_distribution,
            "recent_accuracy": recent_accuracy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))