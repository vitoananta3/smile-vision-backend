from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:3000"],  # Add your Next.js frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = YOLO("best.pt")

@app.get("/")
async def hello():
    return {"message": "hello"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Perform prediction
    results = model.predict(image)

    # Get prediction results
    prediction = results[0]

    # Get all classes and their probabilities
    classes_probs = {}
    for i, prob in enumerate(prediction.probs.data):
        class_name = prediction.names[i]
        classes_probs[class_name] = float(prob)

    return {
        "predictions": classes_probs
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8889)