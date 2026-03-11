from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Cervical Cancer Detection API")

# Load model once when server starts
model = tf.keras.models.load_model("cervical_model.keras", compile=False)

# Class labels
classes = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate"
]


# Image preprocessing
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.get("/")
def home():
    return {"message": "Cervical Cancer Detection API is running 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img = preprocess(image)

    prediction = model.predict(img)

    class_id = np.argmax(prediction)
    result = classes[class_id]
    confidence = float(np.max(prediction))

    return {
        "prediction": result,
        "confidence": confidence
    }