from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# ── App ────────────────────────────────────────
app = FastAPI(
    title="CervAI · Cervical Cell Detection API",
    description="MobileNetV2-based cervical cell morphology classifier",
    version="2.2.0"
)

# ── CORS ───────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cer-ai-five.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Class Labels ───────────────────────────────
CLASSES = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate"
]

# ── Model Loading (robust) ─────────────────────
BASE_DIR = os.path.dirname(__file__)
KERAS_MODEL = os.path.join(BASE_DIR, "cervical_model.keras")
H5_MODEL    = os.path.join(BASE_DIR, "cervical_model.h5")

model = None

print("📂 Files inside container:", os.listdir(BASE_DIR))

try:
    if os.path.exists(H5_MODEL):
        print("🔄 Loading H5 model...")
        model = tf.keras.models.load_model(H5_MODEL, compile=False)
    elif os.path.exists(KERAS_MODEL):
        print("🔄 Loading Keras model...")
        model = tf.keras.models.load_model(KERAS_MODEL, compile=False)
    else:
        raise FileNotFoundError("No model file found.")

    print("✅ Model loaded successfully!")

except Exception as e:
    print("❌ Model load failed:", e)
    model = None


# ── Accepted MIME types ────────────────────────
ALLOWED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "application/octet-stream",
    "",
}


# ── Preprocessing ──────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# ── Routes ─────────────────────────────────────
@app.get("/")
def home():
    return {
        "status": "online",
        "version": "2.2.0",
        "model": "loaded" if model is not None else "not loaded",
        "classes": CLASSES
    }


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "ready"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    print("📁 File:", file.filename)
    print("📁 Content-Type:", file.content_type)

    content_type = (file.content_type or "").lower()

    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}"
        )

    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")

    try:
        contents = await file.read()

        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        img = preprocess(image)

        prediction = model.predict(img, verbose=0)

        class_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        label = CLASSES[class_id]

        probabilities = {
            CLASSES[i]: round(float(prediction[0][i]) * 100, 2)
            for i in range(len(CLASSES))
        }

        print(f"✅ Prediction: {label} ({confidence*100:.2f}%)")

        return {
            "prediction": label,
            "confidence": confidence,
            "class_id": class_id,
            "probabilities": probabilities
        }

    except Exception as e:
        print("❌ Prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))

