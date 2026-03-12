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
    version="2.1.0"
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

# ── Load Model ─────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cervical_model.keras")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    model = None

# ── Class Labels ───────────────────────────────
CLASSES = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate"
]

# ── Accepted MIME types ────────────────────────
# Browsers sometimes send "application/octet-stream"
# for drag-and-dropped files, so we accept it too
# and let Pillow validate the actual image content.
ALLOWED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "application/octet-stream",  # drag & drop fallback
    "",                           # some browsers send empty string
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
        "status":  "online",
        "message": "CervAI API is running 🚀",
        "version": "2.1.0",
        "model":   "loaded" if model is not None else "not loaded",
        "classes": CLASSES,
        "endpoints": {
            "predict": "POST /predict",
            "health":  "GET /health",
            "docs":    "GET /docs"
        }
    }

@app.get("/health")
def health():
    """Railway uses this to check if the service is alive"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model":  "ready"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # ── Debug log (visible in Railway logs too) ──
    print(f"📁 Received file  : {file.filename}")
    print(f"📁 Content-Type   : {file.content_type}")

    # ── Soft content-type check ──────────────────
    # We don't hard-reject here — Pillow will raise
    # an error itself if the bytes aren't a valid image
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file.content_type}'. Please upload a JPEG or PNG image."
        )

    # ── Model check ──────────────────────────────
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available. Try again shortly.")

    try:
        # Read bytes
        contents = await file.read()

        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Decode image — Pillow validates it's a real image
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess
        img = preprocess(image)

        # Inference
        prediction = model.predict(img, verbose=0)

        class_id   = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        label      = CLASSES[class_id]

        # Full probability distribution
        probabilities = {
            CLASSES[i]: round(float(prediction[0][i]) * 100, 2)
            for i in range(len(CLASSES))
        }

        print(f"✅ Prediction: {label} ({confidence * 100:.2f}%)")

        return {
            "prediction":    label,
            "confidence":    confidence,
            "probabilities": probabilities,
            "class_id":      class_id
        }

    except HTTPException:
        raise  # re-raise our own HTTP errors as-is

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )