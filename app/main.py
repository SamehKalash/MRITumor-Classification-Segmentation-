import os
import base64
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from model import VIT, IMAGE_SIZE, CLASS_NAMES, preprocess_image, segment_tumor, create_3d_tumor_data

app = FastAPI(title="Brain Tumor Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).parent.parent
WEIGHTS_PATH = BASE_DIR / "checkpoints" / "vit_weights.weights.h5"
STATIC_DIR = Path(__file__).parent / "static"

# Global model
model = None


def load_model():
    global model
    tf.keras.backend.clear_session()
    model = VIT()
    model(tf.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 1)))  # Build model
    
    if WEIGHTS_PATH.exists():
        model.load_weights(str(WEIGHTS_PATH))
        print(f"Model loaded from {WEIGHTS_PATH}")
    else:
        print(f"Warning: Weights not found at {WEIGHTS_PATH}")
    
    return model


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding='utf-8'))
    return HTMLResponse("<h1>Brain Tumor Classifier</h1>")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    
    try:
        contents = await file.read()
        img = preprocess_image(contents)
        results = segment_tumor(model, img)
        
        # Encode images to base64
        display = results['display_image']
        _, buf = cv2.imencode('.png', display)
        display_b64 = base64.b64encode(buf).decode()
        
        attn = (results['attention_map'] * 255).astype(np.uint8)
        attn_colored = cv2.applyColorMap(attn, cv2.COLORMAP_HOT)
        _, buf = cv2.imencode('.png', attn_colored)
        attn_b64 = base64.b64encode(buf).decode()
        
        seg = results['segmentation_mask']
        overlay = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        if results['has_tumor']:
            overlay[:, :, 2] = np.maximum(overlay[:, :, 2], seg)
        _, buf = cv2.imencode('.png', overlay)
        seg_b64 = base64.b64encode(buf).decode()
        
        return JSONResponse({
            "prediction": results['prediction'],
            "confidence": round(results['confidence'], 2),
            "all_confidences": {k: round(v, 2) for k, v in results['all_confidences'].items()},
            "has_tumor": results['has_tumor'],
            "images": {
                "original": f"data:image/png;base64,{display_b64}",
                "attention": f"data:image/png;base64,{attn_b64}",
                "segmentation": f"data:image/png;base64,{seg_b64}"
            },
            "volume_data": create_3d_tumor_data(seg, results['attention_map'])
        })
    
    except Exception as e:
        raise HTTPException(500, str(e))


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
