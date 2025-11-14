import os
import time
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from paddleocr import TextRecognition
import paddle

# -----------------------------
# GPU / Model init
# -----------------------------
device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
paddle.set_device(device)

model = TextRecognition(model_name="PP-OCRv5_server_rec")

# -----------------------------
# Config workers
# -----------------------------
CPU_CORES = os.cpu_count() or 4
OCR_WORKERS = int(os.environ.get("OCR_WORKERS", max(2, CPU_CORES)))
EXECUTOR = ThreadPoolExecutor(max_workers=OCR_WORKERS)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()


class OCRRequest(BaseModel):
    path: str


# -----------------------------
# OCR prediction (CPU/GPU-bound)
# -----------------------------
def run_ocr(image_path: str) -> dict:
    output = model.predict(input=image_path, batch_size=1)

    if not output:
        raise RuntimeError("No prediction returned")

    result = output[0]

    return {
        "text": result.get("rec_text", ""),
        "confidence": float(result.get("rec_score", 0.0)),
        "device": device,
    }


# -----------------------------
# OCR Endpoint (Local Only)
# -----------------------------
@app.post("/recognize")
async def recognize(req: OCRRequest):
    image_path = req.path

    # Local file check
    if not Path(image_path).exists():
        raise HTTPException(404, "Image not found")

    # Submit to worker thread (non-blocking)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(EXECUTOR, run_ocr, image_path)
    return result


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": device,
        "workers": OCR_WORKERS,
        "cores": CPU_CORES,
    }
