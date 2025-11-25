import os
import time
import asyncio
import contextlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from paddleocr import TextRecognition
import paddle

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None

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

BATCH_SIZE = int(os.environ.get("OCR_BATCH_SIZE", 8))
BATCH_TIMEOUT = float(os.environ.get("OCR_BATCH_TIMEOUT", 0.008))
QUEUE_SIZE = int(os.environ.get("OCR_QUEUE_SIZE", BATCH_SIZE * 64))
QUEUE_PUT_TIMEOUT = float(os.environ.get("OCR_QUEUE_TIMEOUT", 0.1))

REQUEST_QUEUE: Optional["asyncio.Queue[BatchItem]"] = None
WORKER_TASK: Optional[asyncio.Task] = None

# -----------------------------
# Preprocessing config
# -----------------------------
YOLO_ENABLED = os.environ.get("OCR_YOLO_ENABLED", "1").lower() not in {"0", "false", "no"}
YOLO_MODEL_PATH = os.environ.get("OCR_YOLO_MODEL", "yolov8n.pt")
YOLO_DEVICE = os.environ.get("OCR_YOLO_DEVICE")
YOLO_CONFIDENCE = float(os.environ.get("OCR_YOLO_CONFIDENCE", 0.25))
YOLO_PADDING_RATIO = float(os.environ.get("OCR_YOLO_PADDING", 0.02))
YOLO_CLASS_FILTER = os.environ.get("OCR_YOLO_CLASSES")

YOLO_CLASS_IDS = set()
YOLO_CLASS_NAMES = set()
if YOLO_CLASS_FILTER:
    for token in YOLO_CLASS_FILTER.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            YOLO_CLASS_IDS.add(int(token))
            continue
        try:
            YOLO_CLASS_IDS.add(int(token))
            continue
        except ValueError:
            YOLO_CLASS_NAMES.add(token.lower())

YOLO_DETECTOR = None
if YOLO_ENABLED and YOLO is not None:
    try:
        YOLO_DETECTOR = YOLO(YOLO_MODEL_PATH)
        if YOLO_DEVICE:
            YOLO_DETECTOR.to(YOLO_DEVICE)
    except Exception as exc:  # noqa: BLE001
        YOLO_DETECTOR = None
        print(f"[WARN] Unable to initialize YOLO model: {exc}")


def _detect_roi_with_yolo(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detect the most relevant region-of-interest using YOLO predictions."""
    if YOLO_DETECTOR is None:
        return None

    try:
        results = YOLO_DETECTOR(image, conf=YOLO_CONFIDENCE, verbose=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] YOLO inference failed: {exc}")
        return None

    if not results:
        return None

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return None

    height, width = image.shape[:2]
    x1 = y1 = 0
    x2, y2 = width, height
    found = False

    for box in boxes:
        conf_tensor = getattr(box, "conf", None)
        cls_tensor = getattr(box, "cls", None)
        xyxy_tensor = getattr(box, "xyxy", None)
        if conf_tensor is None or cls_tensor is None or xyxy_tensor is None:
            continue

        conf = float(conf_tensor.item())
        if conf < YOLO_CONFIDENCE:
            continue

        cls_id = int(cls_tensor.item())
        class_name = ""
        if hasattr(result, "names"):
            class_name = result.names.get(cls_id, "")

        if YOLO_CLASS_IDS and cls_id not in YOLO_CLASS_IDS:
            continue

        if YOLO_CLASS_NAMES and class_name.lower() not in YOLO_CLASS_NAMES:
            continue

        coords = xyxy_tensor[0].tolist()
        cand_x1 = max(int(coords[0]), 0)
        cand_y1 = max(int(coords[1]), 0)
        cand_x2 = min(int(coords[2]), width)
        cand_y2 = min(int(coords[3]), height)

        if not found:
            x1, y1, x2, y2 = cand_x1, cand_y1, cand_x2, cand_y2
            found = True
            continue

        x1 = min(x1, cand_x1)
        y1 = min(y1, cand_y1)
        x2 = max(x2, cand_x2)
        y2 = max(y2, cand_y2)

    if not found:
        return None

    pad = int(min(height, width) * YOLO_PADDING_RATIO)
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


def _opencv_preprocess(image: np.ndarray) -> np.ndarray:
    """Apply classic OpenCV steps to denoise and enhance text regions."""
    if image.ndim == 2:
        work_img = image
    else:
        work_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.bilateralFilter(work_img, d=5, sigmaColor=50, sigmaSpace=50)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
    inverted = cv2.bitwise_not(cleaned)
    scale = 1.5
    resized = cv2.resize(
        inverted,
        dsize=None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)


def _save_temp_image(image: np.ndarray) -> str:
    fd, temp_path = tempfile.mkstemp(prefix="ocr_pre_", suffix=".png")
    os.close(fd)
    success = cv2.imwrite(temp_path, image)
    if not success:
        raise RuntimeError("Failed to persist preprocessed image")
    return temp_path


def preprocess_image_for_ocr(image_path: str) -> Tuple[str, Optional[str]]:
    """Run YOLO + OpenCV preprocessing and return a path PaddleOCR can read."""
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Unable to read image: {image_path}")

    roi = _detect_roi_with_yolo(image)
    if roi:
        x1, y1, x2, y2 = roi
        if x2 - x1 > 0 and y2 - y1 > 0:
            image = image[y1:y2, x1:x2]

    processed = _opencv_preprocess(image)
    temp_path = _save_temp_image(processed)
    return temp_path, temp_path

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()


class OCRRequest(BaseModel):
    path: str


# -----------------------------
# OCR prediction (CPU/GPU-bound)
# -----------------------------
def run_ocr_batch(image_paths: List[str]) -> List[dict]:
    if not image_paths:
        return []

    temp_paths: List[str] = []
    preprocessed_inputs: List[str] = []

    try:
        for path in image_paths:
            processed_path, temp_path = preprocess_image_for_ocr(path)
            preprocessed_inputs.append(processed_path)
            if temp_path:
                temp_paths.append(temp_path)

        output = model.predict(
            input=preprocessed_inputs,
            batch_size=min(BATCH_SIZE, len(preprocessed_inputs)),
        )
    finally:
        for temp_path in temp_paths:
            with contextlib.suppress(OSError):
                os.remove(temp_path)

    if not output:
        raise RuntimeError("No prediction returned")

    results = []
    for result in output:
        results.append(
            {
                "text": result.get("rec_text", ""),
                "confidence": float(result.get("rec_score", 0.0)),
                "device": device,
            }
        )
    return results


@dataclass
class BatchItem:
    image_path: str
    future: asyncio.Future


async def batch_worker():
    global REQUEST_QUEUE
    if REQUEST_QUEUE is None:
        return

    loop = asyncio.get_event_loop()
    while True:
        try:
            item: BatchItem = await REQUEST_QUEUE.get()
        except asyncio.CancelledError:
            break

        batch = [item]
        deadline = time.perf_counter() + BATCH_TIMEOUT
        while len(batch) < BATCH_SIZE:
            timeout = deadline - time.perf_counter()
            if timeout <= 0:
                break
            try:
                next_item = await asyncio.wait_for(REQUEST_QUEUE.get(), timeout=timeout)
                batch.append(next_item)
            except asyncio.TimeoutError:
                break

        image_paths = [entry.image_path for entry in batch]
        try:
            results = await loop.run_in_executor(EXECUTOR, run_ocr_batch, image_paths)
            if len(results) != len(batch):
                raise RuntimeError("Batch size mismatch between inputs and outputs")
            for entry, result in zip(batch, results):
                if not entry.future.cancelled():
                    entry.future.set_result(result)
        except Exception as exc:  # noqa: BLE001
            for entry in batch:
                if not entry.future.cancelled():
                    entry.future.set_exception(exc)
        finally:
            for _ in batch:
                REQUEST_QUEUE.task_done()


@app.on_event("startup")
async def on_startup():
    global REQUEST_QUEUE, WORKER_TASK
    REQUEST_QUEUE = asyncio.Queue(maxsize=QUEUE_SIZE)
    WORKER_TASK = asyncio.create_task(batch_worker())


@app.on_event("shutdown")
async def on_shutdown():
    global WORKER_TASK, REQUEST_QUEUE
    if WORKER_TASK:
        WORKER_TASK.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await WORKER_TASK
    WORKER_TASK = None
    REQUEST_QUEUE = None


# -----------------------------
# OCR Endpoint (Local Only)
# -----------------------------
@app.post("/recognize")
async def recognize(req: OCRRequest):
    image_path = req.path
    # Local file check
    if not Path(image_path).exists():
        raise HTTPException(404, "Image not found")

    if REQUEST_QUEUE is None:
        raise HTTPException(503, "Service queue not initialized")

    loop = asyncio.get_event_loop()
    future: asyncio.Future[Any] = loop.create_future()
    item = BatchItem(image_path=image_path, future=future)

    try:
        await asyncio.wait_for(REQUEST_QUEUE.put(item), timeout=QUEUE_PUT_TIMEOUT)
    except asyncio.TimeoutError as exc:
        future.cancel()
        raise HTTPException(503, "Server busy, please retry") from exc
    result = await future
    print("CAPTCHA", result)
    return result


@app.get("/health")
async def health():
    queue_size = REQUEST_QUEUE.qsize() if REQUEST_QUEUE else 0
    return {
        "status": "ok",
        "device": device,
        "workers": OCR_WORKERS,
        "cores": CPU_CORES,
        "batch_size": BATCH_SIZE,
        "queue_depth": queue_size,
    }
