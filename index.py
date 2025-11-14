import os
import time
import asyncio
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional
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

BATCH_SIZE = int(os.environ.get("OCR_BATCH_SIZE", 8))
BATCH_TIMEOUT = float(os.environ.get("OCR_BATCH_TIMEOUT", 0.008))
QUEUE_SIZE = int(os.environ.get("OCR_QUEUE_SIZE", BATCH_SIZE * 64))
QUEUE_PUT_TIMEOUT = float(os.environ.get("OCR_QUEUE_TIMEOUT", 0.1))

REQUEST_QUEUE: Optional["asyncio.Queue[BatchItem]"] = None
WORKER_TASK: Optional[asyncio.Task] = None

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

    output = model.predict(
        input=image_paths,
        batch_size=min(BATCH_SIZE, len(image_paths)),
    )

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

    return await future


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
