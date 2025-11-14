import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from pathlib import Path

import requests
from flask import Flask, jsonify, request
from paddleocr import TextRecognition
import paddle

device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
paddle.set_device(device)

model = TextRecognition(model_name="PP-OCRv5_server_rec")
MAX_WORKERS = int(os.environ.get("OCR_WORKERS", max(2, os.cpu_count() or 4)))
REQUEST_TIMEOUT = float(os.environ.get("OCR_TIMEOUT", 30))

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


def _load_image_path(image_path_or_url: str) -> str:
    """Load image from URL or local path, return local file path."""
    parsed = urlparse(image_path_or_url)
    
    # Nếu là URL (http/https)
    if parsed.scheme in ("http", "https"):
        response = requests.get(image_path_or_url, timeout=10)
        response.raise_for_status()
        
        # Tạo file tạm từ URL
        suffix = Path(parsed.path).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(response.content)
            return tmp.name
    
    # Nếu là đường dẫn local
    path = Path(image_path_or_url)
    if not path.exists():
        raise FileNotFoundError(f"Image path not found: {image_path_or_url}")
    return str(path.resolve())


def _predict_from_path(image_path_or_url: str) -> dict:
    temp_file = None
    try:
        image_path = _load_image_path(image_path_or_url)
        
        if image_path_or_url.startswith(("http://", "https://")):
            temp_file = image_path
        
        start_time = time.time()
        output = model.predict(input=image_path, batch_size=1)
        latency = time.time() - start_time
        
        if not output:
            raise RuntimeError("Model did not return any predictions.")
        
        prediction = output[0]
        return {
            "text": prediction.get("rec_text", ""),
            "confidence": float(prediction.get("rec_score", 0.0)),
            "latency_ms": round(latency * 1000, 2),
            "device": device,
        }
    finally:    
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "device": device,
            "workers": MAX_WORKERS,
        }
    )


@app.route("/recognize", methods=["POST"])
def recognize():
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400
    
    image_path = data.get("image_path") or data.get("url") or data.get("path")
    
    if not image_path:
        return jsonify({
            "error": "Missing 'image_path', 'url', or 'path' field in JSON body."
        }), 400
    
    if not isinstance(image_path, str):
        return jsonify({"error": "Image path/URL must be a string."}), 400
    
    future = executor.submit(_predict_from_path, image_path)
    try:
        result = future.result(timeout=REQUEST_TIMEOUT)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except requests.RequestException as exc:
        return jsonify({"error": f"Failed to download image from URL: {str(exc)}"}), 400
    except Exception as exc:
        app.logger.exception("OCR inference failed: %s", exc)
        return jsonify({"error": "Failed to process image."}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), threaded=True)