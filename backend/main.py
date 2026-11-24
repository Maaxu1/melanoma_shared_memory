import multiprocessing as mp

mp.set_start_method("fork", force=True)

import io
import json
import os
import time
from pathlib import Path
from typing import List, Literal

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

from kax_manager import KaxBufferManager

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(BASE_DIR / "melanoma_cnn_classifier.keras"))
)

TARGET_SIZE = (128, 128)

app = FastAPI(title="Melanoma CNN Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model(MODEL_PATH)
kax_buf = KaxBufferManager()


def preprocess_image_bytes(raw_bytes: bytes) -> np.ndarray:
    """Redimensiona a (128, 128, 3) y normaliza a [0.0, 1.0]."""
    with Image.open(io.BytesIO(raw_bytes)) as img:
        rgb_img = img.convert("RGB")
        resized = rgb_img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32) / 255.0


def format_prediction(
    filename: str, raw_prob: float, mode: str, elapsed_ms: float
) -> dict:
    malignant_prob = float(np.clip(raw_prob, 0.0, 1.0))
    benign_prob = float(1.0 - malignant_prob)
    is_malignant = malignant_prob >= 0.5
    return {
        "filename": filename,
        "mode": mode,
        "prob_maligno": round(malignant_prob, 6),
        "prob_benigno": round(benign_prob, 6),
        "diagnostico": "Melanoma maligno" if is_malignant else "Melanoma benigno",
        "confianza": round(max(malignant_prob, benign_prob), 6),
        "tiempo_ms": round(elapsed_ms, 2),
    }


@app.post("/predict/batch-buffer")
async def predict_batch_buffer(
    files: List[UploadFile] = File(...),
    buffer_mode: Literal["ram", "disk"] = Form("ram"),
):
    if not files:
        raise HTTPException(
            status_code=400, detail="Debes enviar al menos una imagen."
        )

    t0 = time.perf_counter()
    raw_files: list[tuple[str, bytes]] = []
    for f in files:
        raw_files.append((f.filename or "imagen.jpg", await f.read()))
    filenames = [name for name, _ in raw_files]

    # --- solo el buffer ---
    t_buf = time.perf_counter()
    t_r = t_buf
    try:
        if buffer_mode == "ram":
            handle = kax_buf.write_ram_buffer(raw_files)
            t_w = time.perf_counter()
            try:
                raw_list = kax_buf.read_ram_buffer(handle)
                t_r = time.perf_counter()
            finally:
                kax_buf.cleanup_ram_buffer(handle)
        else:
            handle = kax_buf.write_disk_buffer(raw_files)
            t_w = time.perf_counter()
            try:
                raw_list = kax_buf.read_disk_buffer(handle)
                t_r = time.perf_counter()
            finally:
                kax_buf.cleanup_disk_buffer(handle)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Error en buffer ({buffer_mode}): {exc}"
        ) from exc
    t_end = time.perf_counter()
    # ----------------------

    try:
        batch_tensor = np.stack(
            [preprocess_image_bytes(r) for r in raw_list], axis=0
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Imagen inválida en el lote: {exc}"
        ) from exc

    preds = model.predict(batch_tensor, verbose=0).reshape(-1)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_item_ms = elapsed_ms / max(len(filenames), 1)

    results = [
        format_prediction(
            fname, float(prob), f"buffer-{buffer_mode}", per_item_ms
        )
        for fname, prob in zip(filenames, preds)
    ]

    return {
        "buffer_mode": buffer_mode,
        "total_imagenes": len(results),
        "tiempo_total_ms": round(elapsed_ms, 2),
        "buffer_ms": {
            "write": round((t_w - t_buf) * 1000, 2),
            "read": round((t_r - t_w) * 1000, 2),
            "cleanup": round((t_end - t_r) * 1000, 2),
            "total": round((t_end - t_buf) * 1000, 2),
        },
        "resultados": results,
    }


@app.post("/predict/batch-stream")
async def predict_batch_stream(files: List[UploadFile] = File(...)):
    """
    Sin buffer compartido: procesa y predice una imagen a la vez,
    esperando a que termine cada predicción antes de pasar a la siguiente.
    Emite cada resultado como línea JSON (NDJSON).
    """
    if not files:
        raise HTTPException(
            status_code=400, detail="Debes enviar al menos una imagen."
        )

    payloads: list[tuple[str, bytes]] = []
    for f in files:
        payloads.append((f.filename or "imagen.jpg", await f.read()))

    def event_generator():
        t_stream_start = time.perf_counter()
        for idx, (fname, raw_bytes) in enumerate(payloads):
            t_item = time.perf_counter()
            try:
                img_arr = preprocess_image_bytes(raw_bytes)
                single_input = np.expand_dims(img_arr, axis=0)
                prob = float(model.predict(single_input, verbose=0)[0][0])
                elapsed_ms = (time.perf_counter() - t_item) * 1000.0

                item = format_prediction(
                    filename=fname,
                    raw_prob=prob,
                    mode="no-buffer",
                    elapsed_ms=elapsed_ms,
                )
                item["indice"] = idx
                yield json.dumps(item) + "\n"
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t_item) * 1000.0
                yield json.dumps({
                    "indice": idx,
                    "filename": fname,
                    "mode": "no-buffer",
                    "error": f"Imagen inválida: {exc}",
                    "tiempo_ms": round(elapsed_ms, 2),
                }) + "\n"

        total_elapsed_ms = round(
            (time.perf_counter() - t_stream_start) * 1000.0, 2
        )
        yield json.dumps(
            {"done": True, "tiempo_total_ms": total_elapsed_ms}
        ) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")