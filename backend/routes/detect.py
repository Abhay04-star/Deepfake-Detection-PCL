"""Detection endpoints: upload, frame extraction, ML inference, hash + optional chain store."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from backend.services.pipeline import run_detection_pipeline


bp = Blueprint("detect", __name__)


@bp.post("/upload-detect")
def upload_detect():
    """Upload an image/video and run deepfake detection.

    Form-data:
      - file: image/video file
      - store_on_chain (optional): "1" or "true" to store sha256 on chain
    """
    if "file" not in request.files:
        return jsonify({"error": "Missing file field."}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "Empty file."}), 400

    store_on_chain = (request.form.get("store_on_chain", "") or "").lower() in {"1", "true", "yes", "on"}

    services = current_app.extensions["services"]
    result = run_detection_pipeline(
        services=services,
        file_storage=file,
        store_on_chain=store_on_chain,
    )
    return jsonify(result)

