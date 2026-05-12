"""Flask app factory and route registration."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS

from backend.config import AppConfig
from backend.routes import ALL_BLUEPRINTS
from backend.services.blockchain_service import verify_hash_on_chain
from backend.services.pipeline import run_detection_pipeline
from backend.services.storage import save_upload_and_hash


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def create_app(cfg: AppConfig) -> Flask:
    """Create and configure the Flask app.

    This file exposes the simple endpoints requested by the project spec:
    - POST /upload  : upload file → ML detect → sha256 → store on blockchain
    - POST /verify  : upload file → sha256 → compare with blockchain

    Internally, the app still keeps a modular structure (services + blueprints).
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_upload_mb * 1024 * 1024

    # Simple CORS for local frontend; tighten for production.
    CORS(app)

    # Register via compatibility list.
    for bp in ALL_BLUEPRINTS:
        app.register_blueprint(bp, url_prefix="/api")

    def _json_error(message: str, status: int = 500):
        return jsonify({"error": str(message)}), status

    @app.post("/upload")
    def upload():
        """Upload an image/video, run detection, hash it, and store hash on-chain."""
        if "file" not in request.files:
            return jsonify({"error": "Missing file field."}), 400
        f = request.files["file"]
        if not f or not f.filename:
            return jsonify({"error": "Empty file."}), 400

        services = app.extensions["services"]
        try:
            result = run_detection_pipeline(
                services=services,
                file_storage=f,
                store_on_chain=True,
            )
            return jsonify(result)
        except Exception as exc:
            app.logger.exception("Upload detection failed")
            return _json_error(exc)

    @app.post("/detect")
    def detect():
        """Upload an image/video and run detection without storing on-chain."""
        if "file" not in request.files:
            return jsonify({"error": "Missing file field."}), 400
        f = request.files["file"]
        if not f or not f.filename:
            return jsonify({"error": "Empty file."}), 400

        services = app.extensions["services"]
        try:
            result = run_detection_pipeline(
                services=services,
                file_storage=f,
                store_on_chain=services.blockchain.mock,
            )
            return jsonify(result)
        except Exception as exc:
            app.logger.exception("Detection failed")
            return _json_error(exc)

    @app.post("/verify")
    def verify():
        """Upload a file, hash it, and compare hash with blockchain."""
        if "file" not in request.files:
            return jsonify({"error": "Missing file field."}), 400
        f = request.files["file"]
        if not f or not f.filename:
            return jsonify({"error": "Empty file."}), 400

        services = app.extensions["services"]
        try:
            saved = save_upload_and_hash(services.storage, f)
            chain = verify_hash_on_chain(services.blockchain, saved["sha256"])
            authenticity = bool(chain.get("enabled")) and bool(chain.get("exists"))

            return jsonify(
                {
                    "file": saved,
                    "chain": chain,
                    "authentic": authenticity,
                }
            )
        except Exception as exc:
            app.logger.exception("Verification failed")
            return _json_error(exc)

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True})

    @app.get("/")
    def index():
        """Serve the frontend index.html"""
        frontend_dir = _repo_root() / "frontend"
        return send_file(str(frontend_dir / "index.html"))

    @app.get("/<path:filename>")
    def serve_static(filename):
        """Serve static files (CSS, JS) from frontend directory"""
        frontend_dir = _repo_root() / "frontend"
        return send_from_directory(str(frontend_dir), filename)

    @app.get("/api")
    def api_root():
        """API documentation."""
        return jsonify({
            "message": "Deepfake Detection API",
            "endpoints": {
                "POST /upload": "Upload image/video → detect deepfake → hash → store on blockchain",
                "POST /verify": "Upload image/video → hash → verify against blockchain",
                "POST /api/upload-detect": "Upload and run detection (store optional)",
                "POST /api/store-hash": "Store SHA-256 hash on blockchain",
                "GET /api/verify-hash/<sha256>": "Verify hash on chain",
                "GET /api/health": "Health check",
            }
        })

    return app

