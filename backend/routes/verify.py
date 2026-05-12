"""Verification endpoints: verify sha256 and verify uploaded file."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from backend.services.blockchain_service import verify_hash_on_chain, store_hash_on_chain
from backend.services.storage import save_upload_and_hash


bp = Blueprint("verify", __name__)


@bp.post("/store-hash")
def store_hash():
    """Store a sha256 digest on blockchain.

    JSON body: { "sha256": "<64-hex>" }
    """
    data = request.get_json(silent=True) or {}
    sha256 = (data.get("sha256") or "").strip().lower()
    if len(sha256) != 64:
        return jsonify({"error": "sha256 must be 64 hex characters."}), 400

    services = current_app.extensions["services"]
    resp = store_hash_on_chain(services.blockchain, sha256)
    return jsonify(resp)


@bp.get("/verify-hash/<sha256>")
def verify_hash(sha256: str):
    sha256 = (sha256 or "").strip().lower()
    if len(sha256) != 64:
        return jsonify({"error": "sha256 must be 64 hex characters."}), 400

    services = current_app.extensions["services"]
    resp = verify_hash_on_chain(services.blockchain, sha256)
    return jsonify(resp)


@bp.post("/verify-file")
def verify_file():
    """Upload a file and verify authenticity by comparing sha256 with blockchain."""
    if "file" not in request.files:
        return jsonify({"error": "Missing file field."}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "Empty file."}), 400

    services = current_app.extensions["services"]
    saved = save_upload_and_hash(services.storage, file)
    chain = verify_hash_on_chain(services.blockchain, saved["sha256"])
    return jsonify({"file": saved, "chain": chain})

