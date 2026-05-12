"""End-to-end pipeline for upload → hash → detect → (optional) store on chain."""

from __future__ import annotations

from typing import Any

from werkzeug.datastructures import FileStorage

from backend.services.blockchain_service import store_hash_on_chain
from backend.services.bootstrap import Services
from backend.services.storage import save_upload_and_hash


def run_detection_pipeline(
    services: Services,
    file_storage: FileStorage,
    store_on_chain: bool = False,
) -> dict[str, Any]:
    """Run the detection pipeline and return a JSON-serializable dict."""
    saved = save_upload_and_hash(services.storage, file_storage)
    pred = services.ml.predict_file(saved["path"])

    chain_resp: dict[str, Any] | None = None
    if store_on_chain:
        chain_resp = store_hash_on_chain(services.blockchain, saved["sha256"])

    return {
        "file": saved,
        "prediction": pred,
        "chain": chain_resp,
    }

