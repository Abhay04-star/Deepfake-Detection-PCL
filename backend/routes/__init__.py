"""API route blueprints."""

from __future__ import annotations

from backend.routes.detect import bp as detect_bp
from backend.routes.verify import bp as verify_bp

ALL_BLUEPRINTS = [detect_bp, verify_bp]
