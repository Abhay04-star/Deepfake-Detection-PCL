"""Legacy-compatible routes module.

Your requested structure includes a single `backend/routes.py`.
Internally, routes remain split across `backend/routes/` for modularity.
"""

from __future__ import annotations

from backend.routes.detect import bp as detect_bp
from backend.routes.verify import bp as verify_bp

ALL_BLUEPRINTS = [detect_bp, verify_bp]

