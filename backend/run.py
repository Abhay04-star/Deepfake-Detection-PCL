"""Backend entrypoint: `python -m backend.run`."""

from __future__ import annotations

from .app import create_app
from .config import load_config
from .services.bootstrap import bootstrap_services


def main():
    cfg = load_config()
    services = bootstrap_services(cfg)
    app = create_app(cfg)
    app.extensions["services"] = services
    app.run(host=cfg.host, port=cfg.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

