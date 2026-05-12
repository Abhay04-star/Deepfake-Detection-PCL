"""Interact with deployed DeepfakeHashRegistry.

Usage:
  python -m blockchain.interact store --sha256 <digest>
  python -m blockchain.interact verify --sha256 <digest>

Requires `.env` to be set:
- WEB3_PROVIDER_URL
- CHAIN_ID
- PRIVATE_KEY
- CONTRACT_ADDRESS
"""

from __future__ import annotations

import argparse

from blockchain.web3_client import DeepfakeRegistryClient, load_web3_config_from_env


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_store = sub.add_parser("store")
    sp_store.add_argument("--sha256", required=True)

    sp_verify = sub.add_parser("verify")
    sp_verify.add_argument("--sha256", required=True)

    args = ap.parse_args()

    cfg = load_web3_config_from_env()
    client = DeepfakeRegistryClient(cfg)

    if args.cmd == "store":
        try:
            tx = client.store_hash(args.sha256)
            print({"tx_hash": tx})
        except Exception as e:
            print({"error": str(e)})
        return

    exists = client.verify_hash(args.sha256)
    print({"exists": bool(exists)})


if __name__ == "__main__":
    main()

