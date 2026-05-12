"""Deploy DeepfakeHashRegistry contract using Web3.py.

This script deploys `blockchain/contract.sol`.

Environment variables (recommended; can be overridden with args):
- WEB3_PROVIDER_URL
- CHAIN_ID
- PRIVATE_KEY

It prints the deployed `CONTRACT_ADDRESS` so you can copy it to `.env`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from eth_account import Account
from solcx import compile_source, install_solc
from web3 import Web3


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None, help="RPC URL (or use WEB3_PROVIDER_URL)")
    ap.add_argument("--chain_id", type=int, default=None, help="Chain ID (or use CHAIN_ID)")
    ap.add_argument("--private_key", default=None, help="Deployer private key (or use PRIVATE_KEY)")
    ap.add_argument("--solc_version", default="0.8.20", help="Solidity compiler version")
    args = ap.parse_args()

    provider_url = args.provider or os.getenv("WEB3_PROVIDER_URL", "").strip()
    chain_id = args.chain_id if args.chain_id is not None else int(os.getenv("CHAIN_ID", "0").strip() or 0)
    private_key = args.private_key or os.getenv("PRIVATE_KEY", "").strip()

    if not provider_url or not chain_id or not private_key:
        raise SystemExit("Missing provider/chain_id/private_key. Set env vars or pass args.")

    w3 = Web3(Web3.HTTPProvider(provider_url))
    if not w3.is_connected():
        raise SystemExit("Could not connect to Web3 provider.")

    install_solc(args.solc_version)
    contract_path = Path(__file__).resolve().parent / "contract.sol"
    source = contract_path.read_text(encoding="utf-8")
    compiled = compile_source(source, output_values=["abi", "bin"], solc_version=args.solc_version)

    # Choose first compiled contract interface.
    _, interface = next(iter(compiled.items()))
    abi = interface["abi"]
    bytecode = interface["bin"]

    acct = Account.from_key(private_key)
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)

    nonce = w3.eth.get_transaction_count(acct.address)
    tx = contract.constructor().build_transaction(
        {
            "from": acct.address,
            "nonce": nonce,
            "chainId": chain_id,
            "gas": 2_000_000,
            "maxFeePerGas": w3.to_wei("30", "gwei"),
            "maxPriorityFeePerGas": w3.to_wei("1.5", "gwei"),
        }
    )

    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    print(receipt.contractAddress)

