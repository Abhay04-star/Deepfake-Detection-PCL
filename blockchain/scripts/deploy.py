"""Deploy DeepfakeHashRegistry using Web3.py + solcx.

Usage:
  python -m blockchain.scripts.deploy --provider <RPC_URL> --chain_id <ID> --private_key <KEY>

Output: prints the deployed contract address.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eth_account import Account
from solcx import compile_source, install_solc
from web3 import Web3


def compile_contract(source_path: Path, solc_version: str = "0.8.20"):
    install_solc(solc_version)
    source = source_path.read_text(encoding="utf-8")
    compiled = compile_source(
        source,
        output_values=["abi", "bin"],
        solc_version=solc_version,
    )
    # Only one contract in the file
    _, interface = next(iter(compiled.items()))
    return interface["abi"], interface["bin"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--chain_id", type=int, required=True)
    ap.add_argument("--private_key", required=True)
    args = ap.parse_args()

    w3 = Web3(Web3.HTTPProvider(args.provider))
    if not w3.is_connected():
        raise SystemExit("Could not connect to provider.")

    acct = Account.from_key(args.private_key)
    contract_path = Path(__file__).resolve().parents[1] / "contracts" / "DeepfakeHashRegistry.sol"
    abi, bytecode = compile_contract(contract_path)

    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    nonce = w3.eth.get_transaction_count(acct.address)
    tx = contract.constructor().build_transaction(
        {
            "from": acct.address,
            "nonce": nonce,
            "chainId": args.chain_id,
            "gas": 2_000_000,
            "maxFeePerGas": w3.to_wei("30", "gwei"),
            "maxPriorityFeePerGas": w3.to_wei("1.5", "gwei"),
        }
    )
    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(receipt.contractAddress)


if __name__ == "__main__":
    main()

