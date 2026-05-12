"""Web3 client for interacting with DeepfakeHashRegistry (string hashes).

This module is independent of Flask; it can be used by scripts or services.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from eth_account import Account
from web3 import Web3

from web3.exceptions import ContractLogicError


@dataclass(frozen=True)
class Web3Config:
    provider_url: str
    chain_id: int
    private_key: str
    contract_address: str


def _load_contract_abi() -> list[dict[str, Any]]:
    """Minimal ABI needed by backend (hand-written; avoids compilation dependency)."""
    abi_json = """
    [
      {"inputs":[{"internalType":"string","name":"hash","type":"string"}],
       "name":"storeHash","outputs":[],"stateMutability":"nonpayable","type":"function"},
      {"inputs":[{"internalType":"string","name":"hash","type":"string"}],
       "name":"verifyHash",
       "outputs":[{"internalType":"bool","name":"exists","type":"bool"}],
       "stateMutability":"view","type":"function"}
    ]
    """
    return json.loads(abi_json)


def load_web3_config_from_env() -> Web3Config:
    # Load `.env` if present so scripts can run without manual export.
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=repo_root / ".env", override=False)

    provider_url = os.getenv("WEB3_PROVIDER_URL", "").strip()
    chain_id = int(os.getenv("CHAIN_ID", "0").strip() or "0")
    private_key = os.getenv("PRIVATE_KEY", "").strip()
    contract_address = os.getenv("CONTRACT_ADDRESS", "").strip()

    if not provider_url or not chain_id or not private_key or not contract_address:
        raise ValueError(
            "Missing blockchain config. Set WEB3_PROVIDER_URL, CHAIN_ID, PRIVATE_KEY, CONTRACT_ADDRESS."
        )
    return Web3Config(
        provider_url=provider_url,
        chain_id=chain_id,
        private_key=private_key,
        contract_address=contract_address,
    )


class DeepfakeRegistryClient:
    def __init__(self, cfg: Web3Config):
        self.cfg = cfg
        self.w3 = Web3(Web3.HTTPProvider(cfg.provider_url))
        if not self.w3.is_connected():
            raise ConnectionError("Unable to connect to Web3 provider.")

        self.account = Account.from_key(cfg.private_key)
        self.abi = _load_contract_abi()
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(cfg.contract_address),
            abi=self.abi,
        )

    def store_hash(self, sha256_hex: str) -> str:
        """Store hash on-chain. Returns transaction hash hex string."""
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        txn = self.contract.functions.storeHash(sha256_hex).build_transaction(
            {
                "from": self.account.address,
                "nonce": nonce,
                "chainId": self.cfg.chain_id,
                "gas": 200_000,
                "maxFeePerGas": self.w3.to_wei("30", "gwei"),
                "maxPriorityFeePerGas": self.w3.to_wei("1.5", "gwei"),
            }
        )
        signed = self.account.sign_transaction(txn)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        return tx_hash.hex()

    def verify_hash(self, sha256_hex: str) -> bool:
        """Verify hash exists on-chain."""
        # `verifyHash` returns bool; no tx required.
        try:
            exists = self.contract.functions.verifyHash(sha256_hex).call()
            return bool(exists)
        except ContractLogicError:
            return False

