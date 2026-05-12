# Blockchain module

## Contract

`contract.sol` stores SHA-256 file hashes as strings:

- `storeHash(string hash)` stores `hash` on-chain
- `verifyHash(string hash)` returns whether `hash` exists

Internally it uses a `mapping(bytes32 => bool)` keyed by `keccak256(hash)`.

## Deploy (testnet)

Install Python deps:

```bash
pip install -r requirements.txt
```

Deploy:

```bash
python -m blockchain.deploy --provider "<RPC_URL>" --chain_id <CHAIN_ID> --private_key "<PRIVATE_KEY>"
```

Copy the printed address into `.env` as:

- `CONTRACT_ADDRESS=0x...`

## Backend integration

Backend reads from `.env`:

- `WEB3_PROVIDER_URL`
- `CHAIN_ID`
- `PRIVATE_KEY`
- `CONTRACT_ADDRESS`

