<<<<<<< HEAD
# Deepfake Detection with Blockchain Verification

## Project overview

Full-stack project that:

- Accepts an **image or video** upload
- Extracts frames (for videos) using OpenCV + selects key frames for prediction
- Runs a **ResNet50 transfer-learning CNN** (TensorFlow/Keras) to predict **REAL / FAKE** with confidence
- Computes **SHA-256** of the original uploaded file
- Stores and verifies the SHA-256 on an **Ethereum testnet** via a Solidity smart contract + Web3.py
  - Contract interface: `storeHash(string)` / `verifyHash(string)`

> Note: You must provide trained weights (or train your own) to get meaningful REAL/FAKE predictions.

## Folder structure

- `backend/` Flask API
- `ml_model/` ResNet50 model + prediction + preprocessing
- `blockchain/` Solidity contract + Web3 client + deploy/interact scripts
- `frontend/` HTML/CSS/JS UI
- `utils/` Shared hashing + video helpers

## Setup steps

### 1) Create venv + install deps

Windows (PowerShell):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Configure environment

Copy `.env.example` to `.env`.

Minimum required to run detection:

- Put a model at `ml_model/weights/resnet50_deepfake.h5` (recommended), or set `MODEL_WEIGHTS_PATH` in `.env`

To enable blockchain features, set in `.env`:

- `WEB3_PROVIDER_URL`
- `CHAIN_ID`
- `PRIVATE_KEY`
- `CONTRACT_ADDRESS`

## How to run backend

```bash
python -m backend.run
```

Backend runs at `http://127.0.0.1:5000`.

## How to deploy smart contract

1) Ensure `.env` has:

- `WEB3_PROVIDER_URL`
- `CHAIN_ID`
- `PRIVATE_KEY`

2) Deploy:

```bash
python -m blockchain.deploy --provider "<RPC_URL>" --chain_id <CHAIN_ID> --private_key "<PRIVATE_KEY>"
```

3) Copy the printed address into `.env`:

- `CONTRACT_ADDRESS=0x...`

## How to test system

### 1) Health check

Open `http://127.0.0.1:5000/api/health` and confirm it returns:

```json
{"ok": true}
```

### 2) Run the frontend

Option A: open `frontend/index.html` in your browser.

Option B:

```bash
python -m http.server 8080 --directory frontend
```

Then open `http://127.0.0.1:8080`.

### 3) Detection test (ML)

- Upload an image/video
- Click **Detect**
- Confirm output includes:
  - `Label: REAL|FAKE`
  - `Confidence: ...`
  - `SHA-256: ...`

### 4) Verification test (Blockchain)

Prereqs:

- Contract deployed
- `.env` configured with blockchain variables
- Test account has testnet ETH for gas

Steps:

- Upload a file and click **Detect** (this stores the hash on-chain when blockchain is enabled)
- Click **Verify** on the same file
- Confirm output shows `Authentic: true`

### 5) CLI test (optional)

Store:

```bash
python -m blockchain.interact store --sha256 <64-hex-digest>
```

Verify:

```bash
python -m blockchain.interact verify --sha256 <64-hex-digest>
```

## API (quick reference)

- `POST /upload` (multipart form-data: `file`)
- `POST /verify` (multipart form-data: `file`)
- `GET /api/health`


=======
# Deepfake-Detection_PCL
>>>>>>> e55e75fe2773837984fd30d0c13ab17172abb1a6
