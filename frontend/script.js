/**
 * DeepVerify — script.js
 * Modular JavaScript for Deepfake Detection + Blockchain Verification UI.
 * All API endpoints are preserved from the original Flask backend.
 * Backend base: http://127.0.0.1:5000
 */

// ── Configuration ──────────────────────────────────────────────────────────
const CONFIG = {
  API_BASE:      'http://127.0.0.1:5000',
  DETECT_ENDPOINT: '/detect',
  VERIFY_ENDPOINT: '/verify',
  MAX_FILE_SIZE_MB: 50,
  ALLOWED_TYPES: ['image/jpeg', 'image/png', 'image/webp', 'image/gif',
                   'video/mp4', 'video/quicktime', 'video/avi', 'video/x-msvideo'],
  HISTORY_STORAGE_KEY: 'deepverify_history',
  MAX_HISTORY_ITEMS: 10,
};

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  selectedFile: null,
  isDetecting:  false,
  isVerifying:  false,
};

// ── DOM References ─────────────────────────────────────────────────────────
const dom = {
  dropZone:           document.getElementById('dropZone'),
  fileInput:          document.getElementById('fileInput'),
  dropIdle:           document.getElementById('dropIdle'),
  dropPreview:        document.getElementById('dropPreview'),
  previewMediaWrap:   document.getElementById('previewMediaWrap'),
  previewFilename:    document.getElementById('previewFilename'),
  previewFilesize:    document.getElementById('previewFilesize'),
  removeFileBtn:      document.getElementById('removeFile'),
  uploadProgressWrap: document.getElementById('uploadProgressWrap'),
  uploadProgressBar:  document.getElementById('uploadProgressBar'),
  uploadProgressLabel:document.getElementById('uploadProgressLabel'),

  detectBtn:          document.getElementById('detectBtn'),
  verifyBtn:          document.getElementById('verifyBtn'),

  // Detection result
  detectionEmpty:     document.getElementById('detectionEmpty'),
  detectionLoading:   document.getElementById('detectionLoading'),
  detectionResult:    document.getElementById('detectionResult'),
  verdictBadge:       document.getElementById('verdictBadge'),
  verdictIcon:        document.getElementById('verdictIcon'),
  verdictText:        document.getElementById('verdictText'),
  confidenceValue:    document.getElementById('confidenceValue'),
  confidenceBar:      document.getElementById('confidenceBar'),
  detectionMeta:      document.getElementById('detectionMeta'),

  // Blockchain result
  blockchainEmpty:    document.getElementById('blockchainEmpty'),
  blockchainLoading:  document.getElementById('blockchainLoading'),
  blockchainResult:   document.getElementById('blockchainResult'),
  chainIconWrap:      document.getElementById('chainIconWrap'),
  chainStatusLabel:   document.getElementById('chainStatusLabel'),
  chainStatusDesc:    document.getElementById('chainStatusDesc'),
  blockchainNote:     document.getElementById('blockchainNote'),
  blockchainModeText: document.getElementById('blockchainModeText'),
  blockchainModeBanner: document.getElementById('blockchainModeBanner'),
  hashBlock:          document.getElementById('hashBlock'),
  hashValue:          document.getElementById('hashValue'),
  copyHashBtn:        document.getElementById('copyHash'),

  // History
  historyList:        document.getElementById('historyList'),
  clearHistoryBtn:    document.getElementById('clearHistory'),
  historyToggle:      document.getElementById('historyToggle'),

  // Toast container
  toastContainer:     document.getElementById('toastContainer'),

  // Step indicators
  steps:              document.querySelectorAll('.step'),
};

// ── Initialization ─────────────────────────────────────────────────────────
function init() {
  setupDropZone();
  setupButtons();
  setupCopyHash();
  setupHistory();
  setupHistoryToggle();
  renderHistory();
  updateBlockchainMode('Ready to verify', 'This app can verify file integrity with a blockchain record.');
}

function updateBlockchainMode(mode, hint) {
  if (!dom.blockchainModeText) return;
  dom.blockchainModeText.textContent = mode;
  if (dom.blockchainModeBanner) {
    dom.blockchainModeBanner.dataset.mode = mode;
  }
}

function setupHistoryToggle() {
  if (!dom.historyToggle) return;
  dom.historyToggle.addEventListener('click', () => {
    const historySection = document.getElementById('historyCard');
    if (historySection) historySection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
}

// ══════════════════════════════════════════════════════════════════════════
// SECTION 1 — FILE UPLOAD & DROP ZONE
// ══════════════════════════════════════════════════════════════════════════

function setupDropZone() {
  // Click on drop zone opens file picker
  dom.dropZone.addEventListener('click', (e) => {
    // Prevent click propagation from remove button
    if (e.target.closest('#removeFile')) return;
    dom.fileInput.click();
  });

  // Keyboard accessibility: Enter/Space triggers picker
  dom.dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      dom.fileInput.click();
    }
  });

  // File selected via picker
  dom.fileInput.addEventListener('change', () => {
    if (dom.fileInput.files[0]) handleFileSelected(dom.fileInput.files[0]);
  });

  // Remove file button
  dom.removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile();
  });

  // Drag & drop events
  dom.dropZone.addEventListener('dragenter',  onDragEnter, false);
  dom.dropZone.addEventListener('dragover',   onDragOver,  false);
  dom.dropZone.addEventListener('dragleave',  onDragLeave, false);
  dom.dropZone.addEventListener('drop',       onDrop,      false);
}

function onDragEnter(e) { e.preventDefault(); dom.dropZone.classList.add('drag-over'); }
function onDragOver(e)  { e.preventDefault(); dom.dropZone.classList.add('drag-over'); }
function onDragLeave(e) {
  if (!dom.dropZone.contains(e.relatedTarget)) dom.dropZone.classList.remove('drag-over');
}
function onDrop(e) {
  e.preventDefault();
  dom.dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelected(file);
}

/**
 * Validates and stages a selected file.
 * @param {File} file
 */
function handleFileSelected(file) {
  // Type validation
  if (!CONFIG.ALLOWED_TYPES.includes(file.type)) {
    showToast('error', '🚫 Unsupported file type. Use JPG, PNG, MP4, MOV, or AVI.');
    return;
  }

  // Size validation
  const sizeMB = file.size / (1024 * 1024);
  if (sizeMB > CONFIG.MAX_FILE_SIZE_MB) {
    showToast('error', `📦 File too large (${sizeMB.toFixed(1)} MB). Maximum is ${CONFIG.MAX_FILE_SIZE_MB} MB.`);
    return;
  }

  state.selectedFile = file;

  // Show preview
  dom.dropIdle.hidden = true;
  dom.dropPreview.hidden = false;

  // Populate filename + size
  dom.previewFilename.textContent = file.name;
  dom.previewFilesize.textContent = formatBytes(file.size);

  // Media preview (image or video)
  dom.previewMediaWrap.innerHTML = '';
  if (file.type.startsWith('image/')) {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.alt = 'Preview';
    img.onload = () => URL.revokeObjectURL(img.src);
    dom.previewMediaWrap.appendChild(img);
  } else if (file.type.startsWith('video/')) {
    const vid = document.createElement('video');
    vid.src = URL.createObjectURL(file);
    vid.controls = true;
    vid.muted = true;
    vid.style.maxWidth = '100%';
    dom.previewMediaWrap.appendChild(vid);
  }

  // Enable action buttons
  dom.detectBtn.disabled = false;
  dom.verifyBtn.disabled = false;

  // Advance step indicator
  setStep(1);

  showToast('success', `✅ "${file.name}" ready for analysis.`);
}

/** Resets the drop zone back to idle state. */
function clearFile() {
  state.selectedFile = null;
  dom.fileInput.value = '';
  dom.dropPreview.hidden = true;
  dom.dropIdle.hidden = false;
  dom.previewMediaWrap.innerHTML = '';
  dom.detectBtn.disabled = true;
  dom.verifyBtn.disabled = true;
  // Reset results
  showDetectionEmpty();
  showBlockchainEmpty();
  setStep(0);
}

// ══════════════════════════════════════════════════════════════════════════
// SECTION 2 — DETECT DEEPFAKE
// ══════════════════════════════════════════════════════════════════════════

function setupButtons() {
  dom.detectBtn.addEventListener('click', runDetect);
  dom.verifyBtn.addEventListener('click', runVerify);
}

/**
 * Calls the /detect endpoint with the selected file.
 * Expects JSON response: { result: "REAL"|"FAKE", confidence: 0.0-1.0, ... }
 */
async function runDetect() {
  if (!state.selectedFile || state.isDetecting) return;

  state.isDetecting = true;
  setButtonLoading(dom.detectBtn, true);
  showDetectionLoading();
  setStep(1);
  showToast('info', '🔍 Analyzing media for deepfake signatures…');

  // Simulate upload progress bar
  simulateUploadProgress();

  const formData = new FormData();
  formData.append('file', state.selectedFile);

  try {
    const response = await fetch(`${CONFIG.API_BASE}${CONFIG.DETECT_ENDPOINT}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ error: 'Server error' }));
      throw new Error(err.error || `HTTP ${response.status}`);
    }

    const data = await response.json();
    renderDetectionResult(data);
    setStep(2);
    showToast('success', `🎯 Detection complete: ${data.result || 'Result received'}`);

    // Add to scan history
    addToHistory({
      filename: state.selectedFile.name,
      filesize: state.selectedFile.size,
      verdict:  (data.result || '').toUpperCase(),
      confidence: data.confidence,
      fileType: state.selectedFile.type,
      timestamp: Date.now(),
    });

  } catch (err) {
    showDetectionEmpty();
    showToast('error', `❌ Detection failed: ${err.message}`);
    console.error('[DeepVerify] Detection error:', err);
  } finally {
    state.isDetecting = false;
    setButtonLoading(dom.detectBtn, false);
    hideUploadProgress();
  }
}

/**
 * Renders the detection result into the result card.
 * @param {Object} data - API response object
 */
function renderDetectionResult(data) {
  const prediction   = data.prediction || data;
  const result       = (prediction.label || prediction.result || 'UNKNOWN').toUpperCase();
  const confidence   = parseFloat(prediction.confidence ?? 0);
  const isReal       = result === 'REAL';
  const isFake       = result === 'FAKE';

  // Show result section
  dom.detectionEmpty.hidden = true;
  dom.detectionLoading.hidden = true;
  dom.detectionResult.hidden = false;

  // Verdict badge
  dom.verdictBadge.className = 'verdict-badge ' + (isReal ? 'real' : isFake ? 'fake' : '');
  dom.verdictIcon.textContent  = isReal ? '✅' : isFake ? '🚫' : '❓';
  dom.verdictText.textContent  = result;

  // Confidence bar (animate on next frame)
  const pct = Math.round(confidence * 100);
  dom.confidenceValue.textContent = `${pct}%`;
  requestAnimationFrame(() => {
    dom.confidenceBar.style.width = `${pct}%`;
    // Color the bar based on result
    dom.confidenceBar.style.background = isReal
      ? 'linear-gradient(90deg, #059669, #10b981)'
      : isFake
        ? 'linear-gradient(90deg, #b91c1c, #ef4444)'
        : 'linear-gradient(135deg, #7c3aed, #06b6d4)';
  });

  // Optional metadata lines
  const metaLines = [];
  if (prediction.model)   metaLines.push(`Model: ${prediction.model}`);
  if (prediction.p_fake != null) metaLines.push(`Fake probability: ${(prediction.p_fake * 100).toFixed(1)}%`);
  if (data.file?.sha256)  metaLines.push(`SHA-256: ${data.file.sha256.slice(0, 16)}…`);
  if (data.file?.uploaded_at) metaLines.push(`Uploaded: ${new Date(data.file.uploaded_at).toLocaleTimeString()}`);

  dom.detectionMeta.innerHTML  = metaLines.length
    ? metaLines.map(l => `<span>${l}</span><br>`).join('')
    : '';
  dom.detectionMeta.hidden = metaLines.length === 0;
}

// ══════════════════════════════════════════════════════════════════════════
// SECTION 3 — BLOCKCHAIN VERIFICATION
// ══════════════════════════════════════════════════════════════════════════

/**
 * Calls the /verify endpoint with the selected file.
 * Expects JSON: { status: "verified"|"not_found"|"tampered", hash: "...", message: "..." }
 */
async function runVerify() {
  if (!state.selectedFile || state.isVerifying) return;

  state.isVerifying = true;
  setButtonLoading(dom.verifyBtn, true);
  showBlockchainLoading();
  setStep(2);
  showToast('info', '⛓️ Querying Ethereum blockchain…');

  const formData = new FormData();
  formData.append('file', state.selectedFile);

  try {
    const response = await fetch(`${CONFIG.API_BASE}${CONFIG.VERIFY_ENDPOINT}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ error: 'Server error' }));
      throw new Error(err.error || `HTTP ${response.status}`);
    }

    const data = await response.json();
    renderBlockchainResult(data);
    setStep(3);
    showToast('success', '⛓️ Blockchain verification complete.');

  } catch (err) {
    showBlockchainEmpty();
    showToast('error', `❌ Verification failed: ${err.message}`);
    console.error('[DeepVerify] Blockchain error:', err);
  } finally {
    state.isVerifying = false;
    setButtonLoading(dom.verifyBtn, false);
  }
}

/**
 * Maps blockchain status to display data.
 */
const CHAIN_STATUS_MAP = {
  verified:  {
    cls:   'verified',
    icon:  '✅',
    label: 'Verified Authentic',
    desc:  'The file hash matches a record on the Ethereum blockchain. This file has not been tampered with.',
    color: 'var(--green)',
  },
  not_found: {
    cls:   'not-found',
    icon:  '🔍',
    label: 'Not Found on Chain',
    desc:  'No matching record found on the blockchain. This file may not have been registered yet.',
    color: 'var(--amber)',
  },
  tampered: {
    cls:   'tampered',
    icon:  '⚠️',
    label: 'Possible Tampering Detected',
    desc:  'The file hash does NOT match the blockchain record. This file may have been altered.',
    color: 'var(--red)',
  },
};

/**
 * @param {Object} data - API response
 */
function renderBlockchainResult(data) {
  let status = 'not_found';
  if (data.chain?.enabled === false) {
    status = 'not_found';
  } else if (data.chain?.exists) {
    status = 'verified';
  } else if (data.chain?.enabled === true && !data.chain.exists) {
    status = 'tampered';
  }

  const info = CHAIN_STATUS_MAP[status] || CHAIN_STATUS_MAP.not_found;

  dom.blockchainEmpty.hidden   = true;
  dom.blockchainLoading.hidden = true;
  dom.blockchainResult.hidden  = false;

  // Icon + status
  dom.chainIconWrap.className       = `chain-icon-wrap ${info.cls}`;
  dom.chainIconWrap.textContent     = info.icon;
  dom.chainStatusLabel.textContent  = info.label;
  dom.chainStatusLabel.style.color  = info.color;
  dom.chainStatusDesc.textContent   = data.chain?.error || data.message || info.desc;

  const blockchainMode = data.chain?.mock ? 'Local fallback' : 'Ethereum';
  updateBlockchainMode(blockchainMode, data.chain?.mock ? 'Using local fallback because blockchain config is not provided.' : 'Verified against the configured Ethereum contract.');
  dom.blockchainNote.textContent = data.chain?.mock
    ? 'Local verification is used when Ethereum configuration is missing or unavailable.'
    : 'This file is verified against the configured Ethereum blockchain record.';

  const hashText = data.file?.sha256 || data.hash || '—';
  if (hashText && hashText !== '—') {
    dom.hashValue.textContent = hashText;
    dom.hashBlock.hidden = false;
  } else {
    dom.hashBlock.hidden = true;
  }
}

// ══════════════════════════════════════════════════════════════════════════
// SECTION 4 — SCAN HISTORY
// ══════════════════════════════════════════════════════════════════════════

function setupHistory() {
  dom.clearHistoryBtn.addEventListener('click', () => {
    if (confirm('Clear all scan history?')) {
      localStorage.removeItem(CONFIG.HISTORY_STORAGE_KEY);
      renderHistory();
      showToast('info', '🗑️ Scan history cleared.');
    }
  });
}

/** Adds an entry to scan history in localStorage. */
function addToHistory(entry) {
  let history = getHistory();
  history.unshift(entry);
  history = history.slice(0, CONFIG.MAX_HISTORY_ITEMS);
  try {
    localStorage.setItem(CONFIG.HISTORY_STORAGE_KEY, JSON.stringify(history));
  } catch (_) { /* storage quota exceeded — silently ignore */ }
  renderHistory();
}

/** Reads history from localStorage. */
function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(CONFIG.HISTORY_STORAGE_KEY) || '[]');
  } catch (_) { return []; }
}

/** Renders the history list into the DOM. */
function renderHistory() {
  const history = getHistory();
  dom.historyList.innerHTML = '';

  if (history.length === 0) {
    dom.historyList.innerHTML = '<p class="history-empty">No scans yet — results will appear here.</p>';
    return;
  }

  history.forEach((entry, idx) => {
    const item = document.createElement('div');
    item.className = 'history-item';
    item.style.animationDelay = `${idx * 0.04}s`;

    const verdictClass = entry.verdict === 'REAL' ? 'real' : entry.verdict === 'FAKE' ? 'fake' : 'unknown';
    const confidence   = entry.confidence != null ? Math.round(entry.confidence * 100) + '%' : '—';
    const time         = entry.timestamp  ? new Date(entry.timestamp).toLocaleTimeString() : '';
    const isImage      = (entry.fileType || '').startsWith('image/');

    item.innerHTML = `
      <div class="history-thumb" aria-hidden="true">${isImage ? '🖼️' : '🎬'}</div>
      <div class="history-details">
        <p class="history-filename" title="${escapeHtml(entry.filename)}">${escapeHtml(entry.filename)}</p>
        <p class="history-meta">${confidence} confidence &nbsp;·&nbsp; ${time}</p>
      </div>
      <span class="history-verdict ${verdictClass}">${entry.verdict || '—'}</span>
    `;

    dom.historyList.appendChild(item);
  });
}

// ══════════════════════════════════════════════════════════════════════════
// SECTION 5 — TOAST NOTIFICATIONS
// ══════════════════════════════════════════════════════════════════════════

/**
 * Displays a toast notification.
 * @param {'success'|'error'|'info'|'warning'} type
 * @param {string} message
 * @param {number} duration  ms to auto-dismiss (default 4000)
 */
function showToast(type, message, duration = 4000) {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;

  const iconMap = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
  toast.innerHTML = `
    <span class="toast-icon">${iconMap[type] || '💬'}</span>
    <span class="toast-message">${escapeHtml(message)}</span>
  `;

  dom.toastContainer.appendChild(toast);

  // Auto-dismiss
  setTimeout(() => {
    toast.classList.add('exiting');
    toast.addEventListener('animationend', () => toast.remove(), { once: true });
  }, duration);
}

// ══════════════════════════════════════════════════════════════════════════
// SECTION 6 — STEP INDICATORS
// ══════════════════════════════════════════════════════════════════════════

/**
 * Sets the active / done steps.
 * @param {number} activeIndex  0 = upload, 1 = detect, 2 = verify
 */
function setStep(activeIndex) {
  dom.steps.forEach((step, idx) => {
    step.classList.remove('active', 'done');
    if (idx < activeIndex)  step.classList.add('done');
    if (idx === activeIndex) step.classList.add('active');
  });
}

// ══════════════════════════════════════════════════════════════════════════
// SECTION 7 — HELPERS & UTILITIES
// ══════════════════════════════════════════════════════════════════════════

/** Toggle loading state on a button. */
function setButtonLoading(btn, loading) {
  const label   = btn.querySelector('.btn-label');
  const spinner = btn.querySelector('.btn-spinner');
  const icon    = btn.querySelector('.btn-icon');

  if (loading) {
    btn.disabled = true;
    if (label)   label.style.opacity = '0.5';
    if (icon)    icon.hidden = true;
    if (spinner) spinner.hidden = false;
  } else {
    btn.disabled = state.selectedFile === null;
    if (label)   label.style.opacity = '1';
    if (icon)    icon.hidden = false;
    if (spinner) spinner.hidden = true;
  }
}

/** Shows the detection empty/placeholder state. */
function showDetectionEmpty() {
  dom.detectionEmpty.hidden   = false;
  dom.detectionLoading.hidden = true;
  dom.detectionResult.hidden  = true;
}

/** Shows the detection loading state. */
function showDetectionLoading() {
  dom.detectionEmpty.hidden   = true;
  dom.detectionLoading.hidden = false;
  dom.detectionResult.hidden  = true;
}

/** Shows the blockchain empty/placeholder state. */
function showBlockchainEmpty() {
  dom.blockchainEmpty.hidden   = false;
  dom.blockchainLoading.hidden = true;
  dom.blockchainResult.hidden  = true;
}

/** Shows the blockchain loading state. */
function showBlockchainLoading() {
  dom.blockchainEmpty.hidden   = true;
  dom.blockchainLoading.hidden = false;
  dom.blockchainResult.hidden  = true;
}

/** Simulates a fake upload progress bar (visual polish while API is running). */
function simulateUploadProgress() {
  dom.uploadProgressWrap.hidden = false;
  let pct = 0;
  const step = () => {
    pct = Math.min(pct + Math.random() * 12 + 4, 90);
    dom.uploadProgressBar.style.width = `${pct}%`;
    dom.uploadProgressLabel.textContent = `Uploading… ${Math.round(pct)}%`;
    if (pct < 90 && state.isDetecting) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/** Hides the upload progress bar. */
function hideUploadProgress() {
  dom.uploadProgressBar.style.width = '100%';
  dom.uploadProgressLabel.textContent = 'Upload complete';
  setTimeout(() => { dom.uploadProgressWrap.hidden = true; }, 600);
}

/** Formats bytes to a human-readable string. */
function formatBytes(bytes) {
  if (bytes < 1024)        return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

/** Escapes HTML to prevent XSS in user-supplied filenames. */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** Sets up the copy-hash button. */
function setupCopyHash() {
  dom.copyHashBtn.addEventListener('click', async () => {
    const hash = dom.hashValue.textContent;
    if (!hash || hash === '—') return;
    try {
      await navigator.clipboard.writeText(hash);
      showToast('success', '📋 Hash copied to clipboard!');
    } catch (_) {
      showToast('error', '❌ Unable to copy — please copy manually.');
    }
  });
}

// ── Boot ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);

