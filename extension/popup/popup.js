// ============================================================
// popup.js — Dark Pattern Detector Popup Controller
// ============================================================

const CAT_META = {
  urgency:         { label: 'Fake Urgency',        color: '#ef4444' },
  optin:           { label: 'Pre-checked Opt-in',  color: '#f97316' },
  confirm_shaming: { label: 'Confirm-shaming',     color: '#a855f7' },
  hidden_flows:    { label: 'Hidden Flows',         color: '#06b6d4' },
  disguised_ads:   { label: 'Disguised Ads',        color: '#eab308' },
};

const GRADE_COLORS = {
  A: '#22c55e', B: '#84cc16', C: '#f59e0b', D: '#f97316', F: '#ef4444'
};

// ─── DOM references ──────────────────────────────────────────
const $ = id => document.getElementById(id);

const states = {
  idle:     $('js-state-idle'),
  scanning: $('js-state-scanning'),
  results:  $('js-state-results'),
  error:    $('js-state-error'),
};

// ─── State management ────────────────────────────────────────
function showState(name) {
  Object.entries(states).forEach(([k, el]) => {
    el.classList.toggle('hidden', k !== name);
  });
}

// ─── Initialise popup ────────────────────────────────────────
async function init() {
  // Load auto-scan setting
  const { autoScan } = await chrome.storage.local.get('autoScan');
  $('js-auto-scan').checked = autoScan !== false;

  // Get active tab info
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) { showState('error'); $('js-error-msg').textContent = 'No active tab found.'; return; }

  // Show URL
  try {
    const url = new URL(tab.url);
    $('js-site-url').textContent = url.hostname;
  } catch {
    $('js-site-url').textContent = tab.url?.slice(0, 40) || '—';
  }

  // Ask background if scan is in progress
  const statusRes = await chrome.runtime.sendMessage({ type: 'SCAN_STATUS', tabId: tab.id }).catch(() => null);
  if (statusRes?.inProgress) {
    showState('scanning');
    return;
  }

  // Ask background for existing results
  const resultRes = await chrome.runtime.sendMessage({ type: 'GET_RESULTS', tabId: tab.id }).catch(() => null);
  if (resultRes?.result) {
    if (resultRes.result.error) {
      showError(resultRes.result.error);
    } else {
      renderResults(resultRes.result);
    }
    return;
  }

  // No results yet — show idle
  showState('idle');
}

// ─── Listen for messages from background ─────────────────────
chrome.runtime.onMessage.addListener((message) => {
  switch (message.type) {
    case 'SCAN_STARTED':
      showState('scanning');
      animateScanSteps();
      break;

    case 'SCAN_COMPLETE':
      if (message.result?.error) {
        showError(message.result.error);
      } else {
        renderResults(message.result);
      }
      break;

    case 'SCAN_ERROR':
      showError(message.error || 'Scan failed');
      break;
  }
});

// ─── Render results ──────────────────────────────────────────
function renderResults(result) {
  showState('results');

  const { trust_score, grade, detections = [], category_scores = {}, processing_ms, model_info = {} } = result;

  // Score gauge animation
  animateGauge(trust_score, grade);

  // Detection count + timing
  const issues = detections.length;
  $('js-detection-count').textContent = `${issues} issue${issues !== 1 ? 's' : ''} found`;
  $('js-scan-time').textContent = processing_ms ? `${processing_ms}ms` : '';

  // Grade class on body for glow effect
  document.body.className = `grade-${grade}`;

  // Categories
  renderCategories(category_scores);

  // Detections list
  renderDetections(detections);

  // Model info
  const nlpType = model_info.nlp || 'unknown';
  const cvType  = model_info.cv  || 'unknown';
  $('js-model-info').textContent = `NLP: ${nlpType} · CV: ${cvType}`;

  // Show/hide detections section
  $('js-detections-section').style.display = issues > 0 ? 'block' : 'none';
}

function animateGauge(score, grade) {
  const gaugeFill = $('js-gauge-fill');
  const scoreNum  = $('js-score-num');
  const scoreGrade = $('js-score-grade');
  const scoreLabelEl = $('js-score-label');

  // Arc total path length ≈ 157 (π × 50)
  const totalArc = 157;
  const fraction = Math.max(0, Math.min(100, score)) / 100;
  const dashOffset = totalArc * (1 - fraction);

  // Set gauge color
  const color = GRADE_COLORS[grade] || '#64748b';
  gaugeFill.style.transition = 'stroke-dashoffset 1s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
  gaugeFill.style.stroke = color;

  // Force reflow then animate
  requestAnimationFrame(() => {
    gaugeFill.style.strokeDashoffset = String(dashOffset);
  });

  // Animate counter
  const duration = 900;
  const start = performance.now();
  function step(now) {
    const t = Math.min(1, (now - start) / duration);
    const ease = 1 - Math.pow(1 - t, 3);
    scoreNum.textContent = String(Math.round(ease * score));
    if (t < 1) requestAnimationFrame(step);
    else scoreNum.textContent = String(score);
  }
  requestAnimationFrame(step);

  scoreNum.style.color = color;
  scoreGrade.textContent = `Grade ${grade}`;
  scoreGrade.style.color = color;

  // Score label descriptive text
  const labels = { A: 'Excellent', B: 'Good', C: 'Moderate', D: 'Suspicious', F: 'Deceptive' };
  scoreLabelEl.textContent = labels[grade] || 'Trust Score';
}

function renderCategories(categoryScores) {
  const container = $('js-categories');
  container.innerHTML = '';

  for (const [key, meta] of Object.entries(CAT_META)) {
    const score = categoryScores[key] || 0;
    const pct   = Math.round(score * 100);

    const row = document.createElement('div');
    row.className = 'category-row';
    row.innerHTML = `
      <span class="cat-label">${meta.label}</span>
      <div class="cat-bar-track">
        <div class="cat-bar-fill bar-${key}" data-target="${pct}" style="width:0%"></div>
      </div>
      <span class="cat-pct">${pct}%</span>
    `;
    container.appendChild(row);
  }

  // Animate bars after paint
  requestAnimationFrame(() => {
    container.querySelectorAll('.cat-bar-fill').forEach(el => {
      el.style.width = el.dataset.target + '%';
    });
  });
}

function renderDetections(detections) {
  const list = $('js-detections-list');
  list.innerHTML = '';

  if (!detections.length) return;

  detections.slice(0, 20).forEach((det, i) => {
    const cat     = det.type || 'general';
    const catMeta = CAT_META[cat] || { label: 'Dark Pattern', color: '#64748b' };
    const sev     = det.severity || 'low';

    const item = document.createElement('div');
    item.className = 'detection-item';
    item.dataset.cat = cat;
    item.style.animationDelay = `${i * 0.05}s`;

    item.innerHTML = `
      <div class="detection-header">
        <span class="detection-badge">${catMeta.label}</span>
        <span class="detection-severity sev-${sev}">● ${sev.toUpperCase()}</span>
      </div>
      <div class="detection-desc">${escapeHtml(det.description || '')}</div>
      ${det.text ? `<div class="detection-text">"${escapeHtml(det.text)}"</div>` : ''}
    `;

    // Click → scroll to element on page
    if (det.selector || det.rect) {
      item.addEventListener('click', () => scrollToElement(det));
    }

    list.appendChild(item);
  });

  if (detections.length > 20) {
    const more = document.createElement('div');
    more.style.cssText = 'text-align:center;font-size:11px;color:var(--text-muted);padding:4px 0';
    more.textContent = `+${detections.length - 20} more issues`;
    list.appendChild(more);
  }
}

// ─── Error rendering ─────────────────────────────────────────
function showError(msg) {
  showState('error');
  $('js-error-msg').textContent = msg || 'An unknown error occurred.';
}

// ─── Scroll to detected element ──────────────────────────────
async function scrollToElement(detection) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;

  if (detection.selector) {
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: (selector) => {
        try {
          const el = document.querySelector(selector);
          if (el) {
            el.scrollIntoView({ behavior: 'smooth', block: 'center' });
            el.style.outline = '3px solid #6366f1';
            setTimeout(() => el.style.outline = '', 2500);
          }
        } catch {}
      },
      args: [detection.selector]
    }).catch(() => {});
  } else if (detection.rect) {
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: (y) => { window.scrollTo({ top: y - 200, behavior: 'smooth' }); },
      args: [detection.rect.y]
    }).catch(() => {});
  }
}

// ─── Scan step animation ─────────────────────────────────────
const SCAN_STEPS = [
  'Extracting DOM elements…',
  'Running NLP analysis…',
  'Processing screenshot…',
  'Computing trust score…',
  'Rendering results…',
];
let stepTimer;
function animateScanSteps() {
  clearInterval(stepTimer);
  let idx = 0;
  $('js-scan-step').textContent = SCAN_STEPS[0];
  stepTimer = setInterval(() => {
    idx = (idx + 1) % SCAN_STEPS.length;
    $('js-scan-step').textContent = SCAN_STEPS[idx];
  }, 1400);
}

// ─── Button handlers ─────────────────────────────────────────
$('js-scan-btn').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;
  showState('scanning');
  animateScanSteps();
  chrome.runtime.sendMessage({ type: 'TRIGGER_SCAN', tabId: tab.id });
});

$('js-start-scan-btn').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;
  showState('scanning');
  animateScanSteps();
  chrome.runtime.sendMessage({ type: 'TRIGGER_SCAN', tabId: tab.id });
});

$('js-retry-btn').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;
  showState('scanning');
  animateScanSteps();
  chrome.runtime.sendMessage({ type: 'TRIGGER_SCAN', tabId: tab.id });
});

// Toggle overlay visibility
let overlayVisible = true;
$('js-toggle-overlay').addEventListener('click', async () => {
  overlayVisible = !overlayVisible;
  $('js-toggle-overlay').classList.toggle('active', overlayVisible);
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: (visible) => {
      const container = document.getElementById('dpd-overlay-container');
      if (container) container.style.opacity = visible ? '1' : '0';
    },
    args: [overlayVisible]
  }).catch(() => {});
});

// Auto-scan toggle
$('js-auto-scan').addEventListener('change', (e) => {
  chrome.runtime.sendMessage({ type: 'TOGGLE_AUTO_SCAN', enabled: e.target.checked });
});

// ─── Utility ─────────────────────────────────────────────────
function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ─── Boot ────────────────────────────────────────────────────
init().catch(err => {
  showError('Extension initialization error: ' + err.message);
});
