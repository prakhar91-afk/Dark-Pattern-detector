// ============================================================
// background.js — Dark Pattern Detector Service Worker (MV3)
// ============================================================
// Responsibilities:
//   1. Listen for scan requests from content.js and popup.js
//   2. Capture a screenshot of the current tab
//   3. POST DOM text + screenshot to FastAPI backend
//   4. Store results and notify content.js + popup.js
// ============================================================

const API_BASE = 'http://localhost:8000';

// ─── State ──────────────────────────────────────────────────
const tabResults = new Map();   // tabId → latest scan result
const scanInProgress = new Set(); // tabIds currently scanning

// ─── Startup ────────────────────────────────────────────────
chrome.runtime.onInstalled.addListener(() => {
  console.log('[DPD] Dark Pattern Detector installed');
  chrome.storage.local.set({ autoScan: true, backendUrl: API_BASE });

  // Context menu for manual scan
  chrome.contextMenus.create({
    id: 'dpd-scan',
    title: '🔍 Scan this page for dark patterns',
    contexts: ['page']
  });
});

// ─── Context menu handler ────────────────────────────────────
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'dpd-scan') {
    triggerScan(tab.id);
  }
});

// ─── Tab navigation: auto-scan on page load ──────────────────
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status !== 'complete') return;
  if (!tab.url || tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://')) return;

  const { autoScan } = await chrome.storage.local.get('autoScan');
  if (autoScan) {
    // Small delay to let the page finish rendering
    setTimeout(() => triggerScan(tabId), 1500);
  }
});

// ─── Message listener ────────────────────────────────────────
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const tabId = sender.tab?.id || message.tabId;

  switch (message.type) {
    case 'DOM_READY':
      // Content script has extracted DOM data — proceed with screenshot + API call
      handleDomReady(tabId, message.payload, sendResponse);
      return true; // keep channel open for async response

    case 'GET_RESULTS':
      // Popup requesting latest results
      sendResponse({ result: tabResults.get(message.tabId) || null });
      break;

    case 'TRIGGER_SCAN':
      triggerScan(message.tabId);
      sendResponse({ ok: true });
      break;

    case 'SCAN_STATUS':
      sendResponse({ inProgress: scanInProgress.has(message.tabId) });
      break;

    case 'TOGGLE_AUTO_SCAN':
      chrome.storage.local.set({ autoScan: message.enabled });
      sendResponse({ ok: true });
      break;
  }
});

// ─── Core: trigger a scan on a tab ──────────────────────────
async function triggerScan(tabId) {
  if (scanInProgress.has(tabId)) return;

  try {
    // Ask the content script to extract DOM data
    await chrome.tabs.sendMessage(tabId, { type: 'EXTRACT_DOM' });
  } catch (err) {
    // Content script might not be injected yet; inject it
    try {
      await chrome.scripting.executeScript({
        target: { tabId },
        files: ['content.js']
      });
      await chrome.tabs.sendMessage(tabId, { type: 'EXTRACT_DOM' });
    } catch (e) {
      console.warn('[DPD] Could not inject content script:', e.message);
    }
  }
}

// ─── Core: handle DOM data + capture screenshot + call API ──
async function handleDomReady(tabId, domPayload, sendResponse) {
  if (!tabId) { sendResponse({ error: 'No tab ID' }); return; }
  scanInProgress.add(tabId);

  // Notify popup that scan has started
  broadcastToPopup({ type: 'SCAN_STARTED', tabId });

  try {
    // 1. Capture screenshot
    let screenshotB64 = null;
    try {
      screenshotB64 = await chrome.tabs.captureVisibleTab(null, {
        format: 'jpeg',
        quality: 70  // compress to keep payload size manageable
      });
    } catch (e) {
      console.warn('[DPD] Screenshot capture failed:', e.message);
    }

    // 2. Get current tab URL
    const tab = await chrome.tabs.get(tabId);

    // 3. Build request body
    const requestBody = {
      url: tab.url,
      dom_texts: domPayload.texts || [],
      dom_elements: domPayload.elements || [],
      screenshot_b64: screenshotB64 || null,
      page_title: tab.title || ''
    };

    // 4. Fetch from FastAPI backend
    const { backendUrl } = await chrome.storage.local.get('backendUrl');
    const apiUrl = (backendUrl || API_BASE) + '/analyze';

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(30000)  // 30s timeout
    });

    if (!response.ok) {
      throw new Error(`API error ${response.status}: ${await response.text()}`);
    }

    const result = await response.json();

    // 5. Store results
    tabResults.set(tabId, {
      ...result,
      timestamp: Date.now(),
      url: tab.url
    });

    // 6. Update extension badge
    updateBadge(tabId, result.trust_score);

    // 7. Send highlights back to content script
    chrome.tabs.sendMessage(tabId, {
      type: 'APPLY_HIGHLIGHTS',
      detections: result.detections || []
    }).catch(() => {});

    // 8. Notify popup
    broadcastToPopup({ type: 'SCAN_COMPLETE', tabId, result });

    if (sendResponse) sendResponse({ ok: true, result });

  } catch (err) {
    console.error('[DPD] Scan failed:', err);
    const errorResult = { error: err.message, trust_score: null };
    tabResults.set(tabId, errorResult);
    broadcastToPopup({ type: 'SCAN_ERROR', tabId, error: err.message });
    if (sendResponse) sendResponse({ error: err.message });
  } finally {
    scanInProgress.delete(tabId);
  }
}

// ─── Badge update ────────────────────────────────────────────
function updateBadge(tabId, score) {
  if (score === null || score === undefined) {
    chrome.action.setBadgeText({ text: '?', tabId });
    chrome.action.setBadgeBackgroundColor({ color: '#64748b', tabId });
    return;
  }

  const text = String(score);
  let color;
  if (score >= 75)      color = '#22c55e';  // green
  else if (score >= 50) color = '#f59e0b';  // amber
  else if (score >= 25) color = '#f97316';  // orange
  else                  color = '#ef4444';  // red

  chrome.action.setBadgeText({ text, tabId });
  chrome.action.setBadgeBackgroundColor({ color, tabId });
}

// ─── Broadcast to popup if open ──────────────────────────────
function broadcastToPopup(message) {
  chrome.runtime.sendMessage(message).catch(() => {
    // Popup might be closed — ignore
  });
}

// ─── Alarms: keep SW alive during long operations ────────────
chrome.alarms.create('keepalive', { periodInMinutes: 0.4 });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'keepalive') {
    // No-op — just prevents SW from dying
  }
});
