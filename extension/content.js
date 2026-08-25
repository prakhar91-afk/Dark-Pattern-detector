// ============================================================
// content.js — Dark Pattern Detector Content Script
// ============================================================
// Injected into every webpage. Responsibilities:
//   1. Extract all user-visible text with DOM selectors + rects
//   2. Flag structural dark patterns (countdowns, pre-checks, etc.)
//   3. Receive APPLY_HIGHLIGHTS from background.js → paint overlays
//   4. Manage overlay lifecycle (clear on re-scan, remove on nav)
// ============================================================

(function () {
  'use strict';

  // Prevent double-injection
  if (window.__dpd_injected__) return;
  window.__dpd_injected__ = true;

  // ─── Constants ──────────────────────────────────────────────
  const OVERLAY_CLASS = 'dpd-overlay';
  const BADGE_CLASS   = 'dpd-badge';

  const CATEGORY_COLORS = {
    urgency:         { border: '#ef4444', bg: 'rgba(239,68,68,0.12)',  label: 'Urgency' },
    optin:           { border: '#f97316', bg: 'rgba(249,115,22,0.12)', label: 'Pre-checked Opt-in' },
    confirm_shaming: { border: '#a855f7', bg: 'rgba(168,85,247,0.12)', label: 'Confirm-shaming' },
    hidden_flows:    { border: '#06b6d4', bg: 'rgba(6,182,212,0.12)',  label: 'Hidden Flow' },
    disguised_ads:   { border: '#eab308', bg: 'rgba(234,179,8,0.12)',  label: 'Disguised Ad' },
    general:         { border: '#64748b', bg: 'rgba(100,116,139,0.12)', label: 'Dark Pattern' }
  };

  // ─── Inject overlay stylesheet ───────────────────────────────
  const style = document.createElement('style');
  style.id = 'dpd-styles';
  style.textContent = `
    .dpd-overlay {
      position: absolute !important;
      pointer-events: none !important;
      z-index: 2147483647 !important;
      border-radius: 4px !important;
      transition: opacity 0.3s ease !important;
      box-sizing: border-box !important;
    }
    .dpd-badge {
      position: absolute !important;
      z-index: 2147483647 !important;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
      font-size: 10px !important;
      font-weight: 700 !important;
      letter-spacing: 0.3px !important;
      line-height: 1 !important;
      padding: 2px 5px !important;
      border-radius: 3px !important;
      color: #fff !important;
      white-space: nowrap !important;
      pointer-events: none !important;
      box-shadow: 0 1px 4px rgba(0,0,0,0.4) !important;
      text-transform: uppercase !important;
    }
    .dpd-pulse {
      animation: dpd-pulse-anim 2s ease-in-out infinite !important;
    }
    @keyframes dpd-pulse-anim {
      0%, 100% { opacity: 1; }
      50%       { opacity: 0.6; }
    }
  `;
  (document.head || document.documentElement).appendChild(style);

  // ─── Overlay container ───────────────────────────────────────
  const overlayContainer = document.createElement('div');
  overlayContainer.id = 'dpd-overlay-container';
  overlayContainer.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 2147483646; overflow: hidden;
  `;
  document.documentElement.appendChild(overlayContainer);

  // ─── Message listener ────────────────────────────────────────
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    switch (message.type) {
      case 'EXTRACT_DOM':
        extractAndSend(sendResponse);
        return true;

      case 'APPLY_HIGHLIGHTS':
        clearOverlays();
        if (message.detections?.length) {
          applyHighlights(message.detections);
        }
        sendResponse({ ok: true });
        break;

      case 'CLEAR_HIGHLIGHTS':
        clearOverlays();
        sendResponse({ ok: true });
        break;

      case 'PING':
        sendResponse({ alive: true });
        break;
    }
  });

  // ─── DOM Extraction ──────────────────────────────────────────
  function extractAndSend(sendResponse) {
    const texts    = extractTextNodes();
    const elements = extractStructuralElements();

    const payload = { texts, elements };

    // Send to background.js via message
    chrome.runtime.sendMessage(
      { type: 'DOM_READY', payload },
      (response) => {
        if (sendResponse) sendResponse({ ok: true });
      }
    );
  }

  // Collect all visible text blocks with their DOM selector + bounding rect
  function extractTextNodes() {
    const results = [];
    const seen = new Set();

    const candidates = document.querySelectorAll(
      'button, a, label, span, p, h1, h2, h3, h4, h5, h6, ' +
      'div[class*="btn"], div[class*="cta"], div[class*="offer"], ' +
      'div[class*="timer"], div[class*="countdown"], div[class*="urgency"], ' +
      '[role="button"], [role="link"], [role="checkbox"], ' +
      'input[type="submit"], input[type="button"], ' +
      '.modal, .popup, .overlay, .banner, .alert, .notification, ' +
      '[class*="modal"], [class*="popup"], [class*="banner"], ' +
      '[class*="cookie"], [class*="consent"], [class*="gdpr"]'
    );

    for (const el of candidates) {
      const text = (el.innerText || el.textContent || '').trim();
      if (!text || text.length < 3 || text.length > 800) continue;
      if (seen.has(text)) continue;
      seen.add(text);

      const rect = el.getBoundingClientRect();
      if (rect.width === 0 && rect.height === 0) continue;
      if (!isVisible(el)) continue;

      results.push({
        text,
        selector: getSelector(el),
        rect: {
          x: Math.round(rect.left + window.scrollX),
          y: Math.round(rect.top  + window.scrollY),
          w: Math.round(rect.width),
          h: Math.round(rect.height)
        },
        tag: el.tagName.toLowerCase(),
        classes: Array.from(el.classList).slice(0, 8).join(' ')
      });

      if (results.length >= 200) break; // cap to prevent huge payloads
    }

    return results;
  }

  // Detect structural dark pattern signals directly in DOM
  function extractStructuralElements() {
    const elements = [];

    // 1. Countdown / urgency timers
    const timerSelectors = [
      '[data-countdown]', '[data-timer]', '[class*="countdown"]',
      '[class*="timer"]', '[class*="urgency"]', '[id*="countdown"]',
      '[id*="timer"]', 'time[datetime]'
    ];
    for (const sel of timerSelectors) {
      document.querySelectorAll(sel).forEach(el => {
        const text = (el.innerText || el.textContent || '').trim();
        const hasTimePattern = /\d{1,2}:\d{2}(:\d{2})?/.test(text) ||
                               /\d+\s*(hour|min|sec|hr)/i.test(text);
        if (hasTimePattern || el.hasAttribute('data-countdown')) {
          elements.push(makeElement(el, 'urgency', 'Countdown timer detected'));
        }
      });
    }

    // 2. Pre-checked opt-in checkboxes
    document.querySelectorAll('input[type="checkbox"]').forEach(el => {
      if (el.checked || el.hasAttribute('checked')) {
        const label = getCheckboxLabel(el);
        const isOptin = /subscribe|newsletter|email|offer|deal|promotion|marketing|update|notify/i.test(label);
        if (isOptin) {
          elements.push(makeElement(el, 'optin', `Pre-checked opt-in: "${label.slice(0, 60)}"`));
        }
      }
    });

    // 3. Confirm-shaming language in dismiss buttons
    const shamingPatterns = [
      /no\s*thanks[,.]?\s*i\s*(hate|don'?t|don't)/i,
      /no\s*thanks[,.]?\s*(i'?m|i am)\s*(stupid|broke|cheap|loser|fine)/i,
      /no[,.]?\s*i\s*don'?t\s*want/i,
      /decline\s*(this\s*)?(amazing|great|exclusive|special|free)/i,
      /i\s*(don'?t|do not)\s*want\s*to\s*(save|improve|get|learn)/i
    ];
    document.querySelectorAll('button, a, [role="button"]').forEach(el => {
      const text = (el.innerText || el.textContent || '').trim();
      for (const pattern of shamingPatterns) {
        if (pattern.test(text)) {
          elements.push(makeElement(el, 'confirm_shaming', `Confirm-shaming: "${text.slice(0, 80)}"`));
          break;
        }
      }
    });

    // 4. Hidden / camouflaged unsubscribe / cancel links
    document.querySelectorAll('a, span, button').forEach(el => {
      const text = (el.innerText || el.textContent || '').trim().toLowerCase();
      if (!/unsubscribe|opt.?out|cancel\s*(anytime|subscription)|remove me/i.test(text)) return;

      const style = window.getComputedStyle(el);
      const fontSize = parseFloat(style.fontSize);
      const opacity  = parseFloat(style.opacity);
      const color    = style.color;
      const bgColor  = style.backgroundColor;

      const isHidden = fontSize < 11 || opacity < 0.4 ||
                       isLowContrast(color, bgColor) ||
                       style.display === 'none' ||
                       style.visibility === 'hidden';

      if (isHidden) {
        elements.push(makeElement(el, 'hidden_flows', 'Hidden unsubscribe / cancel link'));
      }
    });

    // 5. Disguised advertisements
    const adSelectors = [
      '[data-ad]', '[data-ad-slot]', '[aria-label*="sponsor" i]',
      '[aria-label*="advertisement" i]', '[class*="native-ad"]',
      '[class*="sponsored"]', '[id*="adsense"]', '[class*="advert"]',
      'ins.adsbygoogle', '[data-google-query-id]'
    ];
    for (const sel of adSelectors) {
      document.querySelectorAll(sel).forEach(el => {
        const rect = el.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) {
          elements.push(makeElement(el, 'disguised_ads', 'Potential disguised advertisement'));
        }
      });
    }

    return elements;
  }

  // ─── Highlight rendering ────────────────────────────────────
  function applyHighlights(detections) {
    detections.forEach((detection, idx) => {
      const cat   = CATEGORY_COLORS[detection.type] || CATEGORY_COLORS.general;
      const rect  = resolveRect(detection);
      if (!rect) return;

      // Border overlay
      const overlay = document.createElement('div');
      overlay.className = `${OVERLAY_CLASS} dpd-pulse`;
      overlay.dataset.dpdIdx = idx;
      overlay.style.cssText = `
        left:   ${rect.x}px;
        top:    ${rect.y}px;
        width:  ${rect.w}px;
        height: ${rect.h}px;
        border: 2px solid ${cat.border};
        background: ${cat.bg};
      `;

      // Label badge
      const badge = document.createElement('div');
      badge.className = BADGE_CLASS;
      badge.style.cssText = `
        left: ${rect.x}px;
        top:  ${Math.max(0, rect.y - 18)}px;
        background: ${cat.border};
      `;
      badge.textContent = cat.label;

      overlayContainer.appendChild(overlay);
      overlayContainer.appendChild(badge);
    });
  }

  function clearOverlays() {
    overlayContainer.innerHTML = '';
  }

  // ─── Helpers ─────────────────────────────────────────────────
  function makeElement(el, type, description) {
    const rect = el.getBoundingClientRect();
    return {
      type,
      description,
      selector: getSelector(el),
      rect: {
        x: Math.round(rect.left + window.scrollX),
        y: Math.round(rect.top  + window.scrollY),
        w: Math.round(rect.width),
        h: Math.round(rect.height)
      },
      text: (el.innerText || el.textContent || '').trim().slice(0, 120)
    };
  }

  function resolveRect(detection) {
    // Try selector first, then fall back to stored rect
    if (detection.selector) {
      try {
        const el = document.querySelector(detection.selector);
        if (el) {
          const r = el.getBoundingClientRect();
          return {
            x: Math.round(r.left + window.scrollX),
            y: Math.round(r.top  + window.scrollY),
            w: Math.round(r.width),
            h: Math.round(r.height)
          };
        }
      } catch {}
    }
    return detection.rect || null;
  }

  function isVisible(el) {
    const style = window.getComputedStyle(el);
    return style.display !== 'none' &&
           style.visibility !== 'hidden' &&
           parseFloat(style.opacity) > 0.1;
  }

  function getCheckboxLabel(checkbox) {
    // Check for linked <label> element
    if (checkbox.id) {
      const label = document.querySelector(`label[for="${checkbox.id}"]`);
      if (label) return (label.innerText || label.textContent || '').trim();
    }
    // Check parent <label>
    const parent = checkbox.closest('label');
    if (parent) return (parent.innerText || parent.textContent || '').trim();
    // Sibling text
    const sibling = checkbox.nextElementSibling;
    if (sibling) return (sibling.innerText || sibling.textContent || '').trim();
    return checkbox.getAttribute('aria-label') || '';
  }

  function getSelector(el) {
    if (el.id) return `#${CSS.escape(el.id)}`;
    const path = [];
    let current = el;
    while (current && current !== document.body) {
      let part = current.tagName.toLowerCase();
      if (current.className && typeof current.className === 'string') {
        const classes = Array.from(current.classList)
          .filter(c => c && !c.includes(':') && c.length < 40)
          .slice(0, 2)
          .map(c => `.${CSS.escape(c)}`)
          .join('');
        part += classes;
      }
      const siblings = current.parentElement
        ? Array.from(current.parentElement.children).filter(c => c.tagName === current.tagName)
        : [];
      if (siblings.length > 1) {
        const idx = siblings.indexOf(current) + 1;
        part += `:nth-of-type(${idx})`;
      }
      path.unshift(part);
      current = current.parentElement;
      if (path.length >= 4) break;
    }
    return path.join(' > ');
  }

  function isLowContrast(fg, bg) {
    // Very rough check — if both are close shades of grey/white
    try {
      const fgVal = parseRGB(fg);
      const bgVal = parseRGB(bg);
      if (!fgVal || !bgVal) return false;
      const diff = Math.abs(getLuminance(fgVal) - getLuminance(bgVal));
      return diff < 0.1;
    } catch { return false; }
  }

  function parseRGB(color) {
    const m = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
    return m ? [+m[1], +m[2], +m[3]] : null;
  }

  function getLuminance([r, g, b]) {
    const toLinear = v => {
      v /= 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * toLinear(r) + 0.7152 * toLinear(g) + 0.0722 * toLinear(b);
  }

  // ─── Auto-trigger on first load ──────────────────────────────
  // Signal background that content script is ready
  chrome.runtime.sendMessage({ type: 'CONTENT_READY', url: location.href })
    .catch(() => {}); // ignore if SW not yet awake

})();
