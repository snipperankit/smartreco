/*
 * SmartReco tracker — non-blocking behavioral event capture.
 *
 * Design:
 *  - Events buffer in memory + localStorage (survives navigation).
 *  - Flush is triggered by size (batch >= FLUSH_SIZE), a timer, or page-hide.
 *  - Page-hide uses navigator.sendBeacon so the request never blocks unload.
 *  - High-frequency events (scroll/hover) are throttled before buffering.
 *  - Time-on-page is measured and attached to the view event on exit.
 */
(function () {
  const FLUSH_SIZE = 3;
  const FLUSH_INTERVAL_MS = 3000;
  const BUFFER_KEY = "sr_event_buffer";
  const BEACON_URL = "/api/events/track-beacon";
  const FETCH_URL = "/api/events/track";

  const meta = document.querySelector('meta[name="sr-context"]');
  const ctx = meta ? JSON.parse(meta.content) : {};
  const pageEnteredAt = Date.now();

  // Session ID: persists for the tab's lifetime (EVT-10).
  const SESSION_KEY = "sr_session_id";
  function getSessionId() {
    let sid = sessionStorage.getItem(SESSION_KEY);
    if (!sid) { sid = crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2); sessionStorage.setItem(SESSION_KEY, sid); }
    return sid;
  }
  const sessionId = getSessionId();

  function loadBuffer() {
    try {
      return JSON.parse(localStorage.getItem(BUFFER_KEY) || "[]");
    } catch (_) {
      return [];
    }
  }
  function saveBuffer(b) {
    try {
      localStorage.setItem(BUFFER_KEY, JSON.stringify(b));
    } catch (_) {}
  }

  let buffer = loadBuffer();

  function track(type, payload) {
    buffer.push({ type, payload: payload || {}, timestamp: Date.now(), session_id: sessionId });
    saveBuffer(buffer);
    if (buffer.length >= FLUSH_SIZE) flush(false);
  }

  function flush(useBeacon) {
    if (!buffer.length) return;
    const body = JSON.stringify({ events: buffer });
    const sending = buffer;
    buffer = [];
    saveBuffer(buffer);

    if (useBeacon && navigator.sendBeacon) {
      navigator.sendBeacon(BEACON_URL, body); // fire-and-forget, non-blocking
      return;
    }
    // Async fetch; on failure, put events back so nothing is lost.
    fetch(FETCH_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      keepalive: true,
    }).catch(() => {
      buffer = sending.concat(buffer);
      saveBuffer(buffer);
    });
  }

  // --- Throttle helper for high-frequency events ---------------------------
  function throttle(fn, ms) {
    let last = 0;
    return function (...args) {
      const now = Date.now();
      if (now - last >= ms) {
        last = now;
        fn.apply(this, args);
      }
    };
  }

  // --- Auto-capture --------------------------------------------------------
  // Page / product view on load.
  if (ctx.event === "product_view") {
    track("view", { product_id: ctx.product_id, category: ctx.category });
  } else {
    track("view", { page: location.pathname });
  }

  // Any element with data-track fires a click event.
  document.addEventListener("click", function (e) {
    const el = e.target.closest("[data-track]");
    if (!el) return;
    track("click", {
      product_id: el.dataset.productId ? Number(el.dataset.productId) : undefined,
      category: el.dataset.category,
      label: el.dataset.track,
    });
  });

  // Time-on-page attached at exit for the current product.
  function trackDwell() {
    const seconds = Math.round((Date.now() - pageEnteredAt) / 1000);
    if (ctx.event === "product_view" && seconds > 1) {
      track("view", {
        product_id: ctx.product_id,
        category: ctx.category,
        time_spent: seconds,
      });
    }
  }

  // Periodic flush.
  setInterval(() => flush(false), FLUSH_INTERVAL_MS);

  // Flush on page hide via beacon (covers tab close / navigation).
  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "hidden") {
      trackDwell();
      flush(true);
    }
  });
  window.addEventListener("pagehide", function () {
    trackDwell();
    flush(true);
  });

  // --- Scroll depth tracking (EVT-05) -----------------------------------
  let maxScroll = 0;
  const scrollHandler = throttle(function () {
    const docH = document.documentElement.scrollHeight - window.innerHeight;
    if (docH <= 0) return;
    const pct = Math.min(100, Math.round((window.scrollY / docH) * 100));
    if (pct > maxScroll) maxScroll = pct;
  }, 1000);
  window.addEventListener("scroll", scrollHandler, { passive: true });
  // Flush scroll depth on page hide alongside dwell time.
  const _origVisibility = function () {
    if (document.visibilityState === "hidden" && maxScroll > 10) {
      track("scroll_depth", { depth_pct: maxScroll, product_id: ctx.product_id, category: ctx.category });
    }
  };
  document.addEventListener("visibilitychange", _origVisibility);

  // Expose a manual API for search boxes etc.
  window.SmartReco = {
    trackSearch: function (query) {
      track("search", { query });
      flush(false); // searches are high-signal — flush promptly
    },
    track,
  };
})();
