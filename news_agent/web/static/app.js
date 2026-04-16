// ── Expand / collapse ────────────────────────────────────────────────────────

function toggleExpand(card) {
  const expanded = card.querySelector('.card-expanded');
  const isOpen = expanded.style.display !== 'none';

  if (isOpen) {
    // Closing — log how long it was open as read time
    const openedAt = parseFloat(card.dataset.openedAt || '0');
    if (openedAt) {
      const secs = (Date.now() - openedAt) / 1000;
      if (secs >= 5) {
        logInteraction(card.dataset.id, 'read', secs);
      }
    }
    expanded.style.display = 'none';
    delete card.dataset.openedAt;
  } else {
    expanded.style.display = 'block';
    card.dataset.openedAt = Date.now();
    logInteraction(card.dataset.id, 'click');
  }
}

// ── Vote (thumbs up / down) ───────────────────────────────────────────────────

function toggleVote(event, itemId, direction) {
  event.stopPropagation(); // don't expand card
  const btn = event.currentTarget;
  const card = btn.closest('.news-card');
  const isActive = btn.classList.contains('on');

  if (direction === 'up') {
    const wasOn = isActive;
    // toggle this button; clear the down button
    card.querySelectorAll('.vote-btn').forEach(b => b.classList.remove('on'));
    if (!wasOn) btn.classList.add('on');
    card.classList.toggle('liked', !wasOn);
    card.classList.remove('disliked');
    logInteraction(itemId, wasOn ? 'unstar' : 'star');
  } else {
    // dislike: clear like, toggle disliked
    card.querySelectorAll('.vote-btn').forEach(b => b.classList.remove('on'));
    if (!isActive) btn.classList.add('on');
    card.classList.remove('liked');
    card.classList.toggle('disliked', !isActive);
    logInteraction(itemId, isActive ? 'undislike' : 'dislike');
  }
}

// ── Click on external link ────────────────────────────────────────────────────

function logClick(itemId) {
  logInteraction(itemId, 'click');
}

// ── API ───────────────────────────────────────────────────────────────────────

function logInteraction(itemId, action, readSeconds) {
  const payload = { item_id: itemId, action };
  if (readSeconds !== undefined) payload.read_seconds = readSeconds;

  // Fire-and-forget; use sendBeacon for unload-time events
  if (navigator.sendBeacon && action === 'read') {
    const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
    navigator.sendBeacon('/api/interaction', blob);
  } else {
    fetch('/api/interaction', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      keepalive: true,
    }).catch(() => {}); // best-effort
  }
}

// ── Local time ────────────────────────────────────────────────────────────────

const _headerDate = document.getElementById('header-date');
if (_headerDate) {
  _headerDate.textContent = new Date().toLocaleDateString(undefined, {
    year: 'numeric', month: 'long', day: 'numeric',
  });
}

document.querySelectorAll('.pub-time[data-utc]').forEach(el => {
  const iso = el.dataset.utc;
  if (!iso) return;
  const d = new Date(iso.endsWith('Z') ? iso : iso + 'Z');
  el.textContent = d.toLocaleString(undefined, {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
});

// ── Refresh panel ────────────────────────────────────────────────────────────

async function refreshPanel(panelIdx, topic) {
  const panelEl = document.getElementById(`panel-${panelIdx}`);
  if (!panelEl) return;

  const btn = panelEl.querySelector('.refresh-btn');
  if (btn) { btn.disabled = true; btn.textContent = '⟳'; }

  try {
    // Kick off a fresh API fetch, bypassing the cooldown
    await fetch(`/api/fetch?keyword=${encodeURIComponent(topic)}&force=true`, { method: 'POST' })
      .catch(() => {});

    // Wait until the fetch is no longer running (max ~60s)
    for (let i = 0; i < 40; i++) {
      await new Promise(r => setTimeout(r, 1500));
      const { running } = await fetch(
        `/api/fetch/status?topic=${encodeURIComponent(topic)}&hours=${_hours}`
      ).then(r => r.json()).catch(() => ({ running: false }));
      if (!running) break;
    }

    // Always re-render the panel with whatever is now in the DB
    const stale = panelEl.querySelector('.digest-summary');
    if (stale) stale.remove();
    await updatePanel(panelEl, topic);
    pollDigest(panelEl, topic);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = '↻'; }
  }
}

// ── Podcast ───────────────────────────────────────────────────────────────────

function requestPodcast(topic, btn) {
  btn.disabled = true;
  btn.textContent = '⏳ Generating…';

  const hours = new URLSearchParams(window.location.search).get('hours') || '24';
  fetch(`/api/podcast/${topic}?hours=${hours}`, { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      if (data.reason === 'already generating' || data.started) {
        pollPodcast(topic, btn);
      } else {
        btn.disabled = false;
        btn.textContent = '🎙 Podcast';
        alert(data.reason || 'Could not start podcast generation.');
      }
    })
    .catch(() => { btn.disabled = false; btn.textContent = '🎙 Podcast'; });
}

function pollPodcast(topic, btn) {
  const hours = new URLSearchParams(window.location.search).get('hours') || '24';
  fetch(`/api/podcast/${topic}/status?hours=${hours}`)
    .then(r => r.json())
    .then(status => {
      if (status.ready) {
        const player = document.getElementById(`podcast-player-${topic}`);
        const audio = player.querySelector('audio');
        audio.src = status.url;
        player.style.display = 'block';
        btn.textContent = '▶ Play again';
        btn.disabled = false;
        btn.onclick = () => { player.style.display = 'block'; audio.play(); };
      } else if (status.error) {
        btn.disabled = false;
        btn.textContent = '🎙 Podcast';
        alert('Podcast failed: ' + status.error);
      } else if (status.generating) {
        setTimeout(() => pollPodcast(topic, btn), 3000);
      } else {
        btn.disabled = false;
        btn.textContent = '🎙 Podcast';
      }
    })
    .catch(() => setTimeout(() => pollPodcast(topic, btn), 3000));
}

// On load, check if today's podcast already exists for the active panel topics
document.addEventListener('DOMContentLoaded', () => {
  const hours = new URLSearchParams(window.location.search).get('hours') || '24';
  const urlParams = new URLSearchParams(window.location.search);
  [urlParams.get('topic1'), urlParams.get('topic2')].filter(Boolean).forEach(topic => {
    fetch(`/api/podcast/${encodeURIComponent(topic)}/status?hours=${hours}`)
      .then(r => r.json())
      .then(status => {
        if (status.ready) {
          const player = document.getElementById(`podcast-player-${topic}`);
          if (!player) return;
          const audio = player.querySelector('audio');
          audio.src = status.url;
          player.style.display = 'block';
          const btn = player.previousElementSibling?.querySelector('.podcast-btn');
          if (btn) { btn.textContent = '▶ Play again'; }
        }
      }).catch(() => {});
  });
});

// ── Digest polling ───────────────────────────────────────────────────────────

// ── Digest streaming ─────────────────────────────────────────────────────────

function _boldMd(text) {
  return text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
}

function _renderDigestText(container, text) {
  /** Parse streamed plain-text digest and render headline + bullets. */
  const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
  if (!lines.length) return;
  const headline = lines[0];
  const bullets = lines.slice(1);
  let html = `<div class="digest-label">Summary</div>`;
  if (headline) html += `<p class="digest-headline">${_boldMd(headline)}</p>`;
  if (bullets.length) {
    html += `<ul class="digest-bullets">` +
      bullets.map(b => `<li>${_boldMd(b)}</li>`).join('') +
      `</ul>`;
  }
  container.innerHTML = html;
}

function streamDigest(panelEl, topic) {
  if (!topic) return;
  if (panelEl.querySelector('.digest-summary')) return; // already shown

  const hours = new URLSearchParams(window.location.search).get('hours') || '24';
  const url = `/api/digest-stream/${encodeURIComponent(topic)}?hours=${hours}`;

  // Insert placeholder immediately
  const container = document.createElement('div');
  container.className = 'digest-summary';
  container.innerHTML = '<div class="digest-label">Summary</div><p class="digest-streaming">&#8203;</p>';
  const newsList = panelEl.querySelector('.news-list');
  if (!newsList) return;
  newsList.insertAdjacentElement('beforebegin', container);

  const streamEl = container.querySelector('.digest-streaming');
  let accumulated = '';

  const source = new EventSource(url);

  source.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.done) {
      source.close();
      if (accumulated.trim()) {
        _renderDigestText(container, accumulated);
      } else {
        container.remove();
      }
      return;
    }
    if (data.t) {
      accumulated += data.t;
      // Show raw accumulating text until we have at least one full line
      if (streamEl) streamEl.textContent = accumulated;
      // Once we have at least 2 lines, switch to formatted rendering
      const lines = accumulated.split('\n').filter(l => l.trim());
      if (lines.length >= 2) _renderDigestText(container, accumulated);
    }
  };

  source.onerror = () => {
    source.close();
    if (!accumulated.trim()) container.remove();
  };
}

// Alias for compatibility with places that call injectDigest or pollDigest
function injectDigest(panelEl, topic) { streamDigest(panelEl, topic); return Promise.resolve(false); }
function pollDigest(panelEl, topic) { streamDigest(panelEl, topic); }

// Re-check for digests when the tab becomes visible again
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState !== 'visible') return;
  const urlParams = new URLSearchParams(window.location.search);
  [1, 2].forEach(idx => {
    const panelEl = document.getElementById(`panel-${idx}`);
    if (!panelEl) return;
    const topic = urlParams.get(`topic${idx}`) || '';
    if (topic && !panelEl.querySelector('.digest-summary')) {
      streamDigest(panelEl, topic);
    }
  });
});

// ── Auto-fetch ────────────────────────────────────────────────────────────────
// Each panel polls independently; when results arrive, only that panel updates.

// Shared helpers — defined at module scope so refreshPanel (global) can reach them
const _urlParams = new URLSearchParams(window.location.search);
const _hours = _urlParams.get('hours') || '24';

function _formatPubTime(el) {
  const iso = el.dataset.utc;
  if (!iso) return;
  const d = new Date(iso.endsWith('Z') ? iso : iso + 'Z');
  el.textContent = d.toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
}

async function updatePanel(panelEl, topic) {
  const langs = _urlParams.get('langs') || '';
  const langParam = langs ? `&langs=${encodeURIComponent(langs)}` : '';
  const [panelResp, digestResp] = await Promise.all([
    fetch(`/api/panel?topic=${encodeURIComponent(topic)}&hours=${_hours}${langParam}`),
    topic ? fetch(`/api/digest-fragment?topic=${encodeURIComponent(topic)}&hours=${_hours}`) : Promise.resolve(null),
  ]);

  const html = await panelResp.text();
  const newsList = panelEl.querySelector('.news-list');
  if (newsList) {
    newsList.innerHTML = html;
    const count = newsList.querySelectorAll('.news-card').length;
    newsList.dataset.count = count;
    const countBadge = panelEl.querySelector('.count');
    if (countBadge) countBadge.textContent = `${count} items`;
    newsList.querySelectorAll('.pub-time[data-utc]').forEach(_formatPubTime);
  }

  if (digestResp) {
    const digestHtml = await digestResp.text();
    const existing = panelEl.querySelector('.digest-summary');
    if (digestHtml.trim()) {
      // Only update the digest when we have real content — never clear a live
      // streaming digest by replacing it with an empty fragment response.
      if (existing) {
        existing.outerHTML = digestHtml;
      } else if (newsList) {
        newsList.insertAdjacentHTML('beforebegin', digestHtml);
      }
    }
    // If digestHtml is empty: keep whatever is already shown (may be mid-stream)
  }
}

(function () {
  const urlParams = _urlParams;
  const hours = _hours;

  async function pollTopic(panelEl, topic) {
    const newsList = panelEl.querySelector('.news-list');
    const initial = parseInt(newsList?.dataset.count || '0', 10);
    const emptyEl = panelEl.querySelector('.auto-fetch-empty');
    let noChangeRounds = 0;

    while (true) {
      await new Promise(r => setTimeout(r, 1000));
      const res = await fetch(`/api/fetch/status?topic=${encodeURIComponent(topic)}&hours=${hours}`)
        .then(r => r.json())
        .catch(() => ({ running: false, count: initial }));

      if (res.count > initial) {
        await updatePanel(panelEl, topic);
        // Only regenerate the digest if there isn't one already. Removing and
        // regenerating on every incremental item increase causes left-panel
        // digest churn when only the right panel topic changed.
        if (!panelEl.querySelector('.digest-summary')) {
          pollDigest(panelEl, topic);
        }
        return;
      }

      if (!res.running) {
        noChangeRounds++;
        if (noChangeRounds >= 2) {
          if (emptyEl) {
            if (emptyEl.classList.contains('fetch-more')) emptyEl.remove();
            else emptyEl.innerHTML = '<p style="color:var(--text-muted)">No new items found. Try a wider time range.</p>';
          }
          if (!panelEl.querySelector('.digest-summary')) pollDigest(panelEl, topic);
          return;
        }
      }
    }
  }

  [1, 2].forEach(idx => {
    const panelEl = document.getElementById(`panel-${idx}`);
    if (!panelEl) return;
    const topic = urlParams.get(`topic${idx}`) || '';

    if (!topic) {
      if (panelEl.querySelector('.auto-fetch-empty')) updatePanel(panelEl, '');
      return;
    }

    // Always kick off a background fetch so the panel gets fresh results,
    // even if items are already shown from a previous fetch cycle.
    fetch(`/api/fetch?keyword=${encodeURIComponent(topic)}`, { method: 'POST' }).catch(() => {});

    if (panelEl.querySelector('.auto-fetch-empty')) {
      // No items yet — poll until they arrive (also starts digest when done)
      pollTopic(panelEl, topic);
    } else {
      // Panel already has items — show digest if missing, then watch for new items
      if (!panelEl.querySelector('.digest-summary')) pollDigest(panelEl, topic);
      // Light background poll: refresh panel if more items arrive.
      // Keeps polling while the fetch is running; stops after 3 quiet rounds once done.
      (async () => {
        const initial = parseInt(panelEl.querySelector('.news-list')?.dataset.count || '0', 10);
        let noChange = 0;
        while (noChange < 3) {
          await new Promise(r => setTimeout(r, 2000));
          const res = await fetch(`/api/fetch/status?topic=${encodeURIComponent(topic)}&hours=${hours}`)
            .then(r => r.json()).catch(() => ({ running: false, count: initial }));
          if (res.count > initial) { await updatePanel(panelEl, topic); return; }
          if (!res.running) noChange++;
          else noChange = 0; // reset — keep waiting while fetch is still running
        }
      })();
    }
  });
})();

// ── Send read time when user navigates away ───────────────────────────────────

window.addEventListener('beforeunload', () => {
  document.querySelectorAll('.news-card[data-opened-at]').forEach(card => {
    const secs = (Date.now() - parseFloat(card.dataset.openedAt)) / 1000;
    if (secs >= 5) {
      logInteraction(card.dataset.id, 'read', secs);
    }
  });
});
