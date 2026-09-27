// Thin API client for the FastAPI backend.
// In dev, requests go to /api/* and Vite proxies them to the backend.
// In production, set VITE_API_BASE to the backend origin at build time.

const BASE = import.meta.env.VITE_API_BASE || '/api'

// Free backend hosts sleep when idle and take ~50s to wake; cap the wait so a
// slow/unreachable backend surfaces a clear message instead of hanging forever.
const DEFAULT_TIMEOUT_MS = 70000

async function request(path, options = {}) {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, ...fetchOpts } = options
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  let resp
  try {
    resp = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      ...fetchOpts,
    })
  } catch (e) {
    // Aborted (timeout) or network-level failure: backend unreachable/asleep.
    if (e.name === 'AbortError') {
      throw new Error(
        'The server is taking too long to respond. Free hosting sleeps when idle — wait a moment and try again.'
      )
    }
    throw new Error('Cannot reach the server. Is the backend running?')
  } finally {
    clearTimeout(timer)
  }

  let data = null
  try {
    data = await resp.json()
  } catch {
    data = null
  }

  if (!resp.ok) {
    const detail =
      (data && (data.detail || data.message)) ||
      `Request failed (${resp.status}).`
    // FastAPI validation errors come back as arrays; make them readable.
    if (Array.isArray(detail)) {
      throw new Error(detail.map((d) => d.msg).join('; '))
    }
    throw new Error(typeof detail === 'string' ? detail : 'Request failed.')
  }
  return data
}

export function predict(text) {
  return request('/predict', {
    method: 'POST',
    body: JSON.stringify({ text }),
  })
}

export function extract(text) {
  return request('/extract', {
    method: 'POST',
    body: JSON.stringify({ text }),
  })
}

export function search(query, entityTypes = [], limit = 10) {
  return request('/search', {
    method: 'POST',
    body: JSON.stringify({ query, entity_types: entityTypes, limit }),
  })
}

export function ask(context, question) {
  return request('/assistant', {
    method: 'POST',
    body: JSON.stringify({ context, question }),
  })
}

export function getDatasetInfo() {
  return request('/dataset-info', { method: 'GET' })
}

export function getModelInfo() {
  return request('/model-info', { method: 'GET' })
}

export function getHealth() {
  // Short timeout: this polls every 15s, so a hung request must not pile up.
  return request('/health', { method: 'GET', timeoutMs: 8000 })
}
