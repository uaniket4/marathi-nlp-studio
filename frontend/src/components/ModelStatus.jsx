import { useEffect, useState } from 'react'
import { getHealth } from '../services/api'

// Live model status indicator for the sidebar. Polls /health so the dot
// reflects whether the backend model is actually loaded.
export default function ModelStatus() {
  const [state, setState] = useState({ status: 'loading', detail: '' })

  useEffect(() => {
    let alive = true
    async function check() {
      try {
        const h = await getHealth()
        if (!alive) return
        setState({
          status: h.model_loaded ? 'ok' : 'degraded',
          detail: h.detail || '',
        })
      } catch (e) {
        if (!alive) return
        setState({ status: 'down', detail: e.message || '' })
      }
    }
    check()
    const id = setInterval(check, 15000)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [])

  const MAP = {
    loading: { dot: 'bg-gray-400', text: 'Checking…', ping: false },
    ok: { dot: 'bg-emerald-500', text: 'Model ready', ping: true },
    degraded: { dot: 'bg-amber-500', text: 'Model not loaded', ping: false },
    down: { dot: 'bg-red-500', text: 'Backend offline', ping: false },
  }
  const s = MAP[state.status] || MAP.loading

  return (
    <div
      className="flex items-center gap-2 rounded-md border border-gray-200 px-3 py-2 text-xs dark:border-gray-800"
      title={state.detail || s.text}
    >
      <span className="relative flex h-2.5 w-2.5">
        {s.ping && (
          <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${s.dot}`} />
        )}
        <span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${s.dot}`} />
      </span>
      <span className="font-medium text-gray-600 dark:text-gray-300">{s.text}</span>
    </div>
  )
}
