import { useState } from 'react'
import { extract } from '../services/api'
import PageHeader from '../components/PageHeader'
import Statistics from '../components/Statistics'
import { entityStyle, prettyLabel } from '../services/entityStyles'

const MAX_CHARS = 5000

const EXAMPLE =
  'रतन टाटा यांनी टाटा मोटर्स या कंपनीची स्थापना केली. ही कंपनी मुंबई येथे १९४५ मध्ये सुरू झाली.'

function download(filename, content, type) {
  const blob = new Blob([content], { type })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

function toCSV(entities) {
  const header = 'text,label,start,end,confidence'
  const rows = entities.map((e) => {
    const text = `"${String(e.text).replace(/"/g, '""')}"`
    return [text, e.label, e.start, e.end, e.confidence].join(',')
  })
  return [header, ...rows].join('\n')
}

export default function InformationExtraction() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState(false)

  const trimmed = text.trim()

  async function run() {
    if (!trimmed) {
      setError('Please enter some Marathi text to extract from.')
      return
    }
    setLoading(true)
    setError('')
    setCopied(false)
    try {
      const data = await extract(text)
      setResult(data)
    } catch (e) {
      setError(e.message || 'Extraction failed. Please try again.')
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  const grouped = result?.grouped || {}
  const groupedEntries = Object.entries(grouped)

  async function copyJson() {
    try {
      await navigator.clipboard.writeText(JSON.stringify(grouped, null, 2))
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      setError('Could not copy to clipboard.')
    }
  }

  return (
    <div>
      <PageHeader
        title="Information Extraction"
        subtitle="Extract structured, typed entity lists from Marathi text — ready to copy or download."
      />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Input */}
        <section className="space-y-3">
          <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">Input text</h2>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value.slice(0, MAX_CHARS))}
            rows={10}
            placeholder="मराठी मजकूर येथे टाका…"
            className="input resize-y text-base leading-relaxed"
          />
          <div className="flex items-center justify-between">
            <button
              type="button"
              onClick={() => {
                setText(EXAMPLE)
                setError('')
              }}
              className="text-xs text-accent hover:underline dark:text-indigo-400"
            >
              Load example
            </button>
            <span className="text-xs text-gray-400 dark:text-gray-500">
              {text.length} / {MAX_CHARS}
            </span>
          </div>
          <div className="flex gap-2">
            <button type="button" onClick={run} disabled={loading || !trimmed} className="btn-primary">
              {loading && (
                <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/40 border-t-white" aria-hidden="true" />
              )}
              {loading ? 'Extracting…' : 'Extract Entities'}
            </button>
          </div>
          {error && (
            <div role="alert" className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300">
              {error}
            </div>
          )}
        </section>

        {/* Output */}
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
              Structured output
            </h2>
            {result && result.entities.length > 0 && (
              <div className="flex gap-2">
                <button type="button" onClick={copyJson} className="btn-secondary !px-2.5 !py-1 text-xs">
                  {copied ? 'Copied!' : 'Copy JSON'}
                </button>
                <button
                  type="button"
                  onClick={() => download('entities.json', JSON.stringify(grouped, null, 2), 'application/json')}
                  className="btn-secondary !px-2.5 !py-1 text-xs"
                >
                  JSON
                </button>
                <button
                  type="button"
                  onClick={() => download('entities.csv', toCSV(result.entities), 'text/csv')}
                  className="btn-secondary !px-2.5 !py-1 text-xs"
                >
                  CSV
                </button>
              </div>
            )}
          </div>

          {loading && !result && (
            <div className="space-y-2" aria-hidden="true">
              <div className="h-16 w-full animate-pulse rounded bg-gray-100 dark:bg-gray-800" />
              <div className="h-16 w-full animate-pulse rounded bg-gray-100 dark:bg-gray-800" />
            </div>
          )}

          {!loading && !result && (
            <p className="empty">Extracted entities, grouped by type, appear here.</p>
          )}

          {result && result.entities.length === 0 && (
            <p className="empty">No named entities found in this text.</p>
          )}

          {result && result.entities.length > 0 && (
            <>
              <div className="space-y-3">
                {groupedEntries.map(([label, values]) => {
                  const style = entityStyle(label)
                  return (
                    <div key={label} className="card p-3">
                      <div className="mb-2 flex items-center gap-2">
                        <span className={`h-2 w-2 rounded-full ${style.dot}`} aria-hidden="true" />
                        <span className="text-xs font-semibold uppercase tracking-wide text-gray-600 dark:text-gray-300">
                          {prettyLabel(label)}
                        </span>
                        <span className="text-xs text-gray-400 dark:text-gray-500">
                          {values.length}
                        </span>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {values.map((v, i) => (
                          <span
                            key={i}
                            className={`rounded border px-2 py-0.5 text-sm ${style.bg} ${style.text} ${style.border}`}
                          >
                            {v}
                          </span>
                        ))}
                      </div>
                    </div>
                  )
                })}
              </div>
              <Statistics statistics={result.statistics} />
              <details className="card p-3">
                <summary className="cursor-pointer text-xs font-medium text-gray-600 dark:text-gray-300">
                  View raw JSON
                </summary>
                <pre className="mt-2 overflow-x-auto rounded surface p-3 text-xs text-gray-700 dark:text-gray-300">
                  {JSON.stringify(grouped, null, 2)}
                </pre>
              </details>
            </>
          )}
        </section>
      </div>
    </div>
  )
}
