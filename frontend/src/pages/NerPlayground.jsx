import { useState } from 'react'
import { predict } from '../services/api'
import PageHeader from '../components/PageHeader'
import EntityHighlighter from '../components/EntityHighlighter'
import EntityTable from '../components/EntityTable'
import Statistics from '../components/Statistics'

const MAX_CHARS = 5000

const EXAMPLES = [
  {
    label: 'Person & Location',
    text: 'सचिन तेंडुलकर मुंबईमध्ये राहतात आणि त्यांनी भारतीय क्रिकेट संघासाठी खेळले.',
  },
  {
    label: 'News Example',
    text: 'पंतप्रधान नरेंद्र मोदी यांनी १५ ऑगस्ट रोजी दिल्लीत भाषण केले.',
  },
  {
    label: 'Organization Example',
    text: 'भारतीय जनता पक्ष आणि काँग्रेस यांच्यात महाराष्ट्रात निवडणूक होणार आहे.',
  },
]

export default function NerPlayground() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const trimmed = text.trim()

  async function analyze() {
    if (!trimmed) {
      setError('Please enter some Marathi text to analyze.')
      return
    }
    setLoading(true)
    setError('')
    try {
      const data = await predict(text)
      setResult(data)
    } catch (e) {
      setError(e.message || 'Something went wrong. Please try again.')
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  function clearAll() {
    setText('')
    setResult(null)
    setError('')
  }

  return (
    <div>
      <PageHeader
        title="NER Playground"
        subtitle="Analyze Marathi text and inspect every recognized entity, its type and confidence."
      />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Left: input */}
        <section aria-labelledby="input-heading" className="space-y-3">
          <h2 id="input-heading" className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Enter Marathi Text
          </h2>

          <textarea
            value={text}
            onChange={(e) => setText(e.target.value.slice(0, MAX_CHARS))}
            rows={8}
            placeholder="उदा. सचिन तेंडुलकर मुंबईमध्ये राहतात आणि त्यांनी भारतीय क्रिकेट संघासाठी खेळले."
            className="input resize-y text-base leading-relaxed"
            aria-describedby="char-count"
          />
          <div id="char-count" className="text-right text-xs text-gray-400 dark:text-gray-500">
            {text.length} / {MAX_CHARS}
          </div>

          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={analyze} disabled={loading || !trimmed} className="btn-primary">
              {loading && (
                <span
                  className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/40 border-t-white"
                  aria-hidden="true"
                />
              )}
              {loading ? 'Analyzing…' : 'Analyze Text'}
            </button>
            <button type="button" onClick={clearAll} disabled={loading} className="btn-secondary">
              Clear
            </button>
          </div>

          <div className="pt-1">
            <p className="mb-2 text-xs font-medium uppercase tracking-wide text-gray-400 dark:text-gray-500">
              Load an example
            </p>
            <div className="flex flex-wrap gap-2">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.label}
                  type="button"
                  onClick={() => {
                    setText(ex.text)
                    setError('')
                  }}
                  className="rounded-full border border-gray-300 px-3 py-1 text-xs text-gray-700 transition-colors hover:border-accent hover:text-accent dark:border-gray-700 dark:text-gray-300"
                >
                  {ex.label}
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* Right: results */}
        <section aria-labelledby="results-heading" className="space-y-4">
          <h2 id="results-heading" className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Recognized Entities
          </h2>

          {error && (
            <div
              role="alert"
              className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300"
            >
              {error}
            </div>
          )}

          {loading && !result && (
            <div className="space-y-2" aria-hidden="true">
              <div className="h-4 w-3/4 animate-pulse rounded bg-gray-100 dark:bg-gray-800" />
              <div className="h-4 w-1/2 animate-pulse rounded bg-gray-100 dark:bg-gray-800" />
              <div className="h-24 w-full animate-pulse rounded bg-gray-100 dark:bg-gray-800" />
            </div>
          )}

          {!loading && !result && !error && (
            <p className="empty">
              Enter text and select <span className="font-medium">Analyze Text</span> to
              see recognized entities here.
            </p>
          )}

          {result && (
            <div className="space-y-5">
              <div className="card p-4">
                <EntityHighlighter text={result.text} entities={result.entities} />
              </div>
              <EntityTable entities={result.entities} />
              <Statistics statistics={result.statistics} />
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
