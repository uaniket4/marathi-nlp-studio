import { useEffect, useRef, useState } from 'react'
import { search, searchDocument, uploadDocument, getModelInfo } from '../services/api'
import PageHeader from '../components/PageHeader'
import SearchResult from '../components/SearchResult'
import PipelineSteps from '../components/PipelineSteps'
import { prettyLabel, entityStyle } from '../services/entityStyles'

const EXAMPLES = ['मुंबई', 'नरेंद्र मोदी', 'क्रिकेट', 'भारत']
const MAX_CHARS = 5000

export default function Search() {
  const [mode, setMode] = useState('corpus') // 'corpus' | 'document'
  const [query, setQuery] = useState('')
  const [types, setTypes] = useState([]) // available entity types from the model
  const [selected, setSelected] = useState(new Set())
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [searched, setSearched] = useState(false)

  // Document-mode state.
  const [docText, setDocText] = useState('')
  const [docName, setDocName] = useState('')
  const [uploading, setUploading] = useState(false)
  const fileRef = useRef(null)

  useEffect(() => {
    getModelInfo()
      .then((info) => setTypes(info.entity_types || []))
      .catch(() => setTypes([]))
  }, [])

  function toggleType(t) {
    setSelected((prev) => {
      const next = new Set(prev)
      next.has(t) ? next.delete(t) : next.add(t)
      return next
    })
  }

  function switchMode(next) {
    setMode(next)
    setData(null)
    setError('')
    setSearched(false)
  }

  async function onFile(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    setError('')
    try {
      const res = await uploadDocument(file)
      setDocText(res.text)
      setDocName(res.filename)
      if (res.truncated) {
        setError(`Loaded “${res.filename}”, trimmed to the first ${MAX_CHARS} characters.`)
      }
    } catch (err) {
      setError(err.message || 'Could not read that file.')
    } finally {
      setUploading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }
  async function run(q = query) {
    const trimmed = q.trim()
    if (!trimmed) {
      setError('Enter a search query.')
      return
    }
    if (mode === 'document' && !docText.trim()) {
      setError('Paste or upload a document to search within.')
      return
    }
    setLoading(true)
    setError('')
    setSearched(true)
    try {
      const res =
        mode === 'document'
          ? await searchDocument(docText, trimmed, [...selected], 10)
          : await search(trimmed, [...selected], 10)
      setData(res)
    } catch (e) {
      setError(e.message || 'Search failed. Please try again.')
      setData(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <PageHeader
        title="Search"
        subtitle="Entity-aware search over a Marathi demo corpus or your own document. Ranking combines entity and keyword matches — this is lexical + entity matching, not semantic search."
      />

      {/* Mode toggle */}
      <div className="mb-4 inline-flex rounded-lg border border-gray-200 p-0.5 dark:border-gray-800">
        {[
          ['corpus', 'Demo corpus'],
          ['document', 'My document'],
        ].map(([val, label]) => (
          <button
            key={val}
            type="button"
            onClick={() => switchMode(val)}
            aria-pressed={mode === val}
            className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
              mode === val
                ? 'bg-accent text-white'
                : 'text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-100'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Document input (document mode only) */}
      {mode === 'document' && (
        <div className="mb-4 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <input
              ref={fileRef}
              type="file"
              accept=".txt,.md,.pdf,.docx"
              onChange={onFile}
              className="block text-sm text-gray-600 file:mr-3 file:rounded-md file:border-0 file:bg-gray-100 file:px-3 file:py-1.5 file:text-sm file:text-gray-700 hover:file:bg-gray-200 dark:text-gray-400 dark:file:bg-gray-800 dark:file:text-gray-200"
            />
            {uploading && <span className="text-xs text-gray-400">Reading…</span>}
            {docName && !uploading && (
              <span className="text-xs text-gray-500 dark:text-gray-400">Loaded: {docName}</span>
            )}
          </div>
          <textarea
            value={docText}
            onChange={(e) => setDocText(e.target.value.slice(0, MAX_CHARS))}
            rows={6}
            placeholder="येथे मजकूर पेस्ट करा किंवा वरून .txt / .pdf / .docx फाईल अपलोड करा…"
            className="input resize-y leading-relaxed"
          />
          <p className="text-right text-xs text-gray-400 dark:text-gray-500">
            {docText.length} / {MAX_CHARS}
          </p>
        </div>
      )}

      {/* Search bar */}
      <form
        onSubmit={(e) => {
          e.preventDefault()
          run()
        }}
        className="flex gap-2"
      >
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="उदा. मुंबई मधील क्रिकेट खेळाडू"
          className="input !py-2"
          aria-label="Search query"
        />
        <button type="submit" disabled={loading} className="btn-primary shrink-0">
          {loading && (
            <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/40 border-t-white" aria-hidden="true" />
          )}
          Search
        </button>
      </form>

      {/* Examples (corpus mode) */}
      {mode === 'corpus' && (
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <span className="text-xs text-gray-400 dark:text-gray-500">Try:</span>
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => {
                setQuery(ex)
                run(ex)
              }}
              className="rounded-full border border-gray-300 px-2.5 py-0.5 text-xs text-gray-600 hover:border-accent hover:text-accent dark:border-gray-700 dark:text-gray-300"
            >
              {ex}
            </button>
          ))}
        </div>
      )}

      {/* Entity-type filters (from the actual model labels) */}
      {types.length > 0 && (
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
            Filter by entity type:
          </span>
          {types.map((t) => {
            const active = selected.has(t)
            const style = entityStyle(t)
            return (
              <button
                key={t}
                type="button"
                aria-pressed={active}
                onClick={() => toggleType(t)}
                className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs transition-colors ${
                  active
                    ? `${style.bg} ${style.text} ${style.border}`
                    : 'border-gray-300 text-gray-500 hover:border-gray-400 dark:border-gray-700 dark:text-gray-400'
                }`}
              >
                <span className={`h-1.5 w-1.5 rounded-full ${style.dot}`} aria-hidden="true" />
                {prettyLabel(t)}
              </button>
            )
          })}
          {selected.size > 0 && (
            <button
              type="button"
              onClick={() => setSelected(new Set())}
              className="text-xs text-gray-400 underline hover:text-gray-600 dark:hover:text-gray-200"
            >
              Clear
            </button>
          )}
        </div>
      )}

      {error && (
        <div role="alert" className="mt-4 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300">
          {error}
        </div>
      )}

      {/* Results */}
      <div className="mt-6 space-y-4">
        {loading && (
          <div className="space-y-3" aria-hidden="true">
            {[0, 1, 2].map((i) => (
              <div key={i} className="h-24 w-full animate-pulse rounded-lg bg-gray-100 dark:bg-gray-800" />
            ))}
          </div>
        )}

        {!loading && data && (
          <>
            {data.steps && <PipelineSteps steps={data.steps} title="How the search works" />}

            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-500 dark:text-gray-400">
              <span>
                {data.total} result{data.total === 1 ? '' : 's'} for “{data.query}”
                {typeof data.passage_count === 'number'
                  ? ` across ${data.passage_count} passage${data.passage_count === 1 ? '' : 's'}`
                  : ''}
              </span>
              {data.query_entities?.length > 0 && (
                <span>
                  Query entities:{' '}
                  {data.query_entities.map((e, i) => (
                    <span key={i} className="text-gray-700 dark:text-gray-300">
                      {e.text} ({prettyLabel(e.label)}){i < data.query_entities.length - 1 ? ', ' : ''}
                    </span>
                  ))}
                </span>
              )}
            </div>

            {data.results.length === 0 ? (
              <p className="empty">
                No {mode === 'document' ? 'passages' : 'documents'} matched. Try a different query or remove type filters.
              </p>
            ) : (
              data.results.map((r) => <SearchResult key={r.id} result={r} />)
            )}
          </>
        )}

        {!loading && !data && !searched && (
          <p className="empty">Search results appear here.</p>
        )}
      </div>
    </div>
  )
}
