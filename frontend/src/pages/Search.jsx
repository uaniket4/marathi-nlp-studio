import { useEffect, useState } from 'react'
import { search, getModelInfo } from '../services/api'
import PageHeader from '../components/PageHeader'
import SearchResult from '../components/SearchResult'
import { prettyLabel, entityStyle } from '../services/entityStyles'

const EXAMPLES = ['मुंबई', 'नरेंद्र मोदी', 'क्रिकेट', 'भारत']

export default function Search() {
  const [query, setQuery] = useState('')
  const [types, setTypes] = useState([]) // available entity types from the model
  const [selected, setSelected] = useState(new Set())
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [searched, setSearched] = useState(false)

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

  async function run(q = query) {
    const trimmed = q.trim()
    if (!trimmed) {
      setError('Enter a search query.')
      return
    }
    setLoading(true)
    setError('')
    setSearched(true)
    try {
      const res = await search(trimmed, [...selected], 10)
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
        subtitle="Entity-aware search over a Marathi demo corpus. Ranking combines entity and keyword matches — this is lexical + entity matching, not semantic search."
      />

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

      {/* Examples */}
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
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-500 dark:text-gray-400">
              <span>
                {data.total} result{data.total === 1 ? '' : 's'} for “{data.query}”
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
                No documents matched. Try a different query or remove type filters.
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
