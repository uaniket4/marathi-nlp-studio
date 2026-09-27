import { entityStyle, prettyLabel } from '../services/entityStyles'

// A single search result card: title, relevance, snippet and the entities that
// matched the query (highlighted) plus any matched keywords.
export default function SearchResult({ result }) {
  const relevancePct = Math.round((result.relevance || 0) * 100)

  return (
    <article className="card p-4">
      <div className="flex items-start justify-between gap-3">
        <h3 className="font-medium text-gray-900 dark:text-gray-100">{result.title}</h3>
        <span
          className="shrink-0 rounded-full bg-accent-soft px-2 py-0.5 text-xs font-medium text-accent dark:bg-accent/15 dark:text-indigo-300"
          title="Relevance score"
        >
          {relevancePct}% match
        </span>
      </div>

      <p className="mt-2 text-sm leading-relaxed text-gray-600 dark:text-gray-300">
        {result.snippet}
      </p>

      {(result.matched_entities?.length > 0 || result.matched_keywords?.length > 0) && (
        <div className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-gray-100 pt-3 dark:border-gray-800">
          {result.matched_entities?.length > 0 && (
            <span className="text-xs text-gray-400 dark:text-gray-500">Entities:</span>
          )}
          {dedupeEntities(result.matched_entities).map((e, i) => {
            const style = entityStyle(e.label)
            return (
              <span
                key={`e${i}`}
                className={`rounded border px-1.5 py-0.5 text-xs ${style.bg} ${style.text} ${style.border}`}
                title={prettyLabel(e.label)}
              >
                {e.text}
              </span>
            )
          })}
          {result.matched_keywords?.length > 0 && (
            <>
              <span className="ml-1 text-xs text-gray-400 dark:text-gray-500">Keywords:</span>
              {result.matched_keywords.map((k, i) => (
                <span
                  key={`k${i}`}
                  className="rounded border border-gray-200 px-1.5 py-0.5 text-xs text-gray-600 dark:border-gray-700 dark:text-gray-300"
                >
                  {k}
                </span>
              ))}
            </>
          )}
        </div>
      )}
    </article>
  )
}

function dedupeEntities(entities = []) {
  const seen = new Set()
  const out = []
  for (const e of entities) {
    const key = `${e.text}|${e.label}`
    if (!seen.has(key)) {
      seen.add(key)
      out.push(e)
    }
  }
  return out
}
