import { entityStyle, prettyLabel, formatConfidence } from '../services/entityStyles'

// "How the model sees it" — the per-token trace. Each subword token is shown as
// a chip colored by its predicted entity type, with the BIO tag and the model's
// confidence. Special tokens ([CLS]/[SEP]) are shown muted. All values come
// straight from the model output.
export default function TokenTrace({ tokens }) {
  if (!tokens || tokens.length === 0) return null

  const content = tokens.filter((t) => !t.is_special)
  const labeled = content.filter((t) => t.entity_type)
  // Entity types present in this trace, for a compact legend.
  const legend = [...new Set(labeled.map((t) => t.entity_type))]

  return (
    <section className="card p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
          Token-level predictions
        </h3>
        <span className="text-xs text-gray-400 dark:text-gray-500">
          {content.length} subword tokens · {labeled.length} tagged
        </span>
      </div>

      {legend.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-2">
          {legend.map((t) => {
            const style = entityStyle(t)
            return (
              <span key={t} className="inline-flex items-center gap-1.5 text-xs text-gray-600 dark:text-gray-300">
                <span className={`h-2 w-2 rounded-full ${style.dot}`} aria-hidden="true" />
                {prettyLabel(t)}
              </span>
            )
          })}
        </div>
      )}

      <div className="flex flex-wrap gap-1.5">
        {tokens.map((tok) => {
          if (tok.is_special) {
            return (
              <span
                key={tok.index}
                className="rounded border border-dashed border-gray-300 px-1.5 py-1 font-mono text-xs text-gray-400 dark:border-gray-700 dark:text-gray-500"
                title="Special token"
              >
                {tok.token}
              </span>
            )
          }
          const style = tok.entity_type ? entityStyle(tok.entity_type) : null
          const conf = formatConfidence(tok.confidence)
          const tag = tok.entity_type
            ? `${tok.boundary && tok.boundary !== 'FLAT' ? `${tok.boundary}-` : ''}${tok.entity_type}`
            : 'O'
          return (
            <span
              key={tok.index}
              tabIndex={0}
              className={`group relative flex flex-col items-center rounded border px-1.5 py-1 ${
                style
                  ? `${style.bg} ${style.text} ${style.border}`
                  : 'border-gray-200 text-gray-600 dark:border-gray-700 dark:text-gray-400'
              }`}
              title={`${tag} · confidence ${conf} · predicted "${tok.predicted_label}"`}
            >
              <span className="font-mono text-sm leading-tight">{tok.token}</span>
              <span className="mt-0.5 text-[0.6rem] font-semibold uppercase tracking-wide opacity-70">
                {tag}
              </span>
              <span className="mt-0.5 h-0.5 w-full overflow-hidden rounded bg-black/10 dark:bg-white/10" aria-hidden="true">
                <span
                  className="block h-full bg-current opacity-60"
                  style={{ width: `${Math.round((tok.confidence || 0) * 100)}%` }}
                />
              </span>
            </span>
          )
        })}
      </div>
      <p className="mt-3 text-xs text-gray-400 dark:text-gray-500">
        Chips show each subword token, its predicted tag (B-/I- mark entity
        boundaries, O = outside) and the model's confidence bar.
      </p>
    </section>
  )
}
