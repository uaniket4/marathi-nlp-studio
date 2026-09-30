import { prettyLabel, entityStyle } from '../services/entityStyles'

// A single chat turn. `role` is 'user' or 'assistant'. Assistant answers may
// contain newlines (lists), an optional set of supporting entities, optional
// source links (e.g. the Wikipedia article an answer was retrieved from), an
// optional detected intent (label + confidence from the ML intent classifier),
// optional coreference links (rule-based), the context the answer used, and an
// optional reasoning trace (`steps`) showing how the answer was derived.
export default function ChatMessage({
  role,
  text,
  entities,
  steps,
  sources,
  intentLabel,
  intentConfidence,
  corefLinks,
  contextUsed,
}) {
  const isUser = role === 'user'
  const confPct =
    typeof intentConfidence === 'number' ? Math.round(intentConfidence * 100) : null
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
          isUser
            ? 'bg-accent text-accent-fg'
            : 'card whitespace-pre-wrap text-gray-800 dark:text-gray-100'
        }`}
      >
        <span className={isUser ? '' : 'whitespace-pre-wrap'}>{text}</span>
        {!isUser && intentLabel && (
          <div className="mt-2 flex flex-wrap items-center gap-1.5 border-t border-gray-100 pt-2 text-xs dark:border-gray-800">
            <span className="text-gray-400 dark:text-gray-500">Detected intent:</span>
            <span className="rounded bg-indigo-50 px-1.5 py-0.5 font-medium text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-300">
              {intentLabel}
            </span>
            {confPct !== null && (
              <span className="text-gray-500 dark:text-gray-400">{confPct}% confidence</span>
            )}
          </div>
        )}
        {!isUser && corefLinks?.length > 0 && (
          <div className="mt-2 flex flex-wrap items-center gap-1.5 border-t border-gray-100 pt-2 text-xs dark:border-gray-800">
            <span className="text-gray-400 dark:text-gray-500">Coreference:</span>
            {corefLinks.map((l, i) => (
              <span
                key={i}
                className="rounded bg-amber-50 px-1.5 py-0.5 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300"
              >
                {l.mention} → {l.antecedent}
              </span>
            ))}
          </div>
        )}
        {!isUser && entities?.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1 border-t border-gray-100 pt-2 dark:border-gray-800">
            {entities.map((e, i) => {
              const style = entityStyle(e.label)
              return (
                <span
                  key={i}
                  className={`rounded border px-1.5 py-0.5 text-xs ${style.bg} ${style.text} ${style.border}`}
                  title={prettyLabel(e.label)}
                >
                  {e.text}
                </span>
              )
            })}
          </div>
        )}
        {!isUser && sources?.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2 border-t border-gray-100 pt-2 text-xs dark:border-gray-800">
            <span className="text-gray-400 dark:text-gray-500">स्रोत:</span>
            {sources.map((s, i) => (
              <a
                key={i}
                href={s.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-accent hover:underline dark:text-indigo-400"
              >
                {s.title}
              </a>
            ))}
          </div>
        )}
        {!isUser && contextUsed && (
          <details className="mt-2 border-t border-gray-100 pt-2 dark:border-gray-800">
            <summary className="cursor-pointer text-xs font-medium text-gray-500 dark:text-gray-400">
              Context used
            </summary>
            <p className="mt-1.5 text-xs italic text-gray-500 dark:text-gray-400">
              {contextUsed}
            </p>
          </details>
        )}
        {!isUser && steps?.length > 0 && (
          <details className="mt-2 border-t border-gray-100 pt-2 dark:border-gray-800">
            <summary className="cursor-pointer text-xs font-medium text-gray-500 dark:text-gray-400">
              How this answer was derived
            </summary>
            <ol className="mt-2 space-y-1.5">
              {steps.map((s, i) => (
                <li key={s.id || i} className="text-xs text-gray-600 dark:text-gray-400">
                  <span className="font-medium text-gray-700 dark:text-gray-300">
                    {i + 1}. {s.title}
                    {typeof s.count === 'number' ? ` (${s.count})` : ''}:
                  </span>{' '}
                  {s.description}
                </li>
              ))}
            </ol>
          </details>
        )}
      </div>
    </div>
  )
}
