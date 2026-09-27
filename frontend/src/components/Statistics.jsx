import { prettyLabel, entityStyle } from '../services/entityStyles'

// Small statistics panel. Categories are derived from the response, never
// hardcoded — only types the model actually produced are shown.
export default function Statistics({ statistics }) {
  const entries = Object.entries(statistics || {}).sort((a, b) => b[1] - a[1])
  const total = entries.reduce((sum, [, n]) => sum + n, 0)

  if (total === 0) return null

  return (
    <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3">
      <div className="surface col-span-2 rounded-md border border-gray-200 px-3 py-2 dark:border-gray-800 sm:col-span-3">
        <dt className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
          Entities Found
        </dt>
        <dd className="text-2xl font-semibold tabular-nums text-gray-900 dark:text-gray-100">
          {total}
        </dd>
      </div>
      {entries.map(([label, count]) => {
        const style = entityStyle(label)
        return (
          <div
            key={label}
            className="rounded-md border border-gray-200 px-3 py-2 dark:border-gray-800"
          >
            <dt className="flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400">
              <span className={`h-2 w-2 rounded-full ${style.dot}`} aria-hidden="true" />
              {prettyLabel(label)}
            </dt>
            <dd className="text-xl font-semibold tabular-nums text-gray-900 dark:text-gray-100">
              {count}
            </dd>
          </div>
        )
      })}
    </dl>
  )
}
