import { entityStyle, prettyLabel, formatConfidence } from '../services/entityStyles'

export default function EntityTable({ entities }) {
  if (!entities || entities.length === 0) {
    return <p className="empty">No named entities detected.</p>
  }

  return (
    <div className="overflow-hidden rounded-md border border-gray-200 dark:border-gray-800">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="surface text-left text-gray-600 dark:text-gray-300">
            <th className="px-3 py-2 font-medium">Entity</th>
            <th className="px-3 py-2 font-medium">Type</th>
            <th className="px-3 py-2 font-medium text-right">Confidence</th>
          </tr>
        </thead>
        <tbody>
          {entities.map((ent, i) => {
            const style = entityStyle(ent.label)
            return (
              <tr key={i} className="border-t border-gray-100 dark:border-gray-800">
                <td className="px-3 py-2 text-gray-900 dark:text-gray-100">{ent.text}</td>
                <td className="px-3 py-2">
                  <span className="inline-flex items-center gap-1.5">
                    <span className={`h-2 w-2 rounded-full ${style.dot}`} aria-hidden="true" />
                    <span className={`rounded px-1.5 py-0.5 text-xs font-medium ${style.bg} ${style.text}`}>
                      {prettyLabel(ent.label)}
                    </span>
                  </span>
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-gray-600 dark:text-gray-400">
                  {formatConfidence(ent.confidence) || '—'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
