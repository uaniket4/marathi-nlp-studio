import { entityStyle, prettyLabel, formatConfidence } from '../services/entityStyles'

// Renders the original text with entity spans highlighted, using character
// offsets from the API. Non-entity text is rendered verbatim so spacing and
// punctuation are preserved exactly.
export default function EntityHighlighter({ text, entities }) {
  if (!text) return null

  // Sort by start offset and build alternating plain / entity segments.
  const sorted = [...entities].sort((a, b) => a.start - b.start)
  const segments = []
  let cursor = 0

  sorted.forEach((ent, i) => {
    if (ent.start > cursor) {
      segments.push({ type: 'text', value: text.slice(cursor, ent.start), key: `t${i}` })
    }
    segments.push({ type: 'entity', ent, key: `e${i}` })
    cursor = Math.max(cursor, ent.end)
  })
  if (cursor < text.length) {
    segments.push({ type: 'text', value: text.slice(cursor), key: 'tail' })
  }

  return (
    <p className="whitespace-pre-wrap break-words text-lg leading-loose text-gray-900 dark:text-gray-100">
      {segments.map((seg) => {
        if (seg.type === 'text') return <span key={seg.key}>{seg.value}</span>
        const { ent } = seg
        const style = entityStyle(ent.label)
        const conf = formatConfidence(ent.confidence)
        return (
          <span
            key={seg.key}
            tabIndex={0}
            role="mark"
            className={`entity-mark border ${style.bg} ${style.text} ${style.border}`}
            title={`${prettyLabel(ent.label)}${conf ? ` · ${conf}` : ''}`}
            aria-label={`${ent.text}, ${prettyLabel(ent.label)}${conf ? `, confidence ${conf}` : ''}`}
          >
            {ent.text}
            <sup className="ml-0.5 select-none text-[0.6em] font-semibold uppercase opacity-70">
              {ent.label}
            </sup>
          </span>
        )
      })}
    </p>
  )
}
