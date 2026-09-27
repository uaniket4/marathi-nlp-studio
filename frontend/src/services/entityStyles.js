// Visual styling per entity type. Colors are paired with the always-visible
// label text, so meaning never depends on color alone (accessibility).
// Each string carries its own dark-mode variant so chips read well on both
// light and dark surfaces. Types not listed fall back to a neutral gray.

const STYLES = {
  PERSON: {
    bg: 'bg-blue-100 dark:bg-blue-500/20',
    text: 'text-blue-900 dark:text-blue-200',
    border: 'border-blue-300 dark:border-blue-500/40',
    dot: 'bg-blue-500',
  },
  LOCATION: {
    bg: 'bg-emerald-100 dark:bg-emerald-500/20',
    text: 'text-emerald-900 dark:text-emerald-200',
    border: 'border-emerald-300 dark:border-emerald-500/40',
    dot: 'bg-emerald-500',
  },
  ORGANIZATION: {
    bg: 'bg-amber-100 dark:bg-amber-500/20',
    text: 'text-amber-900 dark:text-amber-200',
    border: 'border-amber-300 dark:border-amber-500/40',
    dot: 'bg-amber-500',
  },
  DATE: {
    bg: 'bg-purple-100 dark:bg-purple-500/20',
    text: 'text-purple-900 dark:text-purple-200',
    border: 'border-purple-300 dark:border-purple-500/40',
    dot: 'bg-purple-500',
  },
  TIME: {
    bg: 'bg-pink-100 dark:bg-pink-500/20',
    text: 'text-pink-900 dark:text-pink-200',
    border: 'border-pink-300 dark:border-pink-500/40',
    dot: 'bg-pink-500',
  },
  MEASURE: {
    bg: 'bg-teal-100 dark:bg-teal-500/20',
    text: 'text-teal-900 dark:text-teal-200',
    border: 'border-teal-300 dark:border-teal-500/40',
    dot: 'bg-teal-500',
  },
  DESIGNATION: {
    bg: 'bg-rose-100 dark:bg-rose-500/20',
    text: 'text-rose-900 dark:text-rose-200',
    border: 'border-rose-300 dark:border-rose-500/40',
    dot: 'bg-rose-500',
  },
}

const FALLBACK = {
  bg: 'bg-gray-100 dark:bg-gray-700/40',
  text: 'text-gray-900 dark:text-gray-200',
  border: 'border-gray-300 dark:border-gray-600',
  dot: 'bg-gray-500',
}

export function entityStyle(label) {
  return STYLES[label] || FALLBACK
}

// Title-case a display label for readability, e.g. PERSON -> Person.
export function prettyLabel(label) {
  if (!label) return ''
  return label.charAt(0) + label.slice(1).toLowerCase()
}

export function formatConfidence(conf) {
  if (typeof conf !== 'number') return ''
  return `${Math.round(conf * 100)}%`
}
