import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getModelInfo } from '../services/api'
import PageHeader from '../components/PageHeader'

const APPS = [
  {
    to: '/playground',
    title: 'NER Playground',
    desc: 'Analyze any Marathi text and see recognized entities highlighted inline, tabulated and counted.',
    accent: 'text-blue-600 dark:text-blue-400',
  },
  {
    to: '/extract',
    title: 'Information Extraction',
    desc: 'Turn unstructured text into structured, typed entity lists you can copy or download as JSON / CSV.',
    accent: 'text-emerald-600 dark:text-emerald-400',
  },
  {
    to: '/search',
    title: 'Search',
    desc: 'Search a Marathi demo corpus with entity-aware ranking and entity-type filters.',
    accent: 'text-amber-600 dark:text-amber-400',
  },
  {
    to: '/assistant',
    title: 'AI Assistant',
    desc: 'Ask questions about a passage in Marathi; answers are derived directly from recognized entities.',
    accent: 'text-rose-600 dark:text-rose-400',
  },
]

export default function Overview() {
  const [info, setInfo] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    getModelInfo()
      .then(setInfo)
      .catch((e) => setError(e.message))
  }, [])

  return (
    <div>
      <PageHeader
        title="Marathi NLP Studio"
        subtitle="Understand, extract and search Marathi text using Named Entity Recognition."
      />

      {/* Model status banner */}
      <div className="card mb-8 p-4">
        {error ? (
          <p className="text-sm text-amber-700 dark:text-amber-400">
            Backend unavailable: {error}
          </p>
        ) : !info ? (
          <div className="h-5 w-64 animate-pulse rounded bg-gray-100 dark:bg-gray-800" />
        ) : (
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
            <span className="flex items-center gap-2 font-medium text-gray-900 dark:text-gray-100">
              <span className="h-2 w-2 rounded-full bg-emerald-500" aria-hidden="true" />
              Model ready
            </span>
            <span className="text-gray-500 dark:text-gray-400">
              {info.model_name}
            </span>
            <span className="text-gray-500 dark:text-gray-400">
              {info.num_entity_classes} entity types
            </span>
            {info.metrics?.summary?.f1 != null && (
              <span className="text-gray-500 dark:text-gray-400">
                F1 {(info.metrics.summary.f1 * 100).toFixed(1)}% (test)
              </span>
            )}
            <Link
              to="/model"
              className="text-accent hover:underline dark:text-indigo-400"
            >
              Model &amp; dataset details →
            </Link>
          </div>
        )}
      </div>

      {/* Application cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {APPS.map((app) => (
          <Link
            key={app.to}
            to={app.to}
            className="card group p-5 transition-all hover:-translate-y-0.5 hover:border-accent/50 hover:shadow-sm"
          >
            <div className="flex items-center justify-between">
              <h2 className={`font-semibold ${app.accent}`}>{app.title}</h2>
              <span className="text-gray-300 transition-transform group-hover:translate-x-0.5 dark:text-gray-600">
                →
              </span>
            </div>
            <p className="mt-2 text-sm leading-relaxed text-gray-600 dark:text-gray-400">
              {app.desc}
            </p>
          </Link>
        ))}
      </div>

      <p className="mt-6 text-xs text-gray-400 dark:text-gray-500">
        All four tools are powered by the same fine-tuned Marathi NER model — no
        external APIs, no fabricated results.
      </p>
    </div>
  )
}
