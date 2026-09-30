import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getModelInfo } from '../services/api'
import PageHeader from '../components/PageHeader'

const APPS = [
  {
    to: '/playground',
    title: 'NER Playground',
    desc: 'Analyze any Marathi text: see entities highlighted, tabulated and counted, walk the full classical-NLP pipeline, and export the results as JSON / CSV.',
    accent: 'text-blue-600 dark:text-blue-400',
  },
  {
    to: '/search',
    title: 'Search',
    desc: 'Search a Marathi demo corpus or your own document with entity-aware ranking, entity-type filters and a step-by-step view of how each search runs.',
    accent: 'text-amber-600 dark:text-amber-400',
  },
  {
    to: '/assistant',
    title: 'AI Assistant',
    desc: 'Ask questions in Marathi; answers are retrieved live from Marathi Wikipedia and grounded with on-device NER — deterministic, no external LLM.',
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
        Every tool is powered by the same fine-tuned Marathi NER model — no
        fabricated results. The assistant additionally retrieves live from
        Marathi Wikipedia; no external LLM is used anywhere.
      </p>
    </div>
  )
}
