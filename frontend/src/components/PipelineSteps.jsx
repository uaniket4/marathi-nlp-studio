import { useState } from 'react'

// Renders an ordered list of NLP pipeline stages returned by the API.
// Each step carries a title, a plain-language description and an optional count.
// The numbers come from real model output — nothing here is hardcoded.
export default function PipelineSteps({ steps, title = 'How the model sees it' }) {
  const [open, setOpen] = useState(true)

  if (!steps || steps.length === 0) return null

  return (
    <section className="card p-4">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center justify-between text-left"
      >
        <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">{title}</h3>
        <span className="text-xs text-gray-400 dark:text-gray-500">
          {open ? 'Hide' : 'Show'} steps
        </span>
      </button>

      {open && (
        <ol className="mt-3 space-y-3">
          {steps.map((step, i) => (
            <li key={step.id || i} className="flex gap-3">
              <span
                className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent/10 text-xs font-semibold text-accent dark:bg-indigo-500/20 dark:text-indigo-300"
                aria-hidden="true"
              >
                {i + 1}
              </span>
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    {step.title}
                  </p>
                  {typeof step.count === 'number' && (
                    <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs tabular-nums text-gray-600 dark:bg-gray-800 dark:text-gray-300">
                      {step.count}
                    </span>
                  )}
                </div>
                <p className="mt-0.5 text-sm leading-relaxed text-gray-600 dark:text-gray-400">
                  {step.description}
                </p>
              </div>
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}
