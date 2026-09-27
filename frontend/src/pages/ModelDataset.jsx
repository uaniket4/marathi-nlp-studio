import { useEffect, useState } from 'react'
import { getModelInfo, getDatasetInfo } from '../services/api'
import PageHeader from '../components/PageHeader'
import { entityStyle, prettyLabel } from '../services/entityStyles'

function Row({ label, value }) {
  return (
    <div className="flex justify-between gap-4 border-b border-gray-100 py-2 text-sm last:border-0 dark:border-gray-800">
      <dt className="text-gray-500 dark:text-gray-400">{label}</dt>
      <dd className="text-right font-medium text-gray-900 dark:text-gray-100">{value}</dd>
    </div>
  )
}

function pct(v) {
  return typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '—'
}

export default function ModelDataset() {
  const [info, setInfo] = useState(null)
  const [dataset, setDataset] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getModelInfo(), getDatasetInfo().catch(() => null)])
      .then(([m, d]) => {
        setInfo(m)
        setDataset(d)
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading)
    return <p className="text-sm text-gray-500 dark:text-gray-400">Loading model & dataset information…</p>
  if (error)
    return (
      <div role="alert" className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300">
        {error}
      </div>
    )

  const metrics = info?.metrics
  const summary = metrics?.summary
  const counts = metrics?.counts
  const dist = dataset?.entity_distribution_train || {}
  const maxCount = Math.max(1, ...Object.values(dist))

  return (
    <div className="max-w-3xl">
      <PageHeader
        title="Model & Dataset"
        subtitle="The fine-tuned model behind every tool, and the dataset it learned from."
      />
      {/* SECTIONS_PLACEHOLDER */}
      <div className="space-y-8">
        <section>
          <h2 className="mb-3 text-sm font-semibold text-gray-900 dark:text-gray-100">
            Model
          </h2>
          <dl className="card p-4">
            <Row label="Architecture" value={info.architecture} />
            <Row label="Model" value={info.model_name} />
            <Row label="Base pretrained model" value={info.base_model} />
            <Row label="Fine-tuning dataset" value={info.dataset} />
            <Row label="Entity classes" value={info.num_entity_classes} />
            <Row label="Max sequence length" value={info.max_length} />
            {counts && (
              <>
                <Row label="Training examples" value={counts.train?.toLocaleString()} />
                <Row label="Validation examples" value={counts.validation?.toLocaleString()} />
                <Row label="Test examples" value={counts.test?.toLocaleString()} />
              </>
            )}
          </dl>
        </section>

        <section>
          <h3 className="mb-3 text-sm font-semibold text-gray-900 dark:text-gray-100">
            Evaluation (test split)
          </h3>
          {summary ? (
            <>
              <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                {[
                  ['Precision', summary.precision],
                  ['Recall', summary.recall],
                  ['F1-score', summary.f1],
                  ['Accuracy', summary.accuracy],
                ].map(([k, v]) => (
                  <div key={k} className="card px-3 py-2">
                    <div className="text-xs text-gray-500 dark:text-gray-400">{k}</div>
                    <div className="text-xl font-semibold tabular-nums text-gray-900 dark:text-gray-100">
                      {pct(v)}
                    </div>
                  </div>
                ))}
              </div>

              {metrics.per_entity && (
                <div className="overflow-hidden rounded-md border border-gray-200 dark:border-gray-800">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="surface text-left text-gray-600 dark:text-gray-300">
                        <th className="px-3 py-2 font-medium">Entity</th>
                        <th className="px-3 py-2 text-right font-medium">Precision</th>
                        <th className="px-3 py-2 text-right font-medium">Recall</th>
                        <th className="px-3 py-2 text-right font-medium">F1</th>
                        <th className="px-3 py-2 text-right font-medium">Support</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(metrics.per_entity).map(([label, m]) => (
                        <tr key={label} className="border-t border-gray-100 dark:border-gray-800">
                          <td className="px-3 py-2 text-gray-900 dark:text-gray-100">{prettyLabel(label)}</td>
                          <td className="px-3 py-2 text-right tabular-nums text-gray-600 dark:text-gray-400">{pct(m.precision)}</td>
                          <td className="px-3 py-2 text-right tabular-nums text-gray-600 dark:text-gray-400">{pct(m.recall)}</td>
                          <td className="px-3 py-2 text-right tabular-nums text-gray-600 dark:text-gray-400">{pct(m.f1)}</td>
                          <td className="px-3 py-2 text-right tabular-nums text-gray-600 dark:text-gray-400">{m.support}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <p className="mt-3 text-xs text-gray-400 dark:text-gray-500">
                Entity-level scores computed with seqeval on the MahaNER test split.
              </p>
            </>
          ) : (
            <div className="empty">
              <p className="font-medium text-gray-700 dark:text-gray-200">Model not trained yet</p>
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                Run <code className="rounded bg-gray-200 px-1 dark:bg-gray-700">python training/evaluate.py</code> to
                generate real metrics.
              </p>
            </div>
          )}
        </section>

        {/* DATASET_PLACEHOLDER */}
        {dataset && (
          <>
            <section>
              <h3 className="mb-3 text-sm font-semibold text-gray-900 dark:text-gray-100">
                Dataset
              </h3>
              <dl className="card p-4">
                <Row label="Name" value={dataset.name} />
                <Row label="Language" value={dataset.language} />
                <Row label="Entity types" value={dataset.num_entity_classes} />
                <Row
                  label="Source"
                  value={
                    <a className="text-accent hover:underline dark:text-indigo-400" href={dataset.source} target="_blank" rel="noreferrer">
                      l3cube-pune/MarathiNLP
                    </a>
                  }
                />
                {dataset.corpus_size > 0 && (
                  <Row label="Search demo corpus" value={`${dataset.corpus_size} documents`} />
                )}
              </dl>
            </section>

            <section>
              <h3 className="mb-3 text-sm font-semibold text-gray-900 dark:text-gray-100">Splits</h3>
              <div className="grid grid-cols-3 gap-3">
                {['train', 'validation', 'test'].map((k) => (
                  <div key={k} className="card px-3 py-2">
                    <div className="text-xs capitalize text-gray-500 dark:text-gray-400">{k}</div>
                    <div className="text-xl font-semibold tabular-nums text-gray-900 dark:text-gray-100">
                      {dataset.splits[k]?.toLocaleString()}
                    </div>
                    <div className="text-xs text-gray-400 dark:text-gray-500">sentences</div>
                  </div>
                ))}
              </div>
            </section>

            <section>
              <h3 className="mb-3 text-sm font-semibold text-gray-900 dark:text-gray-100">
                Entity distribution
              </h3>
              <p className="mb-4 text-xs text-gray-500 dark:text-gray-400">
                Counted over the training split.
              </p>
              <ul className="space-y-2">
                {Object.entries(dist)
                  .sort((a, b) => b[1] - a[1])
                  .map(([label, count]) => {
                    const style = entityStyle(label.toUpperCase())
                    const width = `${(count / maxCount) * 100}%`
                    return (
                      <li key={label} className="flex items-center gap-3">
                        <span className="w-24 shrink-0 text-sm text-gray-700 dark:text-gray-300">
                          {prettyLabel(label.toUpperCase())}
                        </span>
                        <span className="relative h-5 flex-1 rounded bg-gray-100 dark:bg-gray-800">
                          <span className={`absolute inset-y-0 left-0 rounded ${style.dot}`} style={{ width }} />
                        </span>
                        <span className="w-14 shrink-0 text-right text-sm tabular-nums text-gray-600 dark:text-gray-400">
                          {count.toLocaleString()}
                        </span>
                      </li>
                    )
                  })}
              </ul>
            </section>
          </>
        )}
      </div>
    </div>
  )
}
