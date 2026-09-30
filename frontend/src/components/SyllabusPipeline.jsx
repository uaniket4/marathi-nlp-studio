import { useState } from 'react'
import { entityStyle, prettyLabel, formatConfidence } from '../services/entityStyles'

// Renders the syllabus-aligned classical NLP pipeline returned by /analyze.
// Each stage is a collapsible card with a module/experiment badge and a
// stage-specific view of the real computed output. Nothing here is hardcoded —
// every value comes from the backend pipeline.

function Badge({ module, experiment }) {
  if (!module) return null
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-accent/10 px-2 py-0.5 text-[0.65rem] font-medium text-accent dark:bg-indigo-500/20 dark:text-indigo-300">
      {module}
      {experiment && experiment !== '—' ? ` · ${experiment}` : ''}
    </span>
  )
}

function Chips({ items, mono = true }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((t, i) => (
        <span
          key={i}
          className={`rounded border border-gray-200 bg-gray-50 px-1.5 py-0.5 text-xs text-gray-700 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-200 ${
            mono ? 'font-mono' : ''
          }`}
        >
          {t}
        </span>
      ))}
    </div>
  )
}

function TokenizeView({ data }) {
  return <Chips items={data.tokens || []} />
}

function StemView({ data }) {
  const pairs = data.pairs || []
  return (
    <div className="flex flex-wrap gap-2">
      {pairs.map((p, i) => (
        <span
          key={i}
          className="inline-flex items-center gap-1 rounded border border-gray-200 px-1.5 py-0.5 text-xs dark:border-gray-700"
        >
          <span className="font-mono text-gray-700 dark:text-gray-200">{p.word}</span>
          {p.changed && (
            <>
              <span className="text-gray-400">→</span>
              <span className="font-mono font-medium text-accent dark:text-indigo-300">{p.stem}</span>
            </>
          )}
        </span>
      ))}
    </div>
  )
}

function NgramView({ data }) {
  const rows = data.bigrams || []
  return (
    <div>
      <p className="mb-2 text-xs text-gray-500 dark:text-gray-400">
        Vocabulary: {data.vocab_size} · bigram tokens: {data.bigram_count} · δ={data.delta}
      </p>
      {rows.length === 0 ? (
        <p className="text-xs text-gray-400">Need at least two words for a bigram.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-gray-500 dark:text-gray-400">
              <tr>
                <th className="py-1 pr-3 font-medium">bigram (w₁ → w₂)</th>
                <th className="py-1 pr-3 font-medium tabular-nums">count</th>
                <th className="py-1 pr-3 font-medium tabular-nums">Unsmoothed</th>
                <th className="py-1 pr-3 font-medium tabular-nums">Add-One</th>
                <th className="py-1 font-medium tabular-nums">Add-δ</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {rows.map((r, i) => (
                <tr key={i} className="border-t border-gray-100 dark:border-gray-800">
                  <td className="py-1 pr-3">{r.w1} → {r.w2}</td>
                  <td className="py-1 pr-3 tabular-nums">{r.count}</td>
                  <td className="py-1 pr-3 tabular-nums">{r.p_unsmoothed}</td>
                  <td className="py-1 pr-3 tabular-nums">{r.p_add_one}</td>
                  <td className="py-1 tabular-nums">{r.p_add_delta}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}


function MorphologyView({ data }) {
  return (
    <div className="space-y-2">
      {(data.items || []).map((it, i) => (
        <div key={i} className="flex flex-wrap items-center gap-2 text-xs">
          <span className="font-mono font-medium text-gray-900 dark:text-gray-100">{it.word}</span>
          <span className="text-gray-400">→</span>
          <Chips items={it.aksharas} />
          {it.suffix ? (
            <span className="text-gray-500 dark:text-gray-400">
              root <span className="font-mono">{it.root}</span> + suffix{' '}
              <span className="font-mono text-accent dark:text-indigo-300">{it.suffix}</span>
            </span>
          ) : (
            <span className="text-gray-400 dark:text-gray-500">no suffix</span>
          )}
        </div>
      ))}
    </div>
  )
}

// POS tag color by broad class, so the tag sequence is scannable.
const POS_COLOR = {
  NOUN: 'bg-blue-100 text-blue-800 dark:bg-blue-500/20 dark:text-blue-200',
  PROPN: 'bg-blue-100 text-blue-800 dark:bg-blue-500/20 dark:text-blue-200',
  VERB: 'bg-rose-100 text-rose-800 dark:bg-rose-500/20 dark:text-rose-200',
  AUX: 'bg-rose-100 text-rose-800 dark:bg-rose-500/20 dark:text-rose-200',
  ADJ: 'bg-amber-100 text-amber-800 dark:bg-amber-500/20 dark:text-amber-200',
  ADV: 'bg-amber-100 text-amber-800 dark:bg-amber-500/20 dark:text-amber-200',
  PRON: 'bg-purple-100 text-purple-800 dark:bg-purple-500/20 dark:text-purple-200',
}
const POS_FALLBACK = 'bg-gray-100 text-gray-700 dark:bg-gray-700/50 dark:text-gray-200'

function PosView({ data }) {
  const tagged = data.tagged || []
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-1.5">
        {tagged.map((t, i) => (
          <span
            key={i}
            className={`flex flex-col items-center rounded px-1.5 py-1 ${POS_COLOR[t.tag] || POS_FALLBACK}`}
            title={`${t.description}${t.known ? '' : ' · unseen in training'}`}
          >
            <span className="font-mono text-sm leading-tight">{t.token}</span>
            <span className="mt-0.5 text-[0.6rem] font-semibold uppercase tracking-wide opacity-80">
              {t.tag}
              {!t.known && <span title="not in training data">*</span>}
            </span>
          </span>
        ))}
      </div>
      {(data.transitions || []).length > 0 && (
        <p className="text-xs text-gray-500 dark:text-gray-400">
          Sample transitions taken:{' '}
          <span className="font-mono">
            {data.transitions.map((tr) => `P(${tr.to}|${tr.from})=${tr.prob}`).join('  ')}
          </span>
        </p>
      )}
      <p className="text-xs text-gray-400 dark:text-gray-500">
        Trained on the small UD_Marathi-UFAL treebank ({data.treebank_tokens} tokens).
        <span className="ml-1">* = word unseen in training. Teaching demo — no accuracy claimed.</span>
      </p>
    </div>
  )
}

function ChunkView({ data }) {
  const chunks = data.chunks || []
  if (chunks.length === 0)
    return <p className="text-xs text-gray-400">No noun-phrase chunks found.</p>
  return (
    <div className="flex flex-wrap gap-2">
      {chunks.map((c, i) => (
        <span
          key={i}
          className="rounded border border-emerald-300 bg-emerald-50 px-2 py-1 text-xs text-emerald-900 dark:border-emerald-500/40 dark:bg-emerald-500/15 dark:text-emerald-200"
          title={`[NP ${c.tags.join(' ')}]`}
        >
          [NP {c.text}]
        </span>
      ))}
    </div>
  )
}

function NerView({ data }) {
  const entities = data.entities || []
  if (entities.length === 0)
    return <p className="text-xs text-gray-400">No named entities found.</p>
  return (
    <div className="flex flex-wrap gap-2">
      {entities.map((e, i) => {
        const style = entityStyle(e.label)
        return (
          <span
            key={i}
            className={`inline-flex items-center gap-1.5 rounded border px-2 py-1 text-xs ${style.bg} ${style.text} ${style.border}`}
            title={`${prettyLabel(e.label)} · ${formatConfidence(e.confidence)}`}
          >
            {e.text}
            <span className="opacity-70">{prettyLabel(e.label)}</span>
          </span>
        )
      })}
    </div>
  )
}

const SENTIMENT_STYLE = {
  positive: 'border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-500/40 dark:bg-emerald-500/15 dark:text-emerald-200',
  negative: 'border-rose-300 bg-rose-50 text-rose-800 dark:border-rose-500/40 dark:bg-rose-500/15 dark:text-rose-200',
  neutral: 'border-gray-300 bg-gray-50 text-gray-700 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-200',
}
const SENTIMENT_MR = { positive: 'सकारात्मक', negative: 'नकारात्मक', neutral: 'तटस्थ' }

function SentimentView({ data }) {
  return (
    <div className="space-y-2 text-xs">
      <span
        className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 font-medium ${SENTIMENT_STYLE[data.label]}`}
      >
        {SENTIMENT_MR[data.label]} ({data.label}) · score {data.score}
      </span>
      {data.positive_words?.length > 0 && (
        <p className="text-gray-600 dark:text-gray-300">
          Positive: <span className="font-mono">{data.positive_words.join(', ')}</span>
        </p>
      )}
      {data.negative_words?.length > 0 && (
        <p className="text-gray-600 dark:text-gray-300">
          Negative: <span className="font-mono">{data.negative_words.join(', ')}</span>
        </p>
      )}
      <p className="text-gray-400 dark:text-gray-500">
        Small illustrative lexicon — not a trained classifier; no negation/context handling.
      </p>
    </div>
  )
}

const VIEWS = {
  tokenize: TokenizeView,
  morphology: MorphologyView,
  stemming: StemView,
  ngram: NgramView,
  pos: PosView,
  chunk: ChunkView,
  ner: NerView,
  sentiment: SentimentView,
}

function StageCard({ stage, index, defaultOpen }) {
  const [open, setOpen] = useState(defaultOpen)
  const View = VIEWS[stage.id]
  return (
    <li className="card p-4">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-start justify-between gap-3 text-left"
      >
        <div className="flex items-start gap-3">
          <span
            className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent/10 text-xs font-semibold text-accent dark:bg-indigo-500/20 dark:text-indigo-300"
            aria-hidden="true"
          >
            {index + 1}
          </span>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <p className="text-sm font-medium text-gray-900 dark:text-gray-100">{stage.title}</p>
              <Badge module={stage.module} experiment={stage.experiment} />
            </div>
            <p className="mt-0.5 text-sm leading-relaxed text-gray-600 dark:text-gray-400">
              {stage.description}
            </p>
          </div>
        </div>
        <span className="shrink-0 text-xs text-gray-400 dark:text-gray-500">{open ? 'Hide' : 'Show'}</span>
      </button>
      {open && View && (
        <div className="mt-3 border-t border-gray-100 pt-3 dark:border-gray-800">
          <View data={stage.data || {}} />
        </div>
      )}
    </li>
  )
}

export default function SyllabusPipeline({ stages }) {
  if (!stages || stages.length === 0) return null
  return (
    <section className="space-y-3">
      <div>
        <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
          Classical NLP pipeline
        </h3>
        <p className="text-xs text-gray-500 dark:text-gray-400">
          Each stage follows the NLP lab syllabus and shows real computed output for your text.
        </p>
      </div>
      <ol className="space-y-3">
        {stages.map((stage, i) => (
          <StageCard key={stage.id || i} stage={stage} index={i} defaultOpen={i < 3} />
        ))}
      </ol>
    </section>
  )
}
