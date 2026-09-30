import { useEffect, useRef, useState } from 'react'
import { IndicTransliterate } from '@ai4bharat/indic-transliterate'
import { ask } from '../services/api'
import PageHeader from '../components/PageHeader'
import ChatMessage from '../components/ChatMessage'

const SUGGESTIONS = [
  'भारताची राजधानी कोणती?',
  'सचिन तेंडुलकर कोण आहेत?',
  'शिवाजी महाराजांबद्दल सांगा',
  'ताजमहाल कुठे आहे?',
]

export default function Assistant() {
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [translitOn, setTranslitOn] = useState(true)
  // Working context: the last retrieved Marathi-Wikipedia passage. Follow-up
  // questions (e.g. "ते कुठे राहतात?") are answered against this passage, so the
  // assistant supports multi-turn, context-grounded QA + coreference.
  const [context, setContext] = useState('')
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: 'smooth' })
  }, [messages, loading])

  async function send() {
    const q = question.trim()
    if (!q || loading) return
    setMessages((m) => [...m, { role: 'user', text: q }])
    setQuestion('')
    setLoading(true)
    try {
      const res = await ask(q, context)
      // A retrieval turn (sources present) establishes the new working context:
      // the retrieved passage, minus the trailing "स्रोत: …" footer line.
      if (res.sources?.length) {
        const passage = (res.answer || '').split('\n\nस्रोत:')[0].trim()
        if (passage) setContext(passage)
      }
      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          text: res.answer,
          entities: res.entities,
          steps: res.steps,
          sources: res.sources,
          intentLabel: res.intent_label,
          intentConfidence: res.intent_confidence,
          corefLinks: res.coref_links,
          contextUsed: res.context_used,
        },
      ])
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: 'assistant', text: e.message || 'Something went wrong. Please try again.' },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mx-auto flex max-w-3xl flex-col">
      <PageHeader
        title="AI Assistant"
        subtitle="Ask anything in Marathi. Answers are retrieved live from Marathi Wikipedia and grounded with on-device NER — deterministic, no external LLM."
      />

      <section className="flex min-h-[32rem] flex-1 flex-col">
        <div className="flex-1 space-y-3 overflow-y-auto rounded-lg border border-gray-200 p-4 dark:border-gray-800">
          {messages.length === 0 && (
            <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
              <p className="text-sm text-gray-500 dark:text-gray-400">
                कोणताही प्रश्न मराठीत विचारा. उदाहरणे:
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => setQuestion(s)}
                    className="rounded-full border border-gray-300 px-2.5 py-1 text-xs text-gray-600 hover:border-accent hover:text-accent dark:border-gray-700 dark:text-gray-300"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <ChatMessage
              key={i}
              role={m.role}
              text={m.text}
              entities={m.entities}
              steps={m.steps}
              sources={m.sources}
              intentLabel={m.intentLabel}
              intentConfidence={m.intentConfidence}
              corefLinks={m.corefLinks}
              contextUsed={m.contextUsed}
            />
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="card px-4 py-2.5">
                <span className="inline-flex gap-1" aria-label="Thinking">
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400 [animation-delay:-0.3s]" />
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400 [animation-delay:-0.15s]" />
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400" />
                </span>
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        <div className="mt-3 flex items-center justify-end">
          <label className="flex cursor-pointer items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400">
            <input
              type="checkbox"
              checked={translitOn}
              onChange={(e) => setTranslitOn(e.target.checked)}
              className="h-3.5 w-3.5 rounded border-gray-300 text-accent focus:ring-accent dark:border-gray-600"
            />
            Type in English (auto-convert to Marathi)
          </label>
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault()
            send()
          }}
          className="mt-1.5 flex gap-2"
        >
          <div className="flex-1 [&_.rli-container]:w-full [&>div]:w-full">
            {translitOn ? (
              <IndicTransliterate
                lang="mr"
                value={question}
                onChangeText={(t) => setQuestion(t)}
                enabled
                renderComponent={(props) => (
                  <input
                    {...props}
                    type="text"
                    placeholder="prashna vichara… (टाइप करा)"
                    className="input !py-2 w-full"
                    aria-label="Your question"
                  />
                )}
              />
            ) : (
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="प्रश्न विचारा…"
                className="input !py-2 w-full"
                aria-label="Your question"
              />
            )}
          </div>
          <button type="submit" disabled={loading || !question.trim()} className="btn-primary shrink-0">
            Send
          </button>
        </form>
      </section>
    </div>
  )
}
