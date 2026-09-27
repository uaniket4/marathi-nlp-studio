import { useEffect, useRef, useState } from 'react'
import { ask } from '../services/api'
import PageHeader from '../components/PageHeader'
import ChatMessage from '../components/ChatMessage'

const MAX_CHARS = 5000

const EXAMPLE_CONTEXT =
  'पंतप्रधान नरेंद्र मोदी यांनी १५ ऑगस्ट २०२२ रोजी दिल्लीत भाषण केले. यावेळी अमित शहा आणि राजनाथ सिंह उपस्थित होते.'

const SUGGESTIONS = [
  'या मजकुरात कोणत्या व्यक्ती आहेत?',
  'किती ठिकाणे आहेत?',
  'सर्व entities दाखवा',
  'कोणत्या तारखा आहेत?',
]

export default function Assistant() {
  const [context, setContext] = useState('')
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: 'smooth' })
  }, [messages, loading])

  async function send() {
    const q = question.trim()
    if (!q) return
    if (!context.trim()) {
      setError('Add some Marathi context text above, then ask a question.')
      return
    }
    setError('')
    setMessages((m) => [...m, { role: 'user', text: q }])
    setQuestion('')
    setLoading(true)
    try {
      const res = await ask(context, q)
      setMessages((m) => [
        ...m,
        { role: 'assistant', text: res.answer, entities: res.entities },
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
    <div>
      <PageHeader
        title="AI Assistant"
        subtitle="Ask questions in Marathi about a passage. Answers are derived deterministically from recognized entities — no hardcoded replies, no external LLM."
      />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        {/* Context */}
        <section className="space-y-2 lg:col-span-2">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">Context</h2>
            <button
              type="button"
              onClick={() => setContext(EXAMPLE_CONTEXT)}
              className="text-xs text-accent hover:underline dark:text-indigo-400"
            >
              Load example
            </button>
          </div>
          <textarea
            value={context}
            onChange={(e) => setContext(e.target.value.slice(0, MAX_CHARS))}
            rows={12}
            placeholder="ज्या मजकुराबद्दल प्रश्न विचारायचा आहे तो येथे टाका…"
            className="input resize-y leading-relaxed"
          />
          <p className="text-right text-xs text-gray-400 dark:text-gray-500">
            {context.length} / {MAX_CHARS}
          </p>
        </section>

        {/* Chat */}
        <section className="flex min-h-[28rem] flex-col lg:col-span-3">
          <div className="flex-1 space-y-3 overflow-y-auto rounded-lg border border-gray-200 p-4 dark:border-gray-800">
            {messages.length === 0 && (
              <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Ask a question about your context in Marathi.
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
              <ChatMessage key={i} role={m.role} text={m.text} entities={m.entities} />
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

          {error && (
            <div role="alert" className="mt-2 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300">
              {error}
            </div>
          )}

          <form
            onSubmit={(e) => {
              e.preventDefault()
              send()
            }}
            className="mt-3 flex gap-2"
          >
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="प्रश्न विचारा…"
              className="input !py-2"
              aria-label="Your question"
            />
            <button type="submit" disabled={loading || !question.trim()} className="btn-primary shrink-0">
              Send
            </button>
          </form>
        </section>
      </div>
    </div>
  )
}
