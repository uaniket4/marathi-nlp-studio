import { prettyLabel, entityStyle } from '../services/entityStyles'

// A single chat turn. `role` is 'user' or 'assistant'. Assistant answers may
// contain newlines (lists) and an optional set of supporting entities.
export default function ChatMessage({ role, text, entities }) {
  const isUser = role === 'user'
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
          isUser
            ? 'bg-accent text-accent-fg'
            : 'card whitespace-pre-wrap text-gray-800 dark:text-gray-100'
        }`}
      >
        <span className={isUser ? '' : 'whitespace-pre-wrap'}>{text}</span>
        {!isUser && entities?.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1 border-t border-gray-100 pt-2 dark:border-gray-800">
            {entities.map((e, i) => {
              const style = entityStyle(e.label)
              return (
                <span
                  key={i}
                  className={`rounded border px-1.5 py-0.5 text-xs ${style.bg} ${style.text} ${style.border}`}
                  title={prettyLabel(e.label)}
                >
                  {e.text}
                </span>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
