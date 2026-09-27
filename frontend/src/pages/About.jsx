import PageHeader from '../components/PageHeader'

const APPLICATIONS = [
  {
    title: 'Information Extraction',
    body: 'Converts free Marathi text into structured, typed entity lists. Useful for populating databases, tagging documents, or feeding downstream pipelines with clean JSON / CSV.',
  },
  {
    title: 'Search',
    body: 'Ranks a demo corpus by how well each document matches the entities and keywords in your query. This is transparent lexical + entity matching, not semantic vector search.',
  },
  {
    title: 'AI Assistant',
    body: 'Answers Marathi questions about a passage by detecting the asked-about entity type and reading the answer straight from the model’s predictions. Every answer is derived from real output — nothing is hardcoded.',
  },
]

export default function About() {
  return (
    <div className="max-w-3xl">
      <PageHeader
        title="About"
        subtitle="What this studio does and how it works."
      />

      <div className="space-y-8">
        <section>
          <h2 className="mb-2 text-sm font-semibold text-gray-900 dark:text-gray-100">
            What is Named Entity Recognition?
          </h2>
          <p className="text-sm leading-relaxed text-gray-600 dark:text-gray-400">
            Named Entity Recognition (NER) identifies and classifies important
            entities in text — people, organizations, locations, dates and other
            categories defined by the dataset. Instead of treating text as a flat
            string, NER turns it into structured information that downstream
            systems can act on.
          </p>
        </section>

        <section>
          <h3 className="mb-2 text-sm font-semibold text-gray-900 dark:text-gray-100">
            Why Marathi NER?
          </h3>
          <p className="mb-3 text-sm leading-relaxed text-gray-600 dark:text-gray-400">
            Marathi is among the most widely spoken languages in India but has far
            fewer NLP resources than English. A dedicated Marathi NER model enables
            information extraction, search, assistants and media analysis directly
            on Marathi text.
          </p>
        </section>

        <section>
          <h3 className="mb-3 text-sm font-semibold text-gray-900 dark:text-gray-100">
            The three applications
          </h3>
          <div className="space-y-3">
            {APPLICATIONS.map((a) => (
              <div key={a.title} className="card p-4">
                <h4 className="font-medium text-gray-900 dark:text-gray-100">{a.title}</h4>
                <p className="mt-1 text-sm leading-relaxed text-gray-600 dark:text-gray-400">
                  {a.body}
                </p>
              </div>
            ))}
          </div>
        </section>

        <section>
          <h3 className="mb-2 text-sm font-semibold text-gray-900 dark:text-gray-100">
            How it works
          </h3>
          <p className="text-sm leading-relaxed text-gray-600 dark:text-gray-400">
            Text you enter is sent to a transformer-based token-classification model
            fine-tuned on the L3Cube-MahaNER dataset. The model predicts an entity
            type for each token; adjacent tokens of the same type are grouped into a
            single entity and mapped back to the exact characters in your input. All
            four tools share this single model instance — there are no external APIs
            and no fabricated results.
          </p>
        </section>
      </div>
    </div>
  )
}
