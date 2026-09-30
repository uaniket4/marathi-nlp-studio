import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import Assistant from '../pages/Assistant'
import Search from '../pages/Search'
import * as api from '../services/api'

vi.mock('../services/api')

beforeEach(() => vi.clearAllMocks())

describe('Assistant', () => {
  it('sends a question and shows the retrieved answer', async () => {
    api.ask.mockResolvedValue({
      answer: 'नवी दिल्ली ही भारताची राजधानी आहे.\n\nस्रोत: विकिपीडिया — नवी दिल्ली',
      intent: { types: [], is_count: false, is_all: false },
      entities: [{ text: 'नवी दिल्ली', label: 'LOCATION', start: 0, end: 10, confidence: 0.97 }],
      sources: [{ title: 'नवी दिल्ली', url: 'https://mr.wikipedia.org/wiki/नवी_दिल्ली' }],
    })
    render(<Assistant />)
    fireEvent.change(screen.getByRole('textbox'), {
      target: { value: 'भारताची राजधानी कोणती?' },
    })
    fireEvent.click(screen.getByRole('button', { name: /send/i }))
    await waitFor(() => expect(screen.getByText(/भारताची राजधानी/)).toBeInTheDocument())
    expect(api.ask).toHaveBeenCalledWith('भारताची राजधानी कोणती?', '')
  })

  it('shows detected intent + coreference on a context-grounded follow-up', async () => {
    api.ask.mockResolvedValue({
      answer: 'या मजकुरात 1 व्यक्तींचा उल्लेख आहे:\n1. रतन टाटा',
      intent: { types: ['PERSON'], is_count: false, is_all: false },
      entities: [{ text: 'रतन टाटा', label: 'PERSON', start: 0, end: 8, confidence: 0.99 }],
      sources: [],
      intent_label: 'PERSON_QUERY',
      intent_confidence: 0.98,
      coref_links: [{ mention: 'ते', antecedent: 'रतन टाटा', type: 'PERSON' }],
      context_used: 'रतन टाटा हे मुंबई येथे राहत होते.',
    })
    render(<Assistant />)
    fireEvent.change(screen.getByRole('textbox'), {
      target: { value: 'ते कोण आहेत?' },
    })
    fireEvent.click(screen.getByRole('button', { name: /send/i }))
    await waitFor(() => expect(screen.getByText('PERSON_QUERY')).toBeInTheDocument())
    expect(screen.getByText(/98% confidence/)).toBeInTheDocument()
    expect(screen.getByText(/ते → रतन टाटा/)).toBeInTheDocument()
  })
})

describe('Search', () => {
  it('runs a search and lists results', async () => {
    api.getModelInfo.mockResolvedValue({ entity_types: ['PERSON', 'LOCATION'] })
    api.search.mockResolvedValue({
      query: 'मुंबई',
      query_entities: [{ text: 'मुंबई', label: 'LOCATION', start: 0, end: 5, confidence: 0.98 }],
      query_keywords: ['मुंबई'],
      total: 1,
      results: [
        {
          id: 'd1',
          title: 'Doc 1',
          snippet: 'मुंबई विषयी…',
          text: 'मुंबई विषयी मजकूर',
          entities: [],
          matched_entities: [{ text: 'मुंबई', label: 'LOCATION', start: 0, end: 5, confidence: 0.98 }],
          matched_keywords: ['मुंबई'],
          entity_match_count: 1,
          keyword_match_count: 1,
          relevance: 0.8,
        },
      ],
    })
    render(
      <MemoryRouter>
        <Search />
      </MemoryRouter>,
    )
    fireEvent.change(screen.getByRole('textbox', { name: /search query/i }), {
      target: { value: 'मुंबई' },
    })
    fireEvent.click(screen.getByRole('button', { name: /^search$/i }))
    await waitFor(() => expect(screen.getByText('Doc 1')).toBeInTheDocument())
    expect(screen.getByText(/80% match/)).toBeInTheDocument()
  })
})
