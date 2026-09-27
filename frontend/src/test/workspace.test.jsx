import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import InformationExtraction from '../pages/InformationExtraction'
import Assistant from '../pages/Assistant'
import Search from '../pages/Search'
import * as api from '../services/api'

vi.mock('../services/api')

beforeEach(() => vi.clearAllMocks())

describe('InformationExtraction', () => {
  it('renders grouped entities after extraction', async () => {
    api.extract.mockResolvedValue({
      text: 'रतन टाटा',
      entities: [{ text: 'रतन टाटा', label: 'PERSON', start: 0, end: 8, confidence: 0.99 }],
      grouped: { PERSON: ['रतन टाटा'] },
      statistics: { PERSON: 1 },
    })
    render(<InformationExtraction />)
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'रतन टाटा' } })
    fireEvent.click(screen.getByRole('button', { name: /extract entities/i }))
    await waitFor(() => expect(screen.getByText('रतन टाटा')).toBeInTheDocument())
    expect(screen.getByRole('button', { name: /copy json/i })).toBeInTheDocument()
  })
})

describe('Assistant', () => {
  it('sends a question and shows the derived answer', async () => {
    api.ask.mockResolvedValue({
      answer: 'या मजकुरात 1 व्यक्तींचा उल्लेख आहे:\n1. मोदी',
      intent: { types: ['PERSON'], is_count: false, is_all: false },
      entities: [{ text: 'मोदी', label: 'PERSON', start: 0, end: 4, confidence: 0.97 }],
    })
    render(<Assistant />)
    const boxes = screen.getAllByRole('textbox')
    fireEvent.change(boxes[0], { target: { value: 'मोदी यांनी भाषण केले.' } })
    fireEvent.change(boxes[1], { target: { value: 'कोणत्या व्यक्ती आहेत?' } })
    fireEvent.click(screen.getByRole('button', { name: /send/i }))
    await waitFor(() => expect(screen.getByText(/व्यक्तींचा उल्लेख/)).toBeInTheDocument())
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
