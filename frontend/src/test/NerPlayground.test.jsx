import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import NerPlayground from '../pages/NerPlayground'
import * as api from '../services/api'

vi.mock('../services/api')

const SAMPLE = {
  text: 'सचिन तेंडुलकर मुंबईमध्ये राहतात.',
  tokens: ['सचिन', 'तेंडुलकर', 'मुंबईमध्ये', 'राहतात'],
  entities: [
    { text: 'सचिन तेंडुलकर', label: 'PERSON', start: 0, end: 13, confidence: 0.99 },
    { text: 'मुंबई', label: 'LOCATION', start: 14, end: 19, confidence: 0.98 },
  ],
  statistics: { PERSON: 1, LOCATION: 1 },
  stages: [
    {
      id: 'tokenize',
      module: 'Module 6 — Applications',
      experiment: 'Exp 10',
      title: 'Word tokenization',
      description: 'Split into word tokens.',
      data: { tokens: ['सचिन', 'तेंडुलकर', 'मुंबईमध्ये', 'राहतात'] },
    },
  ],
}

describe('NerPlayground', () => {
  beforeEach(() => vi.clearAllMocks())

  it('renders input and disabled analyze button initially', () => {
    render(<NerPlayground />)
    expect(screen.getByText('Enter Marathi Text')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /analyze text/i })).toBeDisabled()
  })

  it('accepts text input and enables analyze', () => {
    render(<NerPlayground />)
    const textarea = screen.getByRole('textbox')
    fireEvent.change(textarea, { target: { value: 'सचिन तेंडुलकर' } })
    expect(textarea.value).toBe('सचिन तेंडुलकर')
    expect(screen.getByRole('button', { name: /analyze text/i })).toBeEnabled()
  })

  it('shows results after a successful analysis', async () => {
    api.analyze.mockResolvedValue(SAMPLE)
    render(<NerPlayground />)
    fireEvent.change(screen.getByRole('textbox'), {
      target: { value: SAMPLE.text },
    })
    fireEvent.click(screen.getByRole('button', { name: /analyze text/i }))

    await waitFor(() => expect(screen.getByRole('table')).toBeInTheDocument())
    expect(screen.getAllByText(/Person/i).length).toBeGreaterThan(0)
    expect(screen.getByText('Entities Found')).toBeInTheDocument()
    // Export controls (folded in from the old Information Extraction page).
    expect(screen.getByRole('button', { name: /copy json/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /csv/i })).toBeInTheDocument()
  })

  it('shows an error message when the API fails', async () => {
    api.analyze.mockRejectedValue(new Error('Cannot reach the server.'))
    render(<NerPlayground />)
    fireEvent.change(screen.getByRole('textbox'), {
      target: { value: 'test' },
    })
    fireEvent.click(screen.getByRole('button', { name: /analyze text/i }))
    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/cannot reach/i),
    )
  })

  it('clears text and results', async () => {
    api.analyze.mockResolvedValue(SAMPLE)
    render(<NerPlayground />)
    const textarea = screen.getByRole('textbox')
    fireEvent.change(textarea, { target: { value: SAMPLE.text } })
    fireEvent.click(screen.getByRole('button', { name: /analyze text/i }))
    await waitFor(() => expect(screen.getByRole('table')).toBeInTheDocument())

    fireEvent.click(screen.getByRole('button', { name: /clear/i }))
    expect(textarea.value).toBe('')
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })

  it('loads an example sentence', () => {
    render(<NerPlayground />)
    fireEvent.click(screen.getByRole('button', { name: /person & location/i }))
    expect(screen.getByRole('textbox').value).toContain('सचिन')
  })
})
