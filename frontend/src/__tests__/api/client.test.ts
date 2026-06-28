import { describe, it, expect, vi, beforeEach } from 'vitest'
import { apiFetch, apiJson, throwIfNotOk } from '../../api/client'

beforeEach(() => {
  vi.mocked(fetch).mockReset()
})

describe('apiFetch', () => {
  it('sets Content-Type for a JSON body', async () => {
    vi.mocked(fetch).mockResolvedValue(new Response('{}', { status: 200 }))
    await apiFetch('/api/predict', {
      method: 'POST',
      body: JSON.stringify({ a: 1 }),
    })
    const headers = vi.mocked(fetch).mock.calls[0][1]?.headers as Headers
    expect(headers.get('Content-Type')).toBe('application/json')
  })

  it('does not set Content-Type for a FormData body', async () => {
    vi.mocked(fetch).mockResolvedValue(new Response('{}', { status: 200 }))
    await apiFetch('/api/predict', { method: 'POST', body: new FormData() })
    const headers = vi.mocked(fetch).mock.calls[0][1]?.headers as Headers
    expect(headers.has('Content-Type')).toBe(false)
  })

  it('preserves custom headers', async () => {
    vi.mocked(fetch).mockResolvedValue(new Response('{}', { status: 200 }))
    await apiFetch('/api/samples', { headers: { 'X-Custom': 'v' } })
    const headers = vi.mocked(fetch).mock.calls[0][1]?.headers as Headers
    expect(headers.get('X-Custom')).toBe('v')
  })
})

describe('apiJson', () => {
  it('returns parsed JSON on success', async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 }),
    )
    const data = await apiJson<{ ok: boolean }>('/api/samples')
    expect(data.ok).toBe(true)
  })

  it('throws the server error message on failure', async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ error: 'nope' }), { status: 400 }),
    )
    await expect(apiJson('/api/predict', { method: 'POST' })).rejects.toThrow('nope')
  })
})

describe('throwIfNotOk', () => {
  it('does nothing for ok responses', async () => {
    await expect(
      throwIfNotOk(new Response('{}', { status: 200 }), 'fallback'),
    ).resolves.toBeUndefined()
  })
})
