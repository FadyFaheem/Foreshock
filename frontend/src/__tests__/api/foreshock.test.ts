import { describe, it, expect, vi, beforeEach } from 'vitest'
import { getSamples, getSignal, predictSample } from '../../api/foreshock'

beforeEach(() => {
  vi.mocked(fetch).mockReset()
})

describe('foreshock api module', () => {
  it('getSamples requests /api/samples', async () => {
    vi.mocked(fetch).mockResolvedValue(new Response('[]', { status: 200 }))
    await getSamples()
    expect(vi.mocked(fetch).mock.calls[0][0]).toBe('/api/samples')
  })

  it('getSignal URL-encodes the id', async () => {
    vi.mocked(fetch).mockResolvedValue(new Response('{}', { status: 200 }))
    await getSignal('inner race')
    expect(vi.mocked(fetch).mock.calls[0][0]).toBe('/api/signal/inner%20race')
  })

  it('predictSample posts FormData carrying the sample id', async () => {
    vi.mocked(fetch).mockResolvedValue(new Response('{}', { status: 200 }))
    await predictSample('ball')
    const [url, opts] = vi.mocked(fetch).mock.calls[0]
    expect(url).toBe('/api/predict')
    expect(opts?.method).toBe('POST')
    expect(opts?.body).toBeInstanceOf(FormData)
    expect((opts?.body as FormData).get('sample_id')).toBe('ball')
  })
})
