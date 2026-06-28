// Minimal API client. The template's version handles JWT auth + 401 refresh;
// Foreshock has no auth (stateless demo), so this is the stripped-down core:
// a single place that all requests flow through, FormData-aware.

export async function throwIfNotOk(res: Response, fallback: string): Promise<void> {
  if (res.ok) return;
  const body = await res.json().catch(() => ({}));
  throw new Error((body as Record<string, string>).error || fallback);
}

export async function apiFetch(
  url: string,
  options: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(options.headers);
  if (
    !headers.has('Content-Type') &&
    options.body &&
    !(options.body instanceof FormData)
  ) {
    headers.set('Content-Type', 'application/json');
  }
  return fetch(url, { ...options, headers });
}

export async function apiJson<T>(
  url: string,
  options: RequestInit = {},
): Promise<T> {
  const res = await apiFetch(url, options);
  await throwIfNotOk(res, 'Request failed');
  return res.json() as Promise<T>;
}
