/** Resolve a public snapshot, an explicit URL, or the local file-picker mode. */
export async function resolveDataSource(pageURL, baseURL, { fetcher = fetch, development = false } = {}) {
  const page = new URL(pageURL);
  if (page.searchParams.has('db')) {
    const path = page.searchParams.get('db');
    if (!path) throw new Error('The data URL is empty.');
    const url = new URL(path, baseURL);
    if (!['http:', 'https:'].includes(url.protocol)) throw new Error('The data URL must use HTTP or HTTPS.');
    return { url: url.href, manifest: null };
  }
  if (page.searchParams.get('local') === '1') return null;

  const manifestURL = new URL('data/manifest.json', baseURL);
  const response = await fetcher(manifestURL.href, { cache: 'no-store' });
  if (development && (response.status === 404 || response.headers.get('content-type')?.includes('text/html'))) return null;
  if (!response.ok) throw new Error('Published data is unavailable. Please try again later.');
  let manifest;
  try {
    manifest = await response.json();
  } catch {
    throw new Error('The published data description is invalid.');
  }
  if (manifest?.schema_version !== 1 ||
      !/^[a-f0-9]{64}$/.test(manifest.sha256) ||
      manifest.snapshot !== `snapshots/${manifest.sha256}.duckdb` ||
      !Number.isSafeInteger(manifest.size_bytes) || manifest.size_bytes <= 0 ||
      typeof manifest.exported_at !== 'string' || !Number.isFinite(Date.parse(manifest.exported_at)) ||
      !Number.isSafeInteger(manifest.row_counts?.search_stats) || manifest.row_counts.search_stats <= 0) {
    throw new Error('The published data description is invalid.');
  }
  return { url: new URL(manifest.snapshot, manifestURL).href, manifest };
}
