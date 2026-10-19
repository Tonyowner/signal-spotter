// Service worker: caches the whole app (including the ~18 MB model) on first visit
// so it works fully offline afterwards.
const VERSION = '1.2';
const SHELL_CACHE = `signal-shell-${VERSION}`;
const ASSET_CACHE = 'signal-assets-v1';   // model + tfjs; rarely change, keep across shell updates

const SHELL = [
  './',
  'index.html',
  'app.js',
  'style.css',
  'manifest.webmanifest',
  'icons/icon-180.png',
  'icons/icon-192.png',
  'icons/icon-512.png',
  'audio/red.wav',
  'audio/yellow.wav',
  'audio/green.wav',
];
const ASSETS = [
  'vendor/tf.min.js',
  'models/ssdlite_mobilenet_v2/model.json',
  'models/ssdlite_mobilenet_v2/group1-shard1of5',
  'models/ssdlite_mobilenet_v2/group1-shard2of5',
  'models/ssdlite_mobilenet_v2/group1-shard3of5',
  'models/ssdlite_mobilenet_v2/group1-shard4of5',
  'models/ssdlite_mobilenet_v2/group1-shard5of5',
];

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const shell = await caches.open(SHELL_CACHE);
    await shell.addAll(SHELL);
    const assets = await caches.open(ASSET_CACHE);
    for (const url of ASSETS) {
      if (!(await assets.match(url))) await assets.add(url);
    }
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keep = new Set([SHELL_CACHE, ASSET_CACHE]);
    for (const key of await caches.keys()) if (!keep.has(key)) await caches.delete(key);
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin) return;
  const path = new URL(req.url).pathname;
  const isAsset = path.includes('/models/') || path.includes('/vendor/');

  if (isAsset) {
    // Cache-first: large, versioned files.
    event.respondWith((async () => {
      const cache = await caches.open(ASSET_CACHE);
      const hit = await cache.match(req);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) cache.put(req, res.clone());
      return res;
    })());
    return;
  }

  // App shell: serve from cache instantly, refresh in the background when online.
  event.respondWith((async () => {
    const cache = await caches.open(SHELL_CACHE);
    const hit = await cache.match(req, { ignoreSearch: true });
    const refresh = fetch(req).then((res) => {
      if (res.ok) cache.put(req, res.clone());
      return res;
    }).catch(() => null);
    if (hit) { event.waitUntil(refresh); return hit; }
    const res = await refresh;
    return res || (await cache.match('index.html')) || Response.error();
  })());
});
