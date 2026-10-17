/* MAP-IN Vision · Identifikasi AI — frontend (GitHub Pages)
 * Semua pemrosesan gambar berjalan di browser:
 *  - Mesin AI     : MediaPipe Image Embedder (MobileNetV3) → cosine similarity ke gambar master
 *  - Mesin Klasik : korelasi (NCC) intensitas + tepi Sobel  (fallback jika model AI gagal dimuat)
 *  - Pelacakan    : template matching NCC dalam area cari → dX/dY
 *  - Trigger AI   : deteksi gerak → diam
 * Backend opsional: Google Apps Script (log, program, gambar NG, proxy Gemini untuk asisten).
 */

const MP_VER = '0.10.14';
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}`;
const MP_MODEL = 'https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite';
const DEFAULT_THR = { ai: 0.85, classic: 0.75 };
const MAX_SAMPLES = 6;

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function load(key, def) {
  try { const v = JSON.parse(localStorage.getItem(key)); return v == null ? def : { ...def, ...v }; } catch { return def; }
}
function store(key, val) {
  try { localStorage.setItem(key, JSON.stringify(val)); } catch (e) { console.warn('localStorage', e); }
}

/* ============================== STATE ============================== */

const S = {
  settings: load('mv_settings', {
    gasUrl: '', token: '', log: true, saveNg: false, tts: true, beep: true, autoAsk: false,
    wkUrl: '', room: 'line-1', stationId: 'st-' + Math.random().toString(36).slice(2, 6), stationName: 'Stasiun 1', rt: true,
    trigger: 'manual', interval: 800, move: 6, deviceId: '', res: '1280x720'
  }),
  program: load('mv_program', newProgram()),
  step: 1,
  cam: { stream: null, ready: false, label: '' },
  ai: { state: 'idle', embedder: null, error: '' },
  feats: { key: '', engine: '', ok: [], ng: [], tmpl: null },
  run: { on: false, timer: null, busy: false },
  last: null,
  counters: load('mv_counters', { ok: 0, ng: 0 }),
  history: [],
  logRows: [],
  logQueue: [],
  lastNgUpload: 0,
  motion: { prev: null, still: 0, moving: false, armed: true, bgSmall: null },
  server: { state: 'off', gemini: false },
  drawing: null,
  showFoundUntil: 0
};

function newProgram() {
  return {
    id: null, name: 'Program-' + new Date().toISOString().slice(0, 10),
    roi: { x: 0.35, y: 0.3, w: 0.3, h: 0.4 }, engine: 'ai', thr: null,
    track: true, margin: 0.25, bright: 100, contrast: 100, ok: [], ng: []
  };
}

const saveSettings = () => store('mv_settings', S.settings);

/* Konfigurasi otomatis: config.js (admin) + parameter link (?room=&station=&name=) */
(function autoConfig() {
  const C = window.MV_CONFIG || {}, st = S.settings, q = new URLSearchParams(location.search);
  if (!st.wkUrl && C.workerUrl) st.wkUrl = C.workerUrl.replace(/\/+$/, '');
  if ((!st.room || st.room === 'line-1') && C.room) st.room = C.room;
  if (q.get('room')) st.room = q.get('room');
  if (q.get('station')) st.stationId = q.get('station');
  if (q.get('name')) st.stationName = q.get('name');
  if (q.get('worker')) st.wkUrl = q.get('worker');
  st.room = String(st.room).toLowerCase().replace(/[^a-z0-9_-]+/g, '-').slice(0, 40) || 'line-1';
  st.stationId = String(st.stationId).toLowerCase().replace(/[^a-z0-9_-]+/g, '-').slice(0, 40);
  if (!st.stationName || st.stationName === 'Stasiun 1') st.stationName = 'Stasiun ' + st.stationId.replace(/^st-/, '').toUpperCase();
})();
const hasServer = () => !!(S.settings.gasUrl || S.settings.wkUrl);
const cleanId = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9_-]+/g, '-').slice(0, 40);
const saveProgramLocal = () => store('mv_program', S.program);

/* ============================== DOM REFS ============================== */

const video = document.createElement('video');
video.playsInline = true; video.muted = true;
const view = $('#view');
const vctx = view.getContext('2d');
const proc = document.createElement('canvas');
const pctx = proc.getContext('2d', { willReadFrequently: true });
const tmpC = document.createElement('canvas');
const tctx = tmpC.getContext('2d', { willReadFrequently: true });

/* ============================== UI HELPERS ============================== */

let toastTimer;
function toast(msg, kind = '') {
  const t = $('#toast');
  t.textContent = msg; t.className = 'toast ' + kind;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add('hidden'), 3500);
}
function chip(id, text, cls) {
  const el = $(id); el.textContent = text; el.className = 'chip ' + (cls || '');
}
function hint(text) {
  const h = $('#hint');
  if (!text) { h.classList.add('hidden'); return; }
  h.textContent = text; h.classList.remove('hidden');
}

function refreshChips() {
  chip('#chipCam', S.cam.ready ? 'Kamera aktif' : 'Kamera mati', S.cam.ready ? 'on' : 'err');
  const eng = activeEngine();
  const engTxt = eng === 'ai' ? 'AI siap' : (S.program.engine === 'ai' && S.ai.state === 'loading' ? 'AI memuat…' : 'Mesin Klasik');
  chip('#chipEngine', engTxt, eng === 'ai' ? 'on' : (S.ai.state === 'loading' ? 'warn' : ''));
  const sv = { off: ['Server: belum diatur', ''], ok: ['Server terhubung', 'on'], err: ['Server error', 'err'], busy: ['Server…', 'warn'] }[S.server.state];
  chip('#chipServer', sv[0], sv[1]);
  const rt = { off: ['Realtime: off', ''], connecting: ['Realtime…', 'warn'], on: ['Realtime ' + S.settings.room, 'on'], retry: ['Realtime terputus', 'err'], err: ['Realtime error', 'err'] }[RT.state];
  chip('#chipRt', rt[0], rt[1]);
  chip('#chipProg', S.program.name + (S.program.ok.length ? '' : ' (tanpa master)'), S.program.ok.length ? 'on' : 'warn');
}

/* ============================== STEPS ============================== */

function hint5() {
  if (RT.state === 'on') return `✓ Tersambung otomatis ke Cloudflare (line ${S.settings.room}). Hasil inspeksi tersimpan di Google Sheet/Drive.`;
  if (RT.state === 'connecting' || RT.state === 'retry') return 'Menghubungkan otomatis ke server…';
  return S.settings.wkUrl ? 'Realtime dimatikan — centang "Kirim hasil realtime" untuk menyalakan.' : 'Server belum diatur di config.js.';
}

function goStep(n) {
  n = clamp(parseInt(n, 10) || 1, 1, 5);
  S.step = n;
  $$('.step').forEach((b) => b.classList.toggle('active', +b.dataset.step === n));
  $$('.pane').forEach((p) => p.classList.toggle('hidden', +p.dataset.pane !== n));
  $$('.step').forEach((b) => {
    const s = +b.dataset.step;
    const done = (s === 1 && S.cam.ready) || (s === 2 && S.program.ok.length > 0) || (s === 3 && S.program.thr != null);
    b.classList.toggle('done', done && s !== n);
  });
  hint({
    1: 'Atur kamera & gambar. ROI (kotak kuning) dipakai untuk saran gambar otomatis.',
    2: 'Klik & tarik di gambar untuk membuat ROI, lalu daftarkan master.',
    3: S.program.ok.length ? 'Skor langsung dihitung — geser ambang sampai OK/NG sesuai.' : 'Daftarkan master dulu di langkah 2.',
    4: 'Tekan Mulai, atau ucapkan "mulai" / "periksa" ke asisten.',
    5: hint5()
  }[n]);
  if (n === 5 && hasServer() && S.server.state === 'ok') listPrograms();
}

/* ============================== CAMERA ============================== */

async function listCams() {
  try {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
    const sel = $('#selCam');
    sel.innerHTML = '<option value="">(default)</option>' +
      devs.map((d, i) => `<option value="${d.deviceId}">${d.label || 'Kamera ' + (i + 1)}</option>`).join('');
    sel.value = devs.some((d) => d.deviceId === S.settings.deviceId) ? S.settings.deviceId : '';
  } catch (e) { console.warn(e); }
}

async function startCam() {
  if (!navigator.mediaDevices?.getUserMedia) {
    toast('Browser tidak mendukung kamera. Pakai Chrome/Edge lewat HTTPS.', 'err'); return;
  }
  stopCam();
  const [w, h] = S.settings.res.split('x').map(Number);
  const video_ = { width: { ideal: w }, height: { ideal: h } };
  if (S.settings.deviceId) video_.deviceId = { exact: S.settings.deviceId };
  else video_.facingMode = 'environment';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: video_, audio: false });
    S.cam.stream = stream;
    S.cam.label = stream.getVideoTracks()[0]?.label || 'kamera';
    video.srcObject = stream;
    await video.play();
    await new Promise((r) => (video.videoWidth ? r() : video.addEventListener('loadedmetadata', r, { once: true })));
    view.width = video.videoWidth; view.height = video.videoHeight;
    S.cam.ready = true;
    $('#placeholder').classList.add('hidden');
    await listCams();
    toast(`Kamera aktif: ${video.videoWidth}×${video.videoHeight}`, 'ok');
  } catch (e) {
    S.cam.ready = false;
    $('#placeholder').classList.remove('hidden');
    toast('Gagal membuka kamera: ' + (e.message || e.name), 'err');
  }
  refreshChips(); goStep(S.step);
}

function stopCam() {
  if (S.cam.stream) S.cam.stream.getTracks().forEach((t) => t.stop());
  S.cam.stream = null; S.cam.ready = false;
}

function filterStr() { return `brightness(${S.program.bright}%) contrast(${S.program.contrast}%)`; }

/** Ambil frame kamera (dengan filter kecerahan/kontras) ke canvas proses. */
function grabFrame() {
  proc.width = video.videoWidth; proc.height = video.videoHeight;
  pctx.filter = filterStr();
  pctx.drawImage(video, 0, 0);
  pctx.filter = 'none';
  return proc;
}

/* ============================== GEOMETRY ============================== */

function roiPx(W = view.width, H = view.height) {
  const r = S.program.roi;
  return { x: Math.round(r.x * W), y: Math.round(r.y * H), w: Math.max(8, Math.round(r.w * W)), h: Math.max(8, Math.round(r.h * H)) };
}
function clipRect(r, W, H) {
  const x = clamp(Math.round(r.x), 0, W - 1), y = clamp(Math.round(r.y), 0, H - 1);
  return { x, y, w: clamp(Math.round(r.w), 1, W - x), h: clamp(Math.round(r.h), 1, H - y) };
}
function searchRect(r, W, H) {
  const m = S.program.track ? S.program.margin : 0;
  return clipRect({ x: r.x - r.w * m, y: r.y - r.h * m, w: r.w * (1 + 2 * m), h: r.h * (1 + 2 * m) }, W, H);
}

function cropCanvas(src, r, maxSide = 256) {
  const s = Math.min(1, maxSide / Math.max(r.w, r.h));
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(r.w * s)); c.height = Math.max(1, Math.round(r.h * s));
  c.getContext('2d').drawImage(src, r.x, r.y, r.w, r.h, 0, 0, c.width, c.height);
  return c;
}

/** Ambil grayscale (Float32Array) dari area sumber, diskalakan ke w×h. */
function grayOf(src, sx, sy, sw, sh, w, h) {
  tmpC.width = w; tmpC.height = h;
  tctx.drawImage(src, sx, sy, sw, sh, 0, 0, w, h);
  const d = tctx.getImageData(0, 0, w, h).data;
  const g = new Float32Array(w * h);
  for (let i = 0, j = 0; i < g.length; i++, j += 4) g[i] = 0.299 * d[j] + 0.587 * d[j + 1] + 0.114 * d[j + 2];
  return g;
}

/* ============================== FEATURES ============================== */

function znorm(a) {
  const n = a.length; let m = 0;
  for (let i = 0; i < n; i++) m += a[i];
  m /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (a[i] - m) ** 2;
  const sd = Math.sqrt(v) || 1;
  const o = new Float32Array(n);
  for (let i = 0; i < n; i++) o[i] = (a[i] - m) / sd;
  return o;
}
function l2(a) {
  let s = 0; for (const x of a) s += x * x;
  s = Math.sqrt(s) || 1;
  return Float32Array.from(a, (x) => x / s);
}
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

function sobel(g, w, h) {
  const o = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    const gx = -g[i - w - 1] - 2 * g[i - 1] - g[i + w - 1] + g[i - w + 1] + 2 * g[i + 1] + g[i + w + 1];
    const gy = -g[i - w - 1] - 2 * g[i - w] - g[i - w + 1] + g[i + w - 1] + 2 * g[i + w] + g[i + w + 1];
    o[i] = Math.hypot(gx, gy);
  }
  return o;
}

function featClassic(cnv) {
  const g = grayOf(cnv, 0, 0, cnv.width, cnv.height, 64, 64);
  return { i: znorm(g), g: znorm(sobel(g, 64, 64)) };
}
const simClassic = (a, b) => clamp((dot(a.i, b.i) + dot(a.g, b.g)) / 2, 0, 1);

function featAI(cnv) {
  const r = S.ai.embedder.embed(cnv);
  return l2(r.embeddings[0].floatEmbedding);
}
const simAI = (a, b) => clamp(dot(a, b), 0, 1);

function activeEngine() {
  return S.program.engine === 'ai' && S.ai.state === 'ready' ? 'ai' : 'classic';
}
const featureOf = (cnv) => (activeEngine() === 'ai' ? featAI(cnv) : featClassic(cnv));
const simOf = (a, b) => (activeEngine() === 'ai' ? simAI(a, b) : simClassic(a, b));

async function loadAI() {
  if (S.ai.state === 'ready' || S.ai.state === 'loading') return;
  S.ai.state = 'loading'; refreshChips(); engineInfo();
  try {
    const { ImageEmbedder, FilesetResolver } = await import(`${MP_BASE}/vision_bundle.mjs`);
    const fileset = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
    S.ai.embedder = await ImageEmbedder.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MP_MODEL },
      runningMode: 'IMAGE', l2Normalize: true, quantize: false
    });
    S.ai.state = 'ready';
    toast('Model AI (MobileNetV3) siap.', 'ok');
  } catch (e) {
    console.warn('AI load failed', e);
    S.ai.state = 'error'; S.ai.error = e.message || String(e);
    toast('Model AI gagal dimuat → memakai mesin Klasik.', 'err');
  }
  S.feats.key = '';
  await rebuildFeatures();
  refreshChips(); engineInfo();
}

function engineInfo() {
  const el = $('#oEngineState');
  const e = activeEngine();
  if (S.program.engine === 'ai' && S.ai.state === 'loading') el.textContent = 'Memuat model AI dari CDN (±3 MB)…';
  else if (S.program.engine === 'ai' && S.ai.state === 'error') el.textContent = 'Model AI gagal dimuat (' + S.ai.error.slice(0, 80) + '). Sementara memakai mesin Klasik.';
  else el.textContent = e === 'ai' ? 'Aktif: AI — kemiripan visual tingkat tinggi, tahan variasi cahaya ringan.' : 'Aktif: Klasik — cepat & ringan, peka pada pola dan tepi.';
}

function loadImage(src) {
  return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src; });
}
function imgToCanvas(im) {
  const c = document.createElement('canvas'); c.width = im.naturalWidth; c.height = im.naturalHeight;
  c.getContext('2d').drawImage(im, 0, 0); return c;
}

/** Hitung ulang fitur master/NG + template pelacakan bila data berubah. */
async function rebuildFeatures() {
  const eng = activeEngine();
  const key = eng + '|' + view.width + 'x' + view.height + '|' + S.program.ok.length + '|' + S.program.ng.length + '|' + (S.program.ok[0] || '').length + '|' + JSON.stringify(S.program.roi);
  if (key === S.feats.key) return;
  const okC = await Promise.all(S.program.ok.map((u) => loadImage(u).then(imgToCanvas)));
  const ngC = await Promise.all(S.program.ng.map((u) => loadImage(u).then(imgToCanvas)));
  S.feats = { key, engine: eng, ok: okC.map(featureOf), ng: ngC.map(featureOf), tmpl: okC[0] ? makeTemplate(okC[0]) : null };
}

/* ============================== POSITION TRACKING ============================== */

function trackScale(r) { return Math.min(1, 40 / Math.max(r.w, r.h)); }

function makeTemplate(masterCnv) {
  const r = roiPx(); const s = trackScale(r);
  const w = Math.max(4, Math.round(r.w * s)), h = Math.max(4, Math.round(r.h * s));
  return { w, h, s, data: znorm(grayOf(masterCnv, 0, 0, masterCnv.width, masterCnv.height, w, h)) };
}

function locate(frame) {
  const t = S.feats.tmpl;
  if (!t) return { dx: 0, dy: 0, conf: 0 };
  const W = frame.width, H = frame.height, r = roiPx(W, H), sr = searchRect(r, W, H);
  const sw = Math.max(t.w, Math.round(sr.w * t.s)), sh = Math.max(t.h, Math.round(sr.h * t.s));
  const g = grayOf(frame, sr.x, sr.y, sr.w, sr.h, sw, sh);
  const n = t.w * t.h; let best = -2, bx = 0, by = 0;
  const win = new Float32Array(n);
  for (let oy = 0; oy <= sh - t.h; oy++) {
    for (let ox = 0; ox <= sw - t.w; ox++) {
      let m = 0, k = 0;
      for (let y = 0; y < t.h; y++) { const row = (oy + y) * sw + ox; for (let x = 0; x < t.w; x++) { const v = g[row + x]; win[k++] = v; m += v; } }
      m /= n;
      let v2 = 0, c = 0;
      for (let i = 0; i < n; i++) { const d = win[i] - m; v2 += d * d; c += d * t.data[i]; }
      const score = c / (Math.sqrt(v2) || 1);
      if (score > best) { best = score; bx = ox; by = oy; }
    }
  }
  const fx = sr.x + bx * (sr.w / sw), fy = sr.y + by * (sr.h / sh);
  return { dx: Math.round(fx - r.x), dy: Math.round(fy - r.y), conf: best };
}

/* ============================== INSPECTION ============================== */

const currentThr = () => (S.program.thr != null ? S.program.thr : DEFAULT_THR[activeEngine()]);

async function evaluate() {
  const t0 = performance.now();
  const f = grabFrame();
  await rebuildFeatures();
  let pos = { dx: 0, dy: 0, conf: 0 };
  if (S.program.track) pos = locate(f);
  const r = roiPx(f.width, f.height);
  const rect = clipRect({ x: r.x + pos.dx, y: r.y + pos.dy, w: r.w, h: r.h }, f.width, f.height);
  const crop = cropCanvas(f, rect, 256);
  const feat = featureOf(crop);
  const score = Math.max(0, ...S.feats.ok.map((m) => simOf(feat, m)));
  const thr = currentThr();
  return { score, thr, result: score >= thr ? 'OK' : 'NG', dx: pos.dx, dy: pos.dy, ms: Math.round(performance.now() - t0), crop, rect };
}

async function inspect(trigger = 'manual') {
  if (!S.cam.ready) { toast('Kamera belum aktif.', 'err'); return null; }
  if (!S.program.ok.length) { toast('Daftarkan master dulu (langkah 2).', 'err'); return null; }
  if (S.run.busy) return null;
  S.run.busy = true;
  try {
    const ev = await evaluate();
    const thumb = ev.crop.toDataURL('image/jpeg', 0.7);
    S.last = { ...ev, crop: undefined, thumb, trigger, ts: new Date().toISOString() };
    S.showFoundUntil = performance.now() + 1500;
    if (ev.result === 'OK') S.counters.ok++; else S.counters.ng++;
    store('mv_counters', S.counters);
    S.history.unshift(S.last); S.history.length = Math.min(S.history.length, 30);
    renderResult(); renderHistory();
    if (ev.result === 'NG' && S.settings.beep) beep();
    recordLog(S.last);
    if (ev.result === 'NG' && S.settings.autoAsk && S.server.state === 'ok') {
      openAssist(); ask('Hasil barusan NG. Tolong analisis gambarnya dibanding master.', { withImages: true, silentUser: true });
    }
    return S.last;
  } catch (e) {
    console.error(e); toast('Inspeksi gagal: ' + e.message, 'err'); return null;
  } finally { S.run.busy = false; }
}

/* live preview skor di langkah 3 */
async function previewTick() {
  if (S.step !== 3 || S.run.on || !S.cam.ready || !S.program.ok.length || S.run.busy) return;
  S.run.busy = true;
  try { const ev = await evaluate(); S.preview = ev; renderScore(ev.score, ev.thr, ev.result); } catch (e) { console.warn(e); }
  finally { S.run.busy = false; }
}

/* ============================== RUN / TRIGGER ============================== */

function startRun() {
  if (!S.program.ok.length) { toast('Daftarkan master dulu (langkah 2).', 'err'); goStep(2); return false; }
  if (!S.cam.ready) { toast('Kamera belum aktif.', 'err'); return false; }
  stopRun(true);
  S.run.on = true;
  const mode = S.settings.trigger;
  if (mode === 'continuous') S.run.timer = setInterval(() => inspect('kontinu'), Math.max(200, S.settings.interval));
  else if (mode === 'ai') { resetMotion(); S.run.timer = setInterval(motionTick, 120); }
  else S.run.timer = null;
  $('#btnRun').textContent = '■ Stop'; $('#btnRun').classList.remove('primary'); $('#btnRun').classList.add('danger');
  toast(mode === 'manual' ? 'Mode manual: tekan Periksa atau ucapkan "periksa".' : 'Inspeksi berjalan.', 'ok');
  rtState();
  return true;
}
function stopRun(silent) {
  if (S.run.timer) clearInterval(S.run.timer);
  S.run.timer = null;
  const was = S.run.on; S.run.on = false;
  $('#btnRun').textContent = '▶ Mulai'; $('#btnRun').classList.add('primary'); $('#btnRun').classList.remove('danger');
  if (was && !silent) toast('Inspeksi dihentikan.');
  rtState();
}

function resetMotion() { Object.assign(S.motion, { prev: null, still: 0, moving: false, armed: true }); }

function smallSearch() {
  const W = video.videoWidth, H = video.videoHeight;
  const sr = searchRect(roiPx(W, H), W, H);
  return grayOf(video, sr.x, sr.y, sr.w, sr.h, 32, 24);
}
function meanAbs(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); return s / a.length; }

function motionTick() {
  if (!S.cam.ready) return;
  const cur = smallSearch(), m = S.motion;
  const diff = m.prev ? meanAbs(cur, m.prev) : 0;
  m.prev = cur;
  $('#motionBar').style.width = clamp(diff / (S.settings.move * 3) * 100, 0, 100) + '%';
  const present = m.bgSmall ? meanAbs(cur, m.bgSmall) > S.settings.move * 2 : true;
  if (diff > S.settings.move) { m.moving = true; m.still = 0; m.armed = true; return; }
  m.still++;
  if (m.armed && m.moving && m.still >= 4 && present) { m.armed = false; m.moving = false; inspect('trigger-ai'); }
}

/* ============================== AUTO IMAGING & THRESHOLD ============================== */

async function autoImaging() {
  if (!S.cam.ready) { toast('Kamera belum aktif.', 'err'); return; }
  if (S.program.ok.length && !confirm('Mengubah gambar setelah master didaftarkan bisa menurunkan skor. Lanjut?')) return;
  const W = video.videoWidth, H = video.videoHeight, r = roiPx(W, H);
  const s = Math.min(1, 160 / Math.max(r.w, r.h));
  const w = Math.max(8, Math.round(r.w * s)), h = Math.max(8, Math.round(r.h * s));
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const cx = c.getContext('2d', { willReadFrequently: true });
  let best = { score: -1 };
  for (const b of [70, 85, 100, 115, 130, 150]) {
    for (const k of [80, 100, 125, 150, 180]) {
      cx.filter = `brightness(${b}%) contrast(${k}%)`;
      cx.drawImage(video, r.x, r.y, r.w, r.h, 0, 0, w, h);
      const d = cx.getImageData(0, 0, w, h).data;
      const g = new Float32Array(w * h); let clip = 0, mean = 0;
      for (let i = 0, j = 0; i < g.length; i++, j += 4) {
        g[i] = 0.299 * d[j] + 0.587 * d[j + 1] + 0.114 * d[j + 2];
        if (g[i] < 4 || g[i] > 251) clip++;
        mean += g[i];
      }
      mean /= g.length;
      let lv = 0, lm = 0, cnt = 0;
      for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
        const i = y * w + x; const L = 4 * g[i] - g[i - 1] - g[i + 1] - g[i - w] - g[i + w];
        lm += L; lv += L * L; cnt++;
      }
      lm /= cnt; const sharp = Math.sqrt(lv / cnt - lm * lm);
      const clipFrac = clip / g.length;
      const score = sharp * (1 - Math.min(1, clipFrac * 4)) * (1 - Math.abs(mean - 128) / 256);
      if (score > best.score) best = { score, b, k };
      await sleep(0);
    }
  }
  cx.filter = 'none';
  S.program.bright = best.b; S.program.contrast = best.k;
  syncImgUI(); saveProgramLocal(); S.feats.key = '';
  toast(`Saran diterapkan: kecerahan ${best.b}%, kontras ${best.k}%.`, 'ok');
}

async function autoThreshold() {
  if (!S.program.ok.length) { toast('Daftarkan master dulu.', 'err'); return; }
  await rebuildFeatures();
  const ok = S.feats.ok, ng = S.feats.ng, eng = activeEngine();
  const okScores = ok.length > 1 ? ok.map((f, i) => Math.max(...ok.filter((_, j) => j !== i).map((m) => simOf(f, m)))) : [];
  const ngScores = ng.map((f) => Math.max(...ok.map((m) => simOf(f, m))));
  let thr, info;
  if (ngScores.length && okScores.length) {
    const minOk = Math.min(...okScores), maxNg = Math.max(...ngScores);
    thr = (minOk + maxNg) / 2;
    info = `OK terendah ${minOk.toFixed(3)} · NG tertinggi ${maxNg.toFixed(3)}` + (maxNg >= minOk ? ' ⚠ tumpang tindih — perbaiki ROI/pencahayaan.' : '');
  } else if (ngScores.length) {
    const maxNg = Math.max(...ngScores);
    thr = Math.min(0.97, maxNg + (1 - maxNg) * 0.4);
    info = `NG tertinggi ${maxNg.toFixed(3)} (tambah sampel OK agar lebih akurat).`;
  } else if (okScores.length) {
    const minOk = Math.min(...okScores);
    thr = minOk - (eng === 'ai' ? 0.05 : 0.08);
    info = `OK terendah ${minOk.toFixed(3)} (tambah sampel NG agar lebih akurat).`;
  } else {
    thr = DEFAULT_THR[eng];
    info = 'Hanya 1 master: memakai ambang bawaan. Tambah sampel OK/NG untuk kalibrasi.';
  }
  setThr(thr);
  $('#oThrInfo').textContent = info;
  toast(`Ambang otomatis: ${S.program.thr.toFixed(2)}`, 'ok');
}

function setThr(v) {
  v = Number(v);
  if (!isFinite(v)) return;
  if (v > 1) v = v / 100;
  S.program.thr = Math.round(clamp(v, 0.3, 0.99) * 100) / 100;
  $('#rThr').value = S.program.thr; $('#vThr').textContent = S.program.thr.toFixed(2);
  saveProgramLocal(); renderScore(S.last?.score, S.program.thr, S.last?.result);
}

/* ============================== SAMPLES & PROGRAM ============================== */

async function captureSample(kind = 'ok') {
  if (!S.cam.ready) { toast('Kamera belum aktif.', 'err'); return; }
  const list = S.program[kind];
  if (list.length >= MAX_SAMPLES) { toast(`Maksimal ${MAX_SAMPLES} sampel.`, 'err'); return; }
  const f = grabFrame();
  let rect = roiPx(f.width, f.height);
  if (kind === 'ng' || (kind === 'ok' && list.length)) {
    await rebuildFeatures();
    if (S.program.track && S.feats.tmpl) { const p = locate(f); rect = clipRect({ ...rect, x: rect.x + p.dx, y: rect.y + p.dy }, f.width, f.height); }
  }
  list.push(cropCanvas(f, rect, 256).toDataURL('image/jpeg', 0.85));
  saveProgramLocal(); S.feats.key = '';
  renderSamples(); refreshChips();
  toast(kind === 'ok' ? `Master OK #${list.length} terdaftar.` : `Sampel NG #${list.length} ditambahkan.`, 'ok');
}

function removeSample(kind, i) {
  S.program[kind].splice(i, 1);
  saveProgramLocal(); S.feats.key = ''; renderSamples(); refreshChips();
}

function setRoi(r) {
  if (S.program.ok.length || S.program.ng.length) {
    if (!confirm('Mengubah ROI akan menghapus semua sampel master/NG. Lanjut?')) return;
    S.program.ok = []; S.program.ng = [];
  }
  S.program.roi = r; S.motion.bgSmall = null;
  saveProgramLocal(); S.feats.key = ''; renderSamples(); refreshChips(); renderRoi();
}

function applyProgram(p) {
  S.program = { ...newProgram(), ...p };
  saveProgramLocal(); S.feats.key = '';
  S.motion.bgSmall = null;
  syncAllUI();
  if (S.program.engine === 'ai') loadAI(); else rebuildFeatures();
  toast(`Program "${S.program.name}" dimuat.`, 'ok');
  rtState();
}

/* ============================== RENDER ============================== */

function renderScore(score, thr, result) {
  const bar = $('#scorebar');
  $('#scoreFill').style.width = (score == null ? 0 : clamp(score, 0, 1) * 100) + '%';
  $('#scoreThr').style.left = clamp(thr ?? currentThr(), 0, 1) * 100 + '%';
  bar.className = 'scorebar ' + (result === 'OK' ? 'ok' : result === 'NG' ? 'ng' : '');
  $('#oScore').textContent = score == null ? '—' : score.toFixed(3);
  $('#lScore').textContent = 'Skor (amb ' + (thr ?? currentThr()).toFixed(2) + ')';
}

function renderResult() {
  const l = S.last; if (!l) return;
  const v = $('#verdict');
  v.textContent = l.result; v.className = 'verdict ' + l.result.toLowerCase();
  $('#kRes').className = 'kpi ' + l.result.toLowerCase();
  $('#oRes').textContent = l.result;
  renderScore(l.score, l.thr, l.result);
  $('#oPos').textContent = S.program.track ? `${l.dx}, ${l.dy}` : 'off';
  $('#oMs').textContent = l.ms + ' ms';
  renderCounters();
}
function renderCounters() {
  const { ok, ng } = S.counters, t = ok + ng;
  $('#oCount').textContent = `${ok} / ${ng}`;
  $('#oYield').textContent = t ? (ok / t * 100).toFixed(1) + '%' : '—';
}
function renderHistory() {
  $('#history').innerHTML = S.history.map((h) =>
    `<figure class="${h.result.toLowerCase()}"><img src="${h.thumb}" alt="${h.result}"><figcaption>${h.result} ${h.score.toFixed(2)}</figcaption></figure>`).join('');
}
function renderSamples() {
  for (const kind of ['ok', 'ng']) {
    const el = $(kind === 'ok' ? '#okList' : '#ngList');
    const list = S.program[kind];
    el.innerHTML = list.length ? list.map((u, i) => `<div class="s"><img src="${u}" alt=""><button data-kind="${kind}" data-i="${i}" title="Hapus">×</button></div>`).join('')
      : `<span class="empty">${kind === 'ok' ? 'Belum ada master.' : 'Belum ada sampel NG.'}</span>`;
  }
}
function renderRoi() {
  const r = S.program.roi;
  $('#oRoi').textContent = `x${(r.x * 100).toFixed(0)}% y${(r.y * 100).toFixed(0)}% · ${(r.w * 100).toFixed(0)}×${(r.h * 100).toFixed(0)}%`;
}
function syncImgUI() {
  $('#rBright').value = S.program.bright; $('#vBright').textContent = S.program.bright + '%';
  $('#rContrast').value = S.program.contrast; $('#vContrast').textContent = S.program.contrast + '%';
}
function syncAllUI() {
  const st = S.settings, p = S.program;
  $('#inpProgName').value = p.name;
  syncImgUI(); renderRoi(); renderSamples(); renderCounters();
  $$('input[name=engine]').forEach((r) => (r.checked = r.value === p.engine));
  $('#chkTrack').checked = p.track;
  $('#rMargin').value = Math.round(p.margin * 100); $('#vMargin').textContent = Math.round(p.margin * 100) + '%';
  const thr = p.thr ?? DEFAULT_THR[p.engine === 'ai' ? 'ai' : 'classic'];
  $('#rThr').value = thr; $('#vThr').textContent = thr.toFixed(2);
  $$('input[name=trig]').forEach((r) => (r.checked = r.value === st.trigger));
  $('#inpInterval').value = st.interval;
  $('#rMove').value = st.move; $('#vMove').textContent = st.move;
  $('#wrapInterval').classList.toggle('hidden', st.trigger !== 'continuous');
  $('#wrapAiTrig').classList.toggle('hidden', st.trigger !== 'ai');
  $('#oBg').textContent = S.motion.bgSmall ? 'terdaftar (sesi ini)' : 'belum ada (opsional)';
  $('#chkBeep').checked = st.beep; $('#chkAutoAsk').checked = st.autoAsk;
  $('#inpGas').value = st.gasUrl; $('#inpToken').value = st.token;
  $('#inpWk').value = st.wkUrl; $('#inpRoom').value = cleanId(st.room) || 'line-1'; $('#inpStId').value = st.stationId; $('#inpStName').value = st.stationName;
  $('#chkRt').checked = st.rt;
  $('#chkLog').checked = st.log; $('#chkSaveNg').checked = st.saveNg; $('#chkTts').checked = st.tts;
  $('#selRes').value = st.res;
  refreshChips(); engineInfo(); renderScore(S.last?.score, null, S.last?.result);
}

/** Loop gambar kamera + overlay. */
function drawLoop() {
  if (S.cam.ready && video.readyState >= 2) {
    const W = view.width, H = view.height;
    vctx.filter = filterStr();
    vctx.drawImage(video, 0, 0, W, H);
    vctx.filter = 'none';
    const lw = Math.max(2, W / 400), r = roiPx(W, H);
    if (S.program.track && S.program.ok.length) {
      const sr = searchRect(r, W, H);
      vctx.setLineDash([lw * 4, lw * 3]); vctx.strokeStyle = 'rgba(56,189,248,.7)'; vctx.lineWidth = lw;
      vctx.strokeRect(sr.x, sr.y, sr.w, sr.h); vctx.setLineDash([]);
    }
    vctx.strokeStyle = '#ffb000'; vctx.lineWidth = lw; vctx.strokeRect(r.x, r.y, r.w, r.h);
    if (S.last && performance.now() < S.showFoundUntil) {
      vctx.strokeStyle = S.last.result === 'OK' ? '#22c55e' : '#ef4444'; vctx.lineWidth = lw * 2;
      vctx.strokeRect(r.x + S.last.dx, r.y + S.last.dy, r.w, r.h);
    }
    if (S.drawing) {
      const d = S.drawing;
      vctx.strokeStyle = '#fff'; vctx.setLineDash([lw * 3, lw * 2]); vctx.lineWidth = lw;
      vctx.strokeRect(Math.min(d.x0, d.x1) * W, Math.min(d.y0, d.y1) * H, Math.abs(d.x1 - d.x0) * W, Math.abs(d.y1 - d.y0) * H);
      vctx.setLineDash([]);
    }
    vctx.font = `600 ${Math.round(lw * 7)}px IBM Plex Mono, monospace`;
    vctx.fillStyle = '#ffb000'; vctx.fillText('ROI', r.x + lw * 2, r.y - lw * 3);
  }
  requestAnimationFrame(drawLoop);
}

/* ROI drawing on canvas */
function normPt(e) {
  const b = view.getBoundingClientRect();
  return { x: clamp((e.clientX - b.left) / b.width, 0, 1), y: clamp((e.clientY - b.top) / b.height, 0, 1) };
}
view.addEventListener('pointerdown', (e) => {
  if (!S.cam.ready || S.step !== 2) { if (S.cam.ready && S.step !== 2) toast('Buka langkah 2 untuk mengubah ROI.'); return; }
  const p = normPt(e); S.drawing = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
  view.setPointerCapture(e.pointerId);
});
view.addEventListener('pointermove', (e) => { if (S.drawing) { const p = normPt(e); S.drawing.x1 = p.x; S.drawing.y1 = p.y; } });
view.addEventListener('pointerup', () => {
  const d = S.drawing; S.drawing = null; if (!d) return;
  const r = { x: Math.min(d.x0, d.x1), y: Math.min(d.y0, d.y1), w: Math.abs(d.x1 - d.x0), h: Math.abs(d.y1 - d.y0) };
  if (r.w * view.width < 20 || r.h * view.height < 20) { toast('ROI terlalu kecil.'); return; }
  setRoi(r);
});

/* ============================== SOUND ============================== */

let audioCtx;
function beep() {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const o = audioCtx.createOscillator(), g = audioCtx.createGain();
    o.type = 'square'; o.frequency.value = 880; g.gain.value = 0.08;
    o.connect(g).connect(audioCtx.destination);
    o.start(); o.stop(audioCtx.currentTime + 0.25);
  } catch { /* abaikan */ }
}

/* ============================== SERVER (GAS) ============================== */

async function gas(action, data = {}) {
  const url = S.settings.wkUrl ? S.settings.wkUrl.replace(/\/+$/, '') + '/gas' : S.settings.gasUrl;
  if (!url) throw new Error('Isi URL Worker Cloudflare atau URL GAS (langkah 5).');
  const res = await fetch(url, {
    method: 'POST',
    body: JSON.stringify({ action, ...data, token: S.settings.token }) // text/plain → tanpa preflight CORS
  });
  if (!res.ok) throw new Error('HTTP ' + res.status);
  const j = await res.json();
  if (!j.ok) throw new Error(j.error || 'Server menolak permintaan.');
  return j;
}

async function ping() {
  S.server.state = 'busy'; refreshChips(); $('#oPing').textContent = 'Menghubungi…';
  try {
    const j = await gas('ping');
    S.server = { state: 'ok', gemini: j.gemini };
    $('#oPing').textContent = `✓ Terhubung · Gemini ${j.gemini ? 'siap (' + j.model + ')' : 'BELUM diisi'}`;
    toast('Server GAS terhubung.', 'ok');
  } catch (e) {
    S.server.state = 'err'; $('#oPing').textContent = '✗ ' + e.message; toast('Koneksi gagal: ' + e.message, 'err');
  }
  refreshChips();
}

function recordLog(l) {
  const row = {
    ts: l.ts, program: S.program.name, result: l.result, score: l.score, thr: l.thr,
    dx: l.dx, dy: l.dy, ms: l.ms, trigger: l.trigger, image: '', device: S.settings.stationName
  };
  S.logRows.push(row); if (S.logRows.length > 5000) S.logRows.shift();
  if (rtSendResult(l, row)) return; // realtime aktif → Cloudflare yang meneruskan ke Google Sheet/Drive
  if (!S.settings.log || !hasServer()) return;
  const now = Date.now();
  if (l.result === 'NG' && S.settings.saveNg && now - S.lastNgUpload > 3000) {
    S.lastNgUpload = now;
    gas('saveImage', { dataUrl: l.thumb, program: S.program.name, result: l.result, ts: l.ts })
      .then((j) => { row.image = j.url; }).catch((e) => console.warn('saveImage', e))
      .finally(() => S.logQueue.push(row));
  } else S.logQueue.push(row);
}

async function flushLog() {
  if (!S.logQueue.length || !hasServer() || !S.settings.log) return;
  const rows = S.logQueue.splice(0, 200);
  try { await gas('log', { rows }); if (S.server.state !== 'ok') { S.server.state = 'ok'; refreshChips(); } }
  catch (e) { S.logQueue.unshift(...rows); S.server.state = 'err'; refreshChips(); console.warn('log', e); }
}
setInterval(flushLog, 5000);

function downloadCsv() {
  if (!S.logRows.length) { toast('Belum ada data log di sesi ini.'); return; }
  const head = ['Waktu', 'Program', 'Hasil', 'Skor', 'Ambang', 'dX', 'dY', 'ms', 'Trigger', 'Gambar'];
  const lines = S.logRows.map((r) => [r.ts, r.program, r.result, r.score.toFixed(4), r.thr.toFixed(2), r.dx, r.dy, r.ms, r.trigger, r.image]
    .map((v) => `"${String(v ?? '').replace(/"/g, '""')}"`).join(','));
  const blob = new Blob(['﻿' + [head.join(','), ...lines].join('\n')], { type: 'text/csv' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = `mapin-vision-log-${new Date().toISOString().slice(0, 10)}.csv`; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function showStats() {
  try {
    const j = await gas('stats');
    const parts = Object.entries(j.byProgram).map(([k, v]) => `${k}: ${v.ok} OK / ${v.ng} NG`);
    $('#oStats').textContent = `${j.date} · total ${j.total} (${j.okCount} OK, ${j.ng} NG)` + (parts.length ? ' · ' + parts.join(' · ') : '');
  } catch (e) { $('#oStats').textContent = '✗ ' + e.message; }
}

async function saveProgram() {
  S.program.name = ($('#inpProgName').value || S.program.name).trim();
  saveProgramLocal(); refreshChips();
  if (!S.program.ok.length) { toast('Program belum punya master.', 'err'); return; }
  if (!hasServer()) { toast('Tersimpan di browser ini. Isi URL Worker/GAS untuk simpan ke server.'); return; }
  try {
    const j = await gas('saveProgram', { program: S.program });
    S.program.id = j.id; saveProgramLocal();
    toast('Program tersimpan di server.', 'ok'); listPrograms();
  } catch (e) { toast('Gagal simpan: ' + e.message, 'err'); }
}

async function listPrograms() {
  const el = $('#progList');
  el.innerHTML = '<span class="empty">Memuat…</span>';
  try {
    const j = await gas('listPrograms');
    el.innerHTML = j.programs.length ? j.programs.map((p) => `
      <div class="p"><div><b>${escapeHtml(p.name)}</b><span class="small muted">${p.engine || '-'} · ambang ${p.thr === '' ? 'bawaan' : p.thr} · ${p.nOk} OK/${p.nNg} NG</span></div>
      <button class="btn" data-load="${p.id}">Muat</button><button class="btn danger" data-del="${p.id}" title="Hapus">×</button></div>`).join('')
      : '<span class="empty">Belum ada program di server.</span>';
  } catch (e) { el.innerHTML = `<span class="empty">✗ ${escapeHtml(e.message)}</span>`; }
}

function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

/* ============================== REALTIME (Cloudflare Worker) ============================== */

const RT = { ws: null, state: 'off', retry: 0, reconnectT: null, pingT: null, seq: 0 };
const rtOpen = () => !!RT.ws && RT.ws.readyState === 1;

function rtUrl() {
  const base = S.settings.wkUrl.replace(/\/+$/, '').replace(/^http/, 'ws');
  const q = new URLSearchParams({ room: S.settings.room || 'line-1', role: 'station', id: S.settings.stationId, name: S.settings.stationName, token: S.settings.token });
  return `${base}/ws?${q}`;
}

function rtConnect() {
  rtClose();
  if (!S.settings.wkUrl || !S.settings.rt) { RT.state = 'off'; refreshChips(); rtInfo(); return; }
  RT.state = 'connecting'; refreshChips(); rtInfo();
  let ws;
  try { ws = new WebSocket(rtUrl()); } catch (e) { RT.state = 'err'; refreshChips(); rtInfo(); return; }
  RT.ws = ws;
  ws.onopen = () => {
    RT.state = 'on'; RT.retry = 0; refreshChips(); rtInfo(); rtState('hello');
    RT.pingT = setInterval(() => ws.readyState === 1 && ws.send('ping'), 25000);
  };
  ws.onmessage = (e) => {
    if (e.data === 'pong') return;
    let m; try { m = JSON.parse(e.data); } catch { return; }
    if (m.t === 'cmd') onRemoteCmd(m);
  };
  ws.onclose = () => {
    clearInterval(RT.pingT);
    if (RT.ws !== ws) return;
    RT.ws = null;
    RT.state = S.settings.rt && S.settings.wkUrl ? 'retry' : 'off';
    refreshChips(); rtInfo();
    if (RT.state === 'retry') RT.reconnectT = setTimeout(rtConnect, Math.min(30000, 1000 * 2 ** RT.retry++));
  };
}
function rtClose() {
  clearTimeout(RT.reconnectT); clearInterval(RT.pingT);
  if (RT.ws) { const w = RT.ws; RT.ws = null; try { w.close(); } catch { /* */ } }
}
function rtSend(o) { if (!rtOpen()) return false; RT.ws.send(JSON.stringify(o)); return true; }
function rtState(t = 'state') {
  rtSend({ t, name: S.settings.stationName, program: S.program.name, running: S.run.on, trigger: S.settings.trigger, engine: activeEngine() });
}
function rtSendResult(l, row) {
  return rtSend({ t: 'result', seq: ++RT.seq, r: { ...row, thumb: l.thumb }, saveNg: S.settings.saveNg, log: S.settings.log });
}
async function onRemoteCmd(m) {
  if (!['start_run', 'stop_run', 'inspect_once', 'reset_counter'].includes(m.cmd)) return;
  const msg = await ACTIONS[m.cmd]();
  toast('Perintah dari monitor: ' + msg);
}
function rtInfo() {
  const el = $('#oRt'); if (!el) return;
  el.textContent = {
    off: S.settings.wkUrl ? 'Realtime dimatikan.' : 'Isi URL Worker untuk mengaktifkan realtime.',
    connecting: 'Menghubungkan…',
    on: `✓ Terhubung · line "${S.settings.room}" · ${S.settings.stationName} (${S.settings.stationId})`,
    retry: 'Terputus — mencoba lagi otomatis… (log sementara dikirim langsung ke GAS)',
    err: 'Gagal membuka koneksi. Cek URL Worker.'
  }[RT.state];
  if (S.step === 5) hint(hint5());
  const a = $('#lnkMonitor'); if (a) a.href = 'monitor.html?room=' + encodeURIComponent(S.settings.room || 'line-1');
}
function rtFields() {
  S.settings.wkUrl = $('#inpWk').value.trim().replace(/\/+$/, '');
  S.settings.room = cleanId($('#inpRoom').value) || 'line-1';
  S.settings.stationId = cleanId($('#inpStId').value) || S.settings.stationId;
  S.settings.stationName = $('#inpStName').value.trim() || S.settings.stationName;
  S.settings.rt = $('#chkRt').checked;
  S.settings.token = $('#inpToken').value.trim();
  saveSettings();
  $('#inpRoom').value = S.settings.room; $('#inpStId').value = S.settings.stationId;
}

/* ============================== ASSISTANT ============================== */

const chat = { history: [], rec: null, listening: false };

function openAssist() { $('#assist').classList.remove('hidden'); if (!$('#msgs').children.length) greet(); }
function closeAssist() { $('#assist').classList.add('hidden'); }

function greet() {
  addMsg('assistant', 'Halo! Saya asisten MAP-IN Vision. Tanya cara pakai, minta analisis hasil NG, atau beri perintah seperti "mulai", "stop", "periksa", "daftarkan master".');
}

function addMsg(role, text, extra = {}) {
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  el.textContent = text;
  if (extra.action && extra.action !== 'none') {
    const a = document.createElement('span'); a.className = 'act';
    a.textContent = '▶ aksi: ' + extra.action + (extra.value ? ' (' + extra.value + ')' : ''); el.appendChild(a);
  }
  (extra.images || []).forEach((src) => { const im = document.createElement('img'); im.src = src; el.appendChild(im); });
  $('#msgs').appendChild(el); $('#msgs').scrollTop = 1e9;
  return el;
}

function speak(text) {
  if (!S.settings.tts || !window.speechSynthesis) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text.replace(/[*_#`>]/g, ''));
  u.lang = 'id-ID';
  const v = speechSynthesis.getVoices().find((x) => x.lang?.toLowerCase().startsWith('id'));
  if (v) u.voice = v;
  u.rate = 1.05;
  speechSynthesis.speak(u);
}

const ACTIONS = {
  start_run: () => (startRun() ? 'Inspeksi dimulai.' : 'Belum bisa mulai.'),
  stop_run: () => { stopRun(); return 'Inspeksi dihentikan.'; },
  inspect_once: async () => { const r = await inspect('asisten'); return r ? `Hasil ${r.result}, skor ${r.score.toFixed(2)}.` : 'Belum bisa memeriksa.'; },
  capture_master: async () => { await captureSample('ok'); return 'Master didaftarkan.'; },
  capture_ng: async () => { await captureSample('ng'); return 'Sampel NG ditambahkan.'; },
  set_threshold: (v) => { setThr(v); return `Ambang diatur ke ${S.program.thr?.toFixed(2)}.`; },
  auto_threshold: async () => { await autoThreshold(); return `Ambang otomatis ${S.program.thr?.toFixed(2)}.`; },
  auto_imaging: async () => { await autoImaging(); return 'Saran gambar diterapkan.'; },
  goto_step: (v) => { goStep(v); return `Pindah ke langkah ${S.step}.`; },
  save_program: async () => { await saveProgram(); return 'Program disimpan.'; },
  reset_counter: () => { resetCounters(); return 'Hitungan direset.'; }
};

/** Perintah suara cepat — diproses lokal tanpa API (gratis & instan). */
function localCommand(t) {
  const s = t.toLowerCase().replace(/[.!?,]/g, '').trim();
  if (/^(mulai|start|jalankan)( inspeksi)?$/.test(s)) return 'start_run';
  if (/^(stop|berhenti|hentikan)( inspeksi)?$/.test(s)) return 'stop_run';
  if (/^(periksa|cek|inspeksi)( sekarang| sekali)?$/.test(s)) return 'inspect_once';
  if (/^daftar(kan)? master$/.test(s)) return 'capture_master';
  if (/^reset( hitungan)?$/.test(s)) return 'reset_counter';
  return null;
}

function appContext() {
  const l = S.last;
  return {
    langkahAktif: S.step,
    kamera: S.cam.ready ? `${S.cam.label} ${video.videoWidth}x${video.videoHeight}` : 'mati',
    program: S.program.name,
    mesinDipilih: S.program.engine, mesinAktif: activeEngine(), statusModelAI: S.ai.state,
    ambang: currentThr(), ambangManual: S.program.thr != null,
    jumlahMasterOK: S.program.ok.length, jumlahSampelNG: S.program.ng.length,
    pelacakanPosisi: S.program.track, areaCari: S.program.margin,
    kecerahan: S.program.bright, kontras: S.program.contrast,
    modeTrigger: S.settings.trigger, sedangBerjalan: S.run.on,
    hitungan: S.counters,
    hasilTerakhir: l ? { hasil: l.result, skor: +l.score.toFixed(3), ambang: l.thr, dX: l.dx, dY: l.dy, ms: l.ms, trigger: l.trigger, waktu: l.ts } : null,
    riwayat10: S.history.slice(0, 10).map((h) => `${h.result}:${h.score.toFixed(2)}`).join(' ')
  };
}

async function ask(text, opts = {}) {
  text = (text || '').trim(); if (!text) return;
  const wantImg = opts.withImages || $('#chkSendImg').checked || /gambar|lihat|foto|kenapa ng|analisis/i.test(text);
  const images = [];
  if (wantImg) {
    if (S.program.ok[0]) images.push({ label: 'master OK', data: S.program.ok[0] });
    if (S.last?.thumb) images.push({ label: `hasil terakhir (${S.last.result}, skor ${S.last.score.toFixed(2)})`, data: S.last.thumb });
    else if (S.cam.ready) images.push({ label: 'kamera saat ini (ROI)', data: cropCanvas(grabFrame(), roiPx(proc.width, proc.height), 384).toDataURL('image/jpeg', 0.8) });
  }
  if (!opts.silentUser) addMsg('user', text, { images: images.slice(1).map((i) => i.data) });

  const cmd = localCommand(text);
  if (cmd) {
    const reply = await ACTIONS[cmd]();
    addMsg('assistant', reply, { action: cmd }); speak(reply);
    chat.history.push({ role: 'user', text }, { role: 'assistant', text: reply });
    return;
  }
  if (!hasServer()) {
    const m = 'Untuk pertanyaan bebas, saya butuh backend (Worker/GAS + Gemini) di langkah 5. Perintah cepat tetap bisa: mulai, stop, periksa, daftarkan master, reset.';
    addMsg('assistant', m); speak(m); return;
  }
  const typing = addMsg('sys', 'Asisten sedang berpikir…');
  $('#oAssistState').textContent = 'Menghubungi Gemini…';
  try {
    const j = await gas('chat', { message: text, history: chat.history.slice(-8), context: appContext(), images });
    typing.remove();
    let reply = j.reply || '(tanpa jawaban)';
    let actMsg = '';
    if (j.action && j.action !== 'none' && ACTIONS[j.action]) actMsg = await ACTIONS[j.action](j.value);
    addMsg('assistant', reply, { action: j.action, value: j.value });
    speak(reply + (actMsg ? ' ' + actMsg : ''));
    chat.history.push({ role: 'user', text }, { role: 'assistant', text: reply });
    $('#oAssistState').textContent = 'Model: ' + j.model;
  } catch (e) {
    typing.remove();
    addMsg('sys', '✗ ' + e.message);
    $('#oAssistState').textContent = 'Gagal — cek langkah 5';
  }
}

function toggleMic() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { toast('Browser ini belum mendukung pengenalan suara. Pakai Chrome/Edge.', 'err'); return; }
  if (chat.listening) { chat.rec?.stop(); return; }
  const rec = new SR();
  rec.lang = 'id-ID'; rec.interimResults = true; rec.continuous = false; rec.maxAlternatives = 1;
  chat.rec = rec;
  rec.onstart = () => { chat.listening = true; $('#btnMic').classList.add('rec'); $('#oAssistState').textContent = 'Mendengarkan…'; speechSynthesis?.cancel(); };
  rec.onresult = (e) => {
    let interim = '', final = '';
    for (const r of e.results) (r.isFinal ? (final += r[0].transcript) : (interim += r[0].transcript));
    $('#inpChat').value = final || interim;
    if (final) { $('#inpChat').value = ''; ask(final); }
  };
  rec.onerror = (e) => { toast('Mic: ' + e.error, 'err'); };
  rec.onend = () => { chat.listening = false; $('#btnMic').classList.remove('rec'); if ($('#oAssistState').textContent === 'Mendengarkan…') $('#oAssistState').textContent = 'Tekan 🎙 lalu bicara'; };
  try { rec.start(); } catch (e) { toast('Mic gagal: ' + e.message, 'err'); }
}

function resetCounters() {
  S.counters = { ok: 0, ng: 0 }; store('mv_counters', S.counters);
  S.history = []; renderHistory(); renderCounters();
}

/* ============================== EVENTS ============================== */

$('#steps').addEventListener('click', (e) => { const b = e.target.closest('.step'); if (b) goStep(b.dataset.step); });
$('#btnCamStart').onclick = startCam;
$('#btnCamRestart').onclick = () => { S.settings.deviceId = $('#selCam').value; S.settings.res = $('#selRes').value; saveSettings(); startCam(); };
$('#rBright').oninput = (e) => { S.program.bright = +e.target.value; $('#vBright').textContent = S.program.bright + '%'; };
$('#rContrast').oninput = (e) => { S.program.contrast = +e.target.value; $('#vContrast').textContent = S.program.contrast + '%'; };
$('#rBright').onchange = $('#rContrast').onchange = () => { saveProgramLocal(); S.feats.key = ''; };
$('#btnAutoImg').onclick = autoImaging;
$('#btnImgReset').onclick = () => { S.program.bright = 100; S.program.contrast = 100; syncImgUI(); saveProgramLocal(); };

$('#inpProgName').onchange = (e) => { S.program.name = e.target.value.trim() || S.program.name; saveProgramLocal(); refreshChips(); };
$('#btnNewProg').onclick = () => {
  if (S.program.ok.length && !confirm('Buat program baru? Program sekarang (jika belum disimpan) akan hilang dari browser.')) return;
  stopRun(true); applyProgram(newProgram()); resetCounters();
};
$('#btnRoiFull').onclick = () => setRoi({ x: 0.35, y: 0.3, w: 0.3, h: 0.4 });
$('#btnCapOk').onclick = () => captureSample('ok');
$('#btnCapNg').onclick = () => captureSample('ng');
document.addEventListener('click', (e) => {
  const b = e.target.closest('.samples .s button'); if (b) removeSample(b.dataset.kind, +b.dataset.i);
});

$$('input[name=engine]').forEach((r) => r.addEventListener('change', async (e) => {
  S.program.engine = e.target.value; saveProgramLocal(); S.feats.key = '';
  if (S.program.engine === 'ai') await loadAI();
  if (S.program.thr == null) { const d = DEFAULT_THR[activeEngine()]; $('#rThr').value = d; $('#vThr').textContent = d.toFixed(2); }
  refreshChips(); engineInfo();
}));
$('#chkTrack').onchange = (e) => { S.program.track = e.target.checked; saveProgramLocal(); };
$('#rMargin').oninput = (e) => { S.program.margin = +e.target.value / 100; $('#vMargin').textContent = e.target.value + '%'; };
$('#rMargin').onchange = saveProgramLocal;
$('#rThr').oninput = (e) => setThr(e.target.value);
$('#btnAutoThr').onclick = autoThreshold;

$$('input[name=trig]').forEach((r) => r.addEventListener('change', (e) => {
  S.settings.trigger = e.target.value; saveSettings(); syncAllUI();
  if (S.run.on) startRun();
}));
$('#inpInterval').onchange = (e) => { S.settings.interval = clamp(+e.target.value || 800, 200, 10000); saveSettings(); if (S.run.on) startRun(); };
$('#rMove').oninput = (e) => { S.settings.move = +e.target.value; $('#vMove').textContent = e.target.value; saveSettings(); };
$('#btnBg').onclick = () => {
  if (!S.cam.ready) return;
  S.motion.bgSmall = smallSearch();
  $('#oBg').textContent = 'terdaftar (sesi ini)'; toast('Latar kosong terdaftar. Trigger hanya jalan bila ada benda.', 'ok');
};
$('#btnRun').onclick = () => (S.run.on ? stopRun() : startRun());
$('#btnOnce').onclick = () => inspect('manual');
$('#btnResetCnt').onclick = resetCounters;
$('#btnAskNg').onclick = () => { openAssist(); ask('Tolong analisis hasil inspeksi terakhir dibanding master. Lihat gambarnya.', { withImages: true }); };
$('#chkBeep').onchange = (e) => { S.settings.beep = e.target.checked; saveSettings(); };
$('#chkAutoAsk').onchange = (e) => { S.settings.autoAsk = e.target.checked; saveSettings(); };

$('#inpGas').onchange = (e) => { S.settings.gasUrl = e.target.value.trim(); saveSettings(); S.server.state = S.settings.gasUrl ? 'busy' : 'off'; refreshChips(); };
$('#inpToken').onchange = (e) => { S.settings.token = e.target.value.trim(); saveSettings(); };
$('#btnPing').onclick = () => { S.settings.gasUrl = $('#inpGas').value.trim(); rtFields(); ping(); };
$('#btnRt').onclick = () => { rtFields(); rtConnect(); if (S.settings.wkUrl) ping(); };
$('#chkRt').onchange = () => { rtFields(); rtConnect(); };
$('#inpStName').onchange = () => { rtFields(); rtState(); };
$('#chkLog').onchange = (e) => { S.settings.log = e.target.checked; saveSettings(); };
$('#chkSaveNg').onchange = (e) => { S.settings.saveNg = e.target.checked; saveSettings(); };
$('#chkTts').onchange = (e) => { S.settings.tts = e.target.checked; saveSettings(); };
$('#btnCsv').onclick = downloadCsv;
$('#btnStats').onclick = showStats;
$('#btnSaveProg').onclick = saveProgram;
$('#btnListProg').onclick = listPrograms;
$('#progList').addEventListener('click', async (e) => {
  const ld = e.target.closest('[data-load]'), del = e.target.closest('[data-del]');
  try {
    if (ld) { const j = await gas('getProgram', { id: ld.dataset.load }); stopRun(true); applyProgram(j.program); }
    if (del && confirm('Hapus program ini dari server?')) { await gas('deleteProgram', { id: del.dataset.del }); listPrograms(); }
  } catch (err) { toast(err.message, 'err'); }
});

$('#btnAssist').onclick = () => ($('#assist').classList.contains('hidden') ? openAssist() : closeAssist());
$('#btnAssistClose').onclick = closeAssist;
$('#btnMic').onclick = toggleMic;
$('#btnSend').onclick = () => { const t = $('#inpChat').value; $('#inpChat').value = ''; ask(t); };
$('#inpChat').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('#btnSend').click(); });
$$('.quick button').forEach((b) => (b.onclick = () => ask(b.dataset.q)));

document.addEventListener('keydown', (e) => {
  if (e.target.matches('input, select, textarea')) return;
  if (e.code === 'Space') { e.preventDefault(); inspect('manual'); }
});

/* ============================== INIT ============================== */

syncAllUI();
goStep(1);
drawLoop();
setInterval(previewTick, 300);
if (S.program.engine === 'ai') loadAI(); else rebuildFeatures();
saveSettings();
if (hasServer()) ping();
rtConnect();
startCam();
window.speechSynthesis?.getVoices();
window.MV = S; window.RT = RT; // untuk debug di console
