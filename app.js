/* Signal Spotter — offline traffic light detection.
 *
 * Pipeline per frame:
 *   1. SSDLite MobileNet v2 (COCO, TensorFlow.js) finds "traffic light" boxes.
 *      Optionally a second, zoomed pass over the upper-center of the view
 *      catches distant lights that are too small at full-frame resolution.
 *   2. Each box is cropped from the full-resolution frame and its lit lamp is
 *      classified red / yellow / green from HSV color, with the lamp's position
 *      in the housing (top/middle/bottom) as a tie-breaker.
 *   3. A short temporal vote stabilises the result before it is announced
 *      visually, by voice (speechSynthesis) and by beep tones (Web Audio).
 */
(() => {
  'use strict';

  const MODEL_URL = 'models/ssdlite_mobilenet_v2/model.json';
  const TRAFFIC_LIGHT_COL = 9;          // COCO id 10 → score column 9
  const MODEL_INPUT = 300;
  const COLORS = { red: '#ff3b30', yellow: '#ffcc00', green: '#30d158', unknown: '#a1a1a6' };
  const SPOKEN = { red: 'Red light', yellow: 'Yellow light', green: 'Green light' };
  const LOST_AFTER_MS = 1500;           // forget the light after this long without a sighting
  const BOX_HOLD_MS = 350;              // keep drawing a box briefly to avoid flicker

  const $ = (id) => document.getElementById(id);
  const els = {
    stage: $('stage'), video: $('video'), photo: $('photo'), overlay: $('overlay'), edge: $('edge'),
    status: $('status'), banner: $('banner'), bannerText: $('bannerText'),
    start: $('start'), btnStart: $('btnStart'), fileInput: $('fileInput'),
    loadBar: $('loadBar'), loadText: $('loadText'),
    settings: $('settings'), btnSettings: $('btnSettings'), btnCloseSettings: $('btnCloseSettings'),
    btnChangeSource: $('btnChangeSource'),
    optVoice: $('optVoice'), optBeep: $('optBeep'), optRepeat: $('optRepeat'), optZoom: $('optZoom'),
    optSens: $('optSens'), sensVal: $('sensVal'), optDebug: $('optDebug'),
    btnVolume: $('btnVolume'), volPanel: $('volPanel'), optVolume: $('optVolume'), volVal: $('volVal'),
    btnTestSound: $('btnTestSound'), btnMuteToggle: $('btnMuteToggle'),
    lensBar: $('lensBar'), lensButtons: $('lensButtons'), zoomRow: $('zoomRow'),
    optCamZoom: $('optCamZoom'), camZoomVal: $('camZoomVal'), diag: $('diag'),
  };

  // ---------- Settings ----------
  const DEFAULTS = {
    voice: true, beep: true, repeat: false, zoom: true, sens: 6, debug: false,
    muted: false, volume: 80,          // alert volume, 0–100
    lens: 'wide', camZoom: 1,          // rear lens ('ultra' | 'wide' | 'tele') and its zoom factor
  };
  const settings = loadSettings();

  function loadSettings() {
    try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem('signal-spotter') || '{}') }; }
    catch { return { ...DEFAULTS }; }
  }
  function saveSettings() {
    try { localStorage.setItem('signal-spotter', JSON.stringify(settings)); } catch { /* private mode */ }
  }
  const minScore = () => 0.62 - settings.sens * 0.04;   // sens 1 → 0.58 … sens 10 → 0.22

  function bindSettings() {
    const bools = { optVoice: 'voice', optBeep: 'beep', optRepeat: 'repeat', optZoom: 'zoom', optDebug: 'debug' };
    for (const [id, key] of Object.entries(bools)) {
      els[id].checked = settings[key];
      els[id].addEventListener('change', () => { settings[key] = els[id].checked; saveSettings(); });
    }
    els.optSens.value = settings.sens;
    const showSens = () => { els.sensVal.textContent = `${settings.sens}/10`; };
    showSens();
    els.optSens.addEventListener('input', () => { settings.sens = +els.optSens.value; showSens(); saveSettings(); });

    els.optVolume.value = settings.volume;
    els.optVolume.addEventListener('input', () => {
      settings.volume = +els.optVolume.value;
      if (settings.volume > 0) settings.muted = false;
      saveSettings(); updateVolumeUI();
    });
    els.optVolume.addEventListener('change', () => playAlertSound('green'));  // preview on release
    updateVolumeUI();
  }
  function updateVolumeUI() {
    const silent = settings.muted || settings.volume === 0;
    els.btnVolume.textContent = silent ? '🔇' : settings.volume < 40 ? '🔈' : settings.volume < 75 ? '🔉' : '🔊';
    els.volVal.textContent = `${settings.volume}%`;
    els.btnMuteToggle.textContent = settings.muted ? 'Unmute' : 'Mute';
    els.btnMuteToggle.classList.toggle('on', settings.muted);
    els.btnVolume.setAttribute('aria-label', silent ? 'Alert volume (muted)' : `Alert volume ${settings.volume}%`);
  }

  // ---------- Model ----------
  let model = null;
  const modelReady = loadModel();

  async function loadModel() {
    try {
      try { await tf.setBackend('webgl'); } catch { await tf.setBackend('cpu'); }
      await tf.ready();
      model = await tf.loadGraphModel(MODEL_URL, {
        onProgress: (p) => {
          els.loadBar.style.width = `${Math.round(p * 90)}%`;
          els.loadText.textContent = `Loading detection model… ${Math.round(p * 100)}%`;
        },
      });
      els.loadText.textContent = 'Warming up…';
      const warm = tf.zeros([1, MODEL_INPUT, MODEL_INPUT, 3], 'int32');
      const out = await model.executeAsync(warm);
      await Promise.all(out.map((t) => t.data()));
      tf.dispose([warm, ...out]);
      els.loadBar.style.width = '100%';
      els.loadText.textContent = `Ready · ${tf.getBackend().toUpperCase()} · works offline`;
      return model;
    } catch (err) {
      console.error(err);
      els.loadText.textContent = `Could not load the model: ${err.message}`;
      throw err;
    }
  }

  // Canvas used to feed the model (300×300, the network's native input size).
  const inCanvas = document.createElement('canvas');
  inCanvas.width = inCanvas.height = MODEL_INPUT;
  const inCtx = inCanvas.getContext('2d', { willReadFrequently: true });

  /** Runs the detector on region (sx,sy,sw,sh) of the source; returns boxes in source pixels. */
  async function detectRegion(src, sx, sy, sw, sh, thresh) {
    inCtx.drawImage(src, sx, sy, sw, sh, 0, 0, MODEL_INPUT, MODEL_INPUT);
    const input = tf.tidy(() => tf.expandDims(tf.browser.fromPixels(inCanvas), 0));
    const outputs = await model.executeAsync(input);
    input.dispose();

    // Identify outputs by shape: scores [1,N,90], boxes [1,N,1,4]
    let scoresT = null, boxesT = null;
    for (const t of outputs) {
      const last = t.shape[t.shape.length - 1];
      if (last === 4) boxesT = t; else if (last >= 90) scoresT = t;
    }
    const colT = tf.tidy(() => scoresT.slice([0, 0, TRAFFIC_LIGHT_COL], [1, -1, 1]));
    const [scores, boxes] = await Promise.all([colT.data(), boxesT.data()]);
    tf.dispose([colT, ...outputs]);

    const dets = [];
    for (let i = 0; i < scores.length; i++) {
      const score = scores[i];
      if (score < thresh) continue;
      const y1 = boxes[i * 4], x1 = boxes[i * 4 + 1], y2 = boxes[i * 4 + 2], x2 = boxes[i * 4 + 3];
      const x = sx + Math.max(0, x1) * sw, y = sy + Math.max(0, y1) * sh;
      const w = (Math.min(1, x2) - Math.max(0, x1)) * sw, h = (Math.min(1, y2) - Math.max(0, y1)) * sh;
      if (w < 4 || h < 4) continue;
      dets.push({ x, y, w, h, score });
    }
    return dets;
  }

  function iou(a, b) {
    const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
    const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    const inter = ix * iy;
    return inter / (a.w * a.h + b.w * b.h - inter);
  }
  function nms(dets, thr = 0.45) {
    dets.sort((a, b) => b.score - a.score);
    const keep = [];
    for (const d of dets) if (keep.every((k) => iou(k, d) < thr)) keep.push(d);
    return keep.slice(0, 10);
  }

  /** Square tile over the upper-center of the frame, where traffic lights usually are. */
  function zoomTile(vw, vh) {
    const side = Math.min(vw, vh) * 0.5;
    const cx = vw / 2, cy = vh * (vh > vw ? 0.36 : 0.4);
    return { sx: cx - side / 2, sy: Math.max(0, cy - side / 2), sw: side, sh: side };
  }

  // ---------- Color classification ----------
  const clsCanvas = document.createElement('canvas');
  const clsCtx = clsCanvas.getContext('2d', { willReadFrequently: true });

  function hueClass(h) {
    if (h < 22 || h >= 320) return 'red';
    if (h < 70) return 'yellow';
    if (h >= 70 && h <= 205) return 'green';
    return null;
  }

  /** Classifies the lit lamp inside box d of the source. Returns {color, conf}. */
  function classifyColor(src, vw, vh, d) {
    const x = Math.max(0, Math.floor(d.x)), y = Math.max(0, Math.floor(d.y));
    const w = Math.min(vw - x, Math.ceil(d.w)), h = Math.min(vh - y, Math.ceil(d.h));
    if (w < 2 || h < 2) return { color: 'unknown', conf: 0 };
    const scale = 64 / Math.max(w, h);
    const cw = Math.max(4, Math.round(w * scale)), ch = Math.max(4, Math.round(h * scale));
    clsCanvas.width = cw; clsCanvas.height = ch;
    clsCtx.drawImage(src, x, y, w, h, 0, 0, cw, ch);
    const px = clsCtx.getImageData(0, 0, cw, ch).data;

    let vmax = 0;
    for (let i = 0; i < px.length; i += 4) {
      const v = Math.max(px[i], px[i + 1], px[i + 2]);
      if (v > vmax) vmax = v;
    }
    if (vmax < 110) return { color: 'unknown', conf: 0 };   // nothing lit

    const bright = Math.max(110, vmax * 0.62);
    const halo = bright * 0.7;
    const score = { red: 0, yellow: 0, green: 0 };
    let cx = 0, cy = 0, cwSum = 0;

    for (let py = 0; py < ch; py++) {
      for (let pxi = 0; pxi < cw; pxi++) {
        const i = (py * cw + pxi) * 4;
        const r = px[i], g = px[i + 1], b = px[i + 2];
        const max = Math.max(r, g, b);
        if (max < halo) continue;
        const min = Math.min(r, g, b);
        const s = (max - min) / max;
        // Lamps sit inside the housing; pixels at the box edge are mostly background (sky, foliage).
        const edge = pxi < cw * 0.1 || pxi >= cw * 0.9 || py < ch * 0.06 || py >= ch * 0.94;
        const wPos = (max >= bright ? 1 : 0.4) * (edge ? 0.25 : 1);
        cx += pxi * wPos; cy += py * wPos; cwSum += wPos;  // lamp centroid (incl. washed-out cores)
        if (s < 0.22) continue;
        let hue;
        const dlt = max - min;
        if (max === r) hue = 60 * (((g - b) / dlt) % 6);
        else if (max === g) hue = 60 * ((b - r) / dlt + 2);
        else hue = 60 * ((r - g) / dlt + 4);
        if (hue < 0) hue += 360;
        const cls = hueClass(hue);
        if (cls) score[cls] += s * (max / 255) * wPos;
      }
    }

    const total = score.red + score.yellow + score.green;
    const pixels = cw * ch;

    // Position prior: vertical housing → top=red, middle=yellow, bottom=green;
    // horizontal housing → left=red, right=green.
    let prior = null;
    if (cwSum > 0) {
      const ny = cy / cwSum / ch, nx = cx / cwSum / cw;
      const pos = h > w * 1.35 ? ny : w > h * 1.35 ? nx : null;
      if (pos !== null) prior = pos < 0.4 ? 'red' : pos > 0.6 ? 'green' : 'yellow';
    }

    // No colored glow → lamp off, or the back/side of a signal head seen against the sky.
    if (total < pixels * 0.004) return { color: 'unknown', conf: 0 };
    if (prior) score[prior] += total * 0.25;

    let best = 'red';
    for (const k of ['yellow', 'green']) if (score[k] > score[best]) best = k;
    const sum = score.red + score.yellow + score.green;
    return { color: best, conf: score[best] / sum };
  }

  // ---------- Announcements ----------
  // iOS mutes Web Audio when the ring/silent switch is on, and ignores the volume of both
  // speechSynthesis and <audio>. So each alert (tones + recorded voice) is rendered here at the
  // chosen volume into a small WAV and played through one <audio> element, which iOS treats as
  // media playback (audible with the silent switch on). The element is unlocked by the Start tap.
  const SR = 24000;
  const alertAudio = new Audio();
  alertAudio.preload = 'auto';
  let alertUrl = null;
  let audioStatus = 'not started';
  const voicePcm = {};                      // color → Float32Array at SR
  ['red', 'yellow', 'green'].forEach((c) => {
    fetch(`audio/${c}.wav`).then((r) => (r.ok ? r.arrayBuffer() : null))
      .then((b) => { if (b) voicePcm[c] = parseWav(b); }).catch(() => {});
  });

  /** Slider position (0–100) → linear gain, on a curve that sounds even across the range. */
  const volumeGain = () => (settings.muted ? 0 : (settings.volume / 100) ** 2);

  function parseWav(buf) {
    const v = new DataView(buf);
    let off = 12, rate = SR, bits = 16, chans = 1;
    while (off + 8 <= v.byteLength) {
      const id = String.fromCharCode(v.getUint8(off), v.getUint8(off + 1), v.getUint8(off + 2), v.getUint8(off + 3));
      const size = v.getUint32(off + 4, true);
      if (id === 'fmt ') { chans = v.getUint16(off + 10, true); rate = v.getUint32(off + 12, true); bits = v.getUint16(off + 22, true); }
      if (id === 'data') {
        if (bits !== 16) return null;
        const n = Math.floor(size / 2 / chans), src = new Float32Array(n);
        for (let i = 0; i < n; i++) src[i] = v.getInt16(off + 8 + i * 2 * chans, true) / 32768;
        if (rate === SR) return src;
        const out = new Float32Array(Math.round(n * SR / rate));   // linear resample
        for (let i = 0; i < out.length; i++) {
          const p = i * rate / SR, j = Math.floor(p), f = p - j;
          out[i] = (src[j] || 0) * (1 - f) + (src[j + 1] || 0) * f;
        }
        return out;
      }
      off += 8 + size + (size & 1);
    }
    return null;
  }

  function encodeWav(pcm) {
    const buf = new ArrayBuffer(44 + pcm.length * 2), v = new DataView(buf);
    const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    str(0, 'RIFF'); v.setUint32(4, 36 + pcm.length * 2, true); str(8, 'WAVE');
    str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, SR, true); v.setUint32(28, SR * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, 'data'); v.setUint32(40, pcm.length * 2, true);
    for (let i = 0; i < pcm.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, pcm[i])) * 32767, true);
    return new Blob([buf], { type: 'audio/wav' });
  }

  const BEEPS = {
    red: [[392, 0, 0.18], [392, 0.26, 0.18]],                       // two low tones
    yellow: [[587, 0, 0.12], [587, 0.18, 0.12], [587, 0.36, 0.12]], // three medium tones
    green: [[659, 0, 0.12], [880, 0.14, 0.22]],                     // rising chime
  };

  /** Renders tones and/or voice for a color at the given gain; returns PCM or null. */
  function renderAlert(color, gain) {
    const lead = 0.08;                       // iOS can clip the very start of a clip
    const beepLen = settings.beep ? Math.max(...BEEPS[color].map(([, at, dur]) => at + dur)) + 0.12 : 0;
    const voice = settings.voice ? voicePcm[color] : null;
    if (!settings.beep && !voice) return null;
    const total = Math.ceil((lead + beepLen + 0.05) * SR) + (voice ? voice.length : 0);
    const pcm = new Float32Array(total);
    if (settings.beep) {
      for (const [freq, at, dur] of BEEPS[color]) {
        const start = Math.round((lead + at) * SR), n = Math.round(dur * SR);
        const peak = color === 'red' ? 0.35 : 0.6;
        for (let i = 0; i < n; i++) {
          const t = i / SR;
          const env = t < 0.01 ? t / 0.01 : Math.exp(-6.9 * (t - 0.01) / (dur - 0.01));  // 10 ms attack, decay to −60 dB
          const ph = 2 * Math.PI * freq * t;
          const wave = color === 'red' ? (Math.sin(ph) >= 0 ? 1 : -1) : Math.sin(ph);
          pcm[start + i] += wave * env * peak;
        }
      }
    }
    if (voice) pcm.set(voice, Math.round((lead + beepLen) * SR));
    for (let i = 0; i < pcm.length; i++) pcm[i] *= gain;
    return pcm;
  }

  function playPcm(pcm) {
    if (alertUrl) URL.revokeObjectURL(alertUrl);
    alertUrl = URL.createObjectURL(encodeWav(pcm));
    alertAudio.src = alertUrl;
    return alertAudio.play();
  }

  /** Must run inside a tap: plays a short silent clip so later alerts may play without a tap. */
  function unlockAudio() {
    playPcm(new Float32Array(Math.round(SR * 0.05)))
      .then(() => { audioStatus = 'ready (media playback)'; })
      .catch((err) => { if (err.name !== 'AbortError') audioStatus = `blocked: ${err.name}`; });
    try {
      if ('speechSynthesis' in window) {     // fallback path, unlocked too
        const u = new SpeechSynthesisUtterance(' ');
        u.volume = 0;
        speechSynthesis.speak(u);
      }
    } catch { /* no speech */ }
  }

  function speakFallback(color) {
    if (!settings.voice || !('speechSynthesis' in window)) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(SPOKEN[color]);
    u.rate = 1.05;
    speechSynthesis.speak(u);
  }

  function playAlertSound(color) {
    const gain = volumeGain();
    if (gain === 0) return;
    const pcm = renderAlert(color, gain);
    if (!pcm) { speakFallback(color); return; }   // voice clip not loaded yet
    playPcm(pcm)
      .then(() => { audioStatus = 'ready (media playback)'; })
      .catch((err) => {
        if (err.name === 'AbortError') return;          // replaced by a newer alert
        audioStatus = `blocked: ${err.name} (using speech fallback)`;
        speakFallback(color);
      });
  }

  function announce(color) {
    els.banner.className = color;
    els.edge.className = color;
    els.bannerText.textContent = color.toUpperCase();
    void els.banner.offsetWidth;            // restart pulse animation
    els.banner.classList.add('pulse');
    playAlertSound(color);
  }

  function clearAnnouncement() {
    els.banner.className = 'none';
    els.edge.className = '';
    els.bannerText.textContent = 'Searching for traffic lights…';
  }

  // ---------- Temporal smoothing ----------
  const history = [];          // recent per-frame colors (null = no light)
  let announced = null, announcedAt = 0, lastSeen = 0;

  function frameVote(dets) {
    const w = { red: 0, yellow: 0, green: 0 };
    for (const d of dets) {
      if (d.color === 'unknown' || d.conf < 0.45) continue;
      w[d.color] += d.score * d.conf * Math.sqrt(d.w * d.h);
    }
    let best = null, bestW = 0;
    for (const k in w) if (w[k] > bestW) { best = k; bestW = w[k]; }
    return best;
  }

  function updateState(color, now) {
    history.push(color);
    if (history.length > 5) history.shift();
    if (color) lastSeen = now;

    const count = (c) => history.filter((h) => h === c).length;
    if (color && count(color) >= 3) {
      if (color !== announced) {
        announced = color; announcedAt = now;
        announce(color);
      } else if (settings.repeat && now - announcedAt > 5000) {
        announcedAt = now;
        announce(color);
      }
    } else if (announced && now - lastSeen > LOST_AFTER_MS) {
      announced = null;
      clearAnnouncement();
    }
  }

  // ---------- Drawing ----------
  const octx = els.overlay.getContext('2d');
  let shownDets = [], shownAt = 0;

  function sizeOverlay() {
    const dpr = window.devicePixelRatio || 1;
    const W = window.innerWidth, H = window.innerHeight;
    els.overlay.width = Math.round(W * dpr); els.overlay.height = Math.round(H * dpr);
    octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /** Mapping from source pixels to screen CSS pixels (object-fit cover/contain, then centred zoom z). */
  function screenMap(vw, vh, fit, z = 1) {
    const W = window.innerWidth, H = window.innerHeight;
    const s = (fit === 'contain' ? Math.min(W / vw, H / vh) : Math.max(W / vw, H / vh)) * z;
    return { s, dx: (W - vw * s) / 2, dy: (H - vh * s) / 2 };
  }

  function roundRect(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function draw(vw, vh, fit, tile, z) {
    const W = window.innerWidth, H = window.innerHeight;
    octx.clearRect(0, 0, W, H);
    const m = screenMap(vw, vh, fit, z);

    if (settings.debug && tile) {
      octx.setLineDash([6, 6]); octx.lineWidth = 1.5; octx.strokeStyle = 'rgba(255,255,255,0.5)';
      octx.strokeRect(m.dx + tile.sx * m.s, m.dy + tile.sy * m.s, tile.sw * m.s, tile.sh * m.s);
      octx.setLineDash([]);
    }
    if (performance.now() - shownAt > BOX_HOLD_MS) return;

    for (const d of shownDets) {
      const pad = 6;
      const x = m.dx + d.x * m.s - pad, y = m.dy + d.y * m.s - pad;
      const w = d.w * m.s + pad * 2, h = d.h * m.s + pad * 2;
      const col = COLORS[d.color];
      octx.save();
      octx.shadowColor = col; octx.shadowBlur = 18;
      octx.lineWidth = 4; octx.strokeStyle = col;
      roundRect(octx, x, y, w, h, 10); octx.stroke();
      octx.restore();
      octx.fillStyle = col + '22';
      roundRect(octx, x, y, w, h, 10); octx.fill();

      const label = d.color === 'unknown' ? 'LIGHT' : d.color.toUpperCase();
      const text = settings.debug ? `${label} ${Math.round(d.score * 100)}% · ${Math.round(d.conf * 100)}%` : label;
      octx.font = '700 15px -apple-system, system-ui, sans-serif';
      const tw = octx.measureText(text).width + 16;
      const ly = y - 28 < 4 ? y + h + 4 : y - 28;
      octx.fillStyle = col;
      roundRect(octx, x, ly, tw, 24, 8); octx.fill();
      octx.fillStyle = d.color === 'yellow' || d.color === 'green' ? '#111' : '#fff';
      octx.fillText(text, x + 8, ly + 17);
    }
  }

  // ---------- Source & main loop ----------
  let running = false, stream = null, sourceKind = 'camera', wakeLock = null;
  let fpsAvg = 0, inferAvg = 0;

  function currentSource() {
    if (sourceKind === 'image') return { el: els.photo, vw: els.photo.naturalWidth, vh: els.photo.naturalHeight, fit: 'contain', zoom: 1 };
    const v = els.video;
    const camera = sourceKind === 'camera';
    return { el: v, vw: v.videoWidth, vh: v.videoHeight, fit: camera ? 'cover' : 'contain', zoom: camera ? digitalZoom : 1 };
  }

  async function loop() {
    let last = performance.now();
    while (running) {
      const src = currentSource();
      if (!model || !src.vw || !src.vh || (src.el === els.video && els.video.readyState < 2)) {
        await nextFrame();
        continue;
      }
      const t0 = performance.now();
      const thresh = minScore();
      // With digital zoom, detect on the zoomed (centre) part of the frame only.
      const z = src.zoom || 1;
      const cw = src.vw / z, ch = src.vh / z, cx = (src.vw - cw) / 2, cy = (src.vh - ch) / 2;
      let dets = await detectRegion(src.el, cx, cy, cw, ch, thresh);
      let tile = null;
      if (settings.zoom) {
        const t = zoomTile(cw, ch);
        tile = { sx: cx + t.sx, sy: cy + t.sy, sw: t.sw, sh: t.sh };
        dets = dets.concat(await detectRegion(src.el, tile.sx, tile.sy, tile.sw, tile.sh, thresh));
      }
      dets = nms(dets);
      for (const d of dets) Object.assign(d, classifyColor(src.el, src.vw, src.vh, d));
      const t1 = performance.now();

      if (dets.length) { shownDets = dets; shownAt = t1; }
      updateState(frameVote(dets), t1);
      draw(src.vw, src.vh, src.fit, tile, z);

      const dt = t1 - last; last = t1;
      fpsAvg = fpsAvg ? fpsAvg * 0.9 + (1000 / dt) * 0.1 : 1000 / dt;
      inferAvg = inferAvg ? inferAvg * 0.9 + (t1 - t0) * 0.1 : t1 - t0;
      els.status.textContent = settings.debug
        ? `${fpsAvg.toFixed(1)} fps · ${inferAvg.toFixed(0)} ms · ${tf.getBackend()}\n${src.vw}×${src.vh} · thr ${thresh.toFixed(2)} · zoom ${z.toFixed(1)}× · ${dets.length} box`
        : (dets.length ? `${dets.length} light${dets.length > 1 ? 's' : ''} in view` : 'Scanning…');

      await nextFrame();
    }
  }
  const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));

  // ---------- Camera, lenses & zoom ----------
  // Some iOS versions list each rear lens as its own camera ("Back Telephoto Camera", …), others
  // expose a single "Back Camera". Tele therefore uses the telephoto lens when it is listed and
  // otherwise zooms the main camera — with hardware zoom if the browser offers it, else a digital
  // crop. The crop is also what the detector sees, so distant lights become larger for it too.
  const TELE_ZOOM = 3, DIGITAL_MAX = 5, HW_MAX = 10;
  const LENS_NAME = { ultra: 'Ultra wide', wide: 'Wide', tele: 'Telephoto' };
  let lenses = {};            // kind → deviceId, rear lenses only
  let cameraList = [];        // all video inputs, for diagnostics
  let activeLens = 'wide';
  let hwZoom = null;          // {min, max} when the track supports the zoom constraint
  let digitalZoom = 1;

  function lensKind(label) {
    const l = (label || '').toLowerCase();
    if (/front|facetime|avant|vorder|frontal|anterior/.test(l)) return 'front';
    if (/ultra/.test(l)) return 'ultra';
    if (/tele/.test(l)) return 'tele';
    if (/dual|triple|double|doble|dreifach|tripla/.test(l)) return 'virtual';   // multi-lens virtual cameras
    return 'wide';
  }

  async function findLenses() {
    const found = {};
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
    cameraList = devices.map((d) => d.label || '(label hidden)');
    for (const d of devices) {
      if (!d.label) continue;
      const k = lensKind(d.label);
      if (k !== 'front' && k !== 'virtual' && !found[k]) found[k] = d.deviceId;
    }
    return found;
  }

  function openStream(deviceId) {
    const video = { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } };
    if (deviceId) video.deviceId = { exact: deviceId };
    else video.facingMode = { ideal: 'environment' };
    return navigator.mediaDevices.getUserMedia({ audio: false, video });
  }

  /** Physical camera a mode needs: Tele falls back to zooming the main camera. */
  const deviceFor = (mode) => (mode === 'tele' && !lenses.tele ? 'wide' : mode);

  async function startCamera(mode = settings.lens) {
    stopSource();
    sourceKind = 'camera';
    els.photo.hidden = true; els.video.hidden = false;
    els.stage.classList.remove('contain');

    const want = deviceFor(mode);
    if (!Object.keys(lenses).length) lenses = await findLenses().catch(() => ({}));
    const id = lenses[deviceFor(mode)] || null;
    try {
      stream = await openStream(id);
    } catch (err) {
      if (!id) throw err;
      stream = await openStream(null);          // lens unavailable → default rear camera
    }
    if (!Object.keys(lenses).length) {
      // First run: permission was just granted, so the lens list is readable now.
      lenses = await findLenses().catch(() => ({}));
      const target = deviceFor(mode);
      if (lenses[target] && lensKind(stream.getVideoTracks()[0].label) !== target) {
        stream.getTracks().forEach((t) => t.stop());
        stream = await openStream(lenses[target]);
      }
    }
    const opened = lensKind(stream.getVideoTracks()[0].label);
    activeLens = opened === 'tele' || opened === 'ultra' ? opened : 'wide';
    if (activeLens !== want && want !== 'wide') settings.lens = activeLens;   // requested lens not available
    els.video.srcObject = stream;
    await els.video.play();
    setupZoom();
    renderLensBar();
  }

  async function selectMode(mode) {
    if (sourceKind !== 'camera') return;
    settings.lens = mode;
    settings.camZoom = mode === 'tele' && !lenses.tele ? TELE_ZOOM : 1;
    saveSettings();
    history.length = 0;
    const target = deviceFor(mode);
    if (lenses[target] && target !== activeLens) {
      try { await startCamera(mode); } catch (err) { els.status.textContent = cameraErrorText(err); }
    } else {
      setupZoom();
      renderLensBar();
    }
  }

  function currentMode() {
    if (activeLens !== 'wide') return activeLens;
    return !lenses.tele && settings.camZoom >= 2 ? 'tele' : 'wide';
  }

  function renderLensBar() {
    const modes = ['ultra', 'wide', 'tele'].filter((k) => k !== 'ultra' || lenses.ultra);
    const label = { ultra: '0.5×', wide: '1×', tele: 'Tele' };
    const cur = currentMode();
    els.lensButtons.textContent = '';
    for (const k of modes) {
      const b = document.createElement('button');
      b.textContent = label[k];
      b.setAttribute('role', 'radio');
      b.setAttribute('aria-label', k === 'tele' && !lenses.tele ? `Telephoto (${TELE_ZOOM}× zoom)` : `${LENS_NAME[k]} lens`);
      b.setAttribute('aria-checked', String(k === cur));
      b.addEventListener('click', () => selectMode(k));
      els.lensButtons.appendChild(b);
    }
    els.lensButtons.hidden = false;
    els.zoomRow.hidden = false;
    els.lensBar.hidden = false;
  }

  /** Zoom slider: hardware zoom when the camera track supports it, otherwise digital. */
  function setupZoom() {
    const track = stream && stream.getVideoTracks()[0];
    const z = track && track.getCapabilities ? track.getCapabilities().zoom : null;
    hwZoom = z && z.max > z.min ? { min: z.min, max: Math.min(z.max, HW_MAX) } : null;
    const min = hwZoom ? hwZoom.min : 1, max = hwZoom ? hwZoom.max : DIGITAL_MAX;
    els.optCamZoom.min = min;
    els.optCamZoom.max = max;
    els.optCamZoom.step = 0.1;
    const v = Math.min(max, Math.max(min, settings.camZoom));
    els.optCamZoom.value = v;
    applyZoom(v);
  }

  function applyZoom(v) {
    els.camZoomVal.textContent = `${v.toFixed(1)}×`;
    if (hwZoom) {
      digitalZoom = 1;
      const track = stream && stream.getVideoTracks()[0];
      if (track) track.applyConstraints({ advanced: [{ zoom: v }] }).catch(() => {});
    } else {
      digitalZoom = v;
    }
    els.video.style.transform = digitalZoom > 1 ? `scale(${digitalZoom})` : '';
  }

  function diagnostics() {
    const track = stream && stream.getVideoTracks()[0];
    const s = track && track.getSettings ? track.getSettings() : {};
    return [
      `Version 1.2 · ${tf.getBackend ? tf.getBackend() : '?'}`,
      `Cameras: ${cameraList.length ? cameraList.join(', ') : 'none listed yet'}`,
      `Rear lenses found: ${Object.keys(lenses).join(', ') || 'none'}`,
      `Active: ${track ? track.label || '(no label)' : 'no camera'}${s.width ? ` · ${s.width}×${s.height}` : ''}`,
      `Zoom: ${hwZoom ? `hardware ${hwZoom.min}–${hwZoom.max}×` : `digital (max ${DIGITAL_MAX}×)`} · now ${settings.camZoom}×`,
      `Audio: ${audioStatus} · voice clips ${Object.keys(voicePcm).length}/3`,
      `Volume ${settings.volume}%${settings.muted ? ' (muted)' : ''}`,
      navigator.userAgent.replace(/^Mozilla\/5\.0 /, ''),
    ].join('\n');
  }

  function stopSource() {
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
    digitalZoom = 1;
    els.video.style.transform = '';
    els.video.pause();
    els.video.srcObject = null;
    if (els.video.src) { URL.revokeObjectURL(els.video.src); els.video.removeAttribute('src'); }
  }

  async function startFile(file) {
    stopSource();
    els.lensBar.hidden = true;
    const url = URL.createObjectURL(file);
    if (file.type.startsWith('image/')) {
      sourceKind = 'image';
      els.video.hidden = true; els.photo.hidden = false;
      els.photo.src = url;
      await els.photo.decode();
    } else {
      sourceKind = 'file';
      els.photo.hidden = true; els.video.hidden = false;
      els.stage.classList.add('contain');
      els.video.src = url; els.video.loop = true;
      await els.video.play();
    }
  }

  async function requestWakeLock() {
    try {
      if ('wakeLock' in navigator && !wakeLock) {
        wakeLock = await navigator.wakeLock.request('screen');
        wakeLock.addEventListener('release', () => { wakeLock = null; });
      }
    } catch { /* not granted */ }
  }

  async function begin(startFn) {
    unlockAudio();
    els.btnStart.disabled = true;
    try {
      await startFn();
    } catch (err) {
      console.error(err);
      els.btnStart.disabled = false;
      els.loadText.textContent = cameraErrorText(err);
      return;
    }
    els.start.hidden = true;
    els.btnStart.disabled = false;
    history.length = 0; announced = null; clearAnnouncement();
    requestWakeLock();
    if (!model) {
      els.status.textContent = 'Loading model…';
      try { await modelReady; } catch { els.status.textContent = 'Model failed to load'; return; }
    }
    if (!running) { running = true; loop(); }
  }

  function cameraErrorText(err) {
    if (!window.isSecureContext) return 'Camera needs a secure (https://) page. See README for hosting.';
    if (err && err.name === 'NotAllowedError') return 'Camera access was denied. Allow it in Settings › Safari › Camera, then try again.';
    if (err && err.name === 'NotFoundError') return 'No camera found on this device.';
    return `Could not start: ${err && err.message ? err.message : err}`;
  }

  // ---------- UI wiring ----------
  bindSettings();
  sizeOverlay();
  window.addEventListener('resize', sizeOverlay);
  window.addEventListener('orientationchange', () => setTimeout(sizeOverlay, 300));

  els.btnStart.addEventListener('click', () => begin(startCamera));
  els.fileInput.addEventListener('change', () => {
    const f = els.fileInput.files && els.fileInput.files[0];
    if (f) begin(() => startFile(f));
    els.fileInput.value = '';
  });
  els.btnSettings.addEventListener('click', () => {
    els.diag.textContent = diagnostics();
    els.settings.hidden = false;
  });
  els.btnCloseSettings.addEventListener('click', () => { els.settings.hidden = true; });
  els.settings.addEventListener('click', (e) => { if (e.target === els.settings) els.settings.hidden = true; });
  els.btnChangeSource.addEventListener('click', () => {
    els.settings.hidden = true;
    running = false;
    stopSource();
    clearAnnouncement();
    octx.clearRect(0, 0, els.overlay.width, els.overlay.height);
    els.status.textContent = '';
    els.lensBar.hidden = true;
    showVolumePanel(false);
    els.start.hidden = false;
  });

  function showVolumePanel(show) {
    els.volPanel.hidden = !show;
    els.btnVolume.setAttribute('aria-expanded', String(show));
  }
  els.btnVolume.addEventListener('click', () => { unlockAudio(); showVolumePanel(els.volPanel.hidden); });
  // Any tap re-enables alert audio if iOS has not allowed it yet.
  document.addEventListener('touchend', () => { if (!audioStatus.startsWith('ready')) unlockAudio(); });
  document.addEventListener('pointerdown', (e) => {
    if (!els.volPanel.hidden && !els.volPanel.contains(e.target) && !els.btnVolume.contains(e.target)) showVolumePanel(false);
  });
  els.btnMuteToggle.addEventListener('click', () => {
    settings.muted = !settings.muted;
    if (!settings.muted && settings.volume === 0) { settings.volume = 50; els.optVolume.value = 50; }
    saveSettings(); updateVolumeUI();
    if (settings.muted && 'speechSynthesis' in window) speechSynthesis.cancel();
  });
  let testColor = 0;
  els.btnTestSound.addEventListener('click', () => {
    playAlertSound(['red', 'yellow', 'green'][testColor++ % 3]);
  });
  els.optCamZoom.addEventListener('input', () => {
    settings.camZoom = +els.optCamZoom.value; saveSettings();
    applyZoom(settings.camZoom);
    const cur = currentMode();
    els.lensButtons.querySelectorAll('button').forEach((b, i, all) => {
      const k = all.length === 3 ? ['ultra', 'wide', 'tele'][i] : ['wide', 'tele'][i];
      b.setAttribute('aria-checked', String(k === cur));
    });
  });

  document.addEventListener('visibilitychange', async () => {
    if (document.visibilityState !== 'visible' || !running) return;
    requestWakeLock();
    if (sourceKind === 'camera') {
      const track = stream && stream.getVideoTracks()[0];
      if (!track || track.readyState === 'ended') {
        try { await startCamera(); } catch (err) { els.status.textContent = cameraErrorText(err); }
      } else {
        els.video.play().catch(() => {});
      }
    }
  });

  // Debug hook (used for testing on still images from the console).
  window.SignalSpotter = {
    async analyze(el, { zoom = settings.zoom } = {}) {
      await modelReady;
      const vw = el.naturalWidth || el.videoWidth || el.width, vh = el.naturalHeight || el.videoHeight || el.height;
      let dets = await detectRegion(el, 0, 0, vw, vh, minScore());
      if (zoom) { const t = zoomTile(vw, vh); dets = dets.concat(await detectRegion(el, t.sx, t.sy, t.sw, t.sh, minScore())); }
      dets = nms(dets);
      for (const d of dets) Object.assign(d, classifyColor(el, vw, vh, d));
      return { dets, vote: frameVote(dets) };
    },
    audioState: () => ({ status: audioStatus, voices: Object.keys(voicePcm), src: alertAudio.src.slice(0, 30), paused: alertAudio.paused }),
    diagnostics: () => diagnostics(),
  };

  // ---------- Offline support ----------
  // Skipped on localhost during development (stale caches) unless ?sw=1 is given.
  const isDevHost = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname) && !/[?&]sw=1/.test(location.search);
  if ('serviceWorker' in navigator && window.isSecureContext && !isDevHost) {
    navigator.serviceWorker.register('sw.js').catch((err) => console.warn('SW registration failed', err));
  }
})();
