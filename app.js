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
    optCamZoom: $('optCamZoom'), camZoomVal: $('camZoomVal'),
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
      saveSettings(); applyVolume(); updateVolumeUI();
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
  // All alert audio runs through one Web Audio gain node so the volume slider controls both the
  // beeps and the voice. The voice is pre-recorded (audio/*.wav) because iOS ignores the volume
  // of speechSynthesis; speechSynthesis is only a fallback if the clips cannot be loaded.
  let audioCtx = null, master = null;
  const voiceData = {}, voiceBuffers = {};
  const voiceLoaded = Promise.all(['red', 'yellow', 'green'].map((c) =>
    fetch(`audio/${c}.wav`).then((r) => (r.ok ? r.arrayBuffer() : null)).then((b) => { voiceData[c] = b; }).catch(() => {})));

  /** Slider position (0–100) → linear gain, on a curve that sounds even across the range. */
  const volumeGain = () => (settings.muted ? 0 : (settings.volume / 100) ** 2);

  function applyVolume() {
    if (!master) return;
    master.gain.cancelScheduledValues(0);
    master.gain.value = volumeGain();
  }

  function unlockAudio() {
    // iOS: route web audio as media playback so it is audible even with the silent switch on.
    try { if (navigator.audioSession) navigator.audioSession.type = 'playback'; } catch { /* unsupported */ }
    try {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        master = audioCtx.createGain();
        master.gain.value = volumeGain();
        master.connect(audioCtx.destination);
      }
      if (audioCtx.state === 'suspended') audioCtx.resume();
      const o = audioCtx.createOscillator(), g = audioCtx.createGain();
      g.gain.value = 0.0001; o.connect(g).connect(audioCtx.destination);
      o.start(); o.stop(audioCtx.currentTime + 0.02);
      voiceLoaded.then(decodeVoices);
    } catch { /* no web audio */ }
    try {
      if ('speechSynthesis' in window) {
        const u = new SpeechSynthesisUtterance(' ');
        u.volume = 0;
        speechSynthesis.speak(u);
      }
    } catch { /* no speech */ }
  }

  function decodeVoices() {
    for (const [c, data] of Object.entries(voiceData)) {
      if (!data || voiceBuffers[c]) continue;
      // slice(): decodeAudioData detaches the buffer it is given
      audioCtx.decodeAudioData(data.slice(0), (buf) => { voiceBuffers[c] = buf; }, () => {});
    }
  }

  /** Schedules the beep pattern at audio time t0; returns its duration in seconds. */
  function beep(color, t0) {
    const patterns = {
      red: [[392, 0, 0.18], [392, 0.26, 0.18]],                       // two low tones
      yellow: [[587, 0, 0.12], [587, 0.18, 0.12], [587, 0.36, 0.12]], // three medium tones
      green: [[659, 0, 0.12], [880, 0.14, 0.22]],                     // rising chime
    };
    let end = 0;
    for (const [freq, at, dur] of patterns[color]) {
      const o = audioCtx.createOscillator(), g = audioCtx.createGain();
      o.type = color === 'red' ? 'square' : 'sine';
      o.frequency.value = freq;
      g.gain.setValueAtTime(0, t0 + at);
      g.gain.linearRampToValueAtTime(color === 'red' ? 0.3 : 0.6, t0 + at + 0.01);
      g.gain.exponentialRampToValueAtTime(0.001, t0 + at + dur);
      o.connect(g).connect(master);
      o.start(t0 + at); o.stop(t0 + at + dur + 0.02);
      end = Math.max(end, at + dur);
    }
    return end;
  }

  function speak(color, t0) {
    const buf = voiceBuffers[color];
    if (buf) {
      const src = audioCtx.createBufferSource();
      src.buffer = buf;
      src.connect(master);
      src.start(t0);
      return;
    }
    if (!('speechSynthesis' in window)) return;
    const delayMs = Math.max(0, (t0 - audioCtx.currentTime) * 1000);
    setTimeout(() => {
      speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(SPOKEN[color]);
      u.rate = 1.05; u.volume = Math.min(1, volumeGain());
      speechSynthesis.speak(u);
    }, delayMs);
  }

  function playAlertSound(color) {
    if (!audioCtx || volumeGain() === 0) return;
    if (audioCtx.state === 'suspended') audioCtx.resume();
    let t = audioCtx.currentTime + 0.02;
    if (settings.beep) t += beep(color, t) + 0.12;
    if (settings.voice) speak(color, t);
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

  /** Mapping from source pixels to screen CSS pixels (object-fit cover or contain). */
  function screenMap(vw, vh, fit) {
    const W = window.innerWidth, H = window.innerHeight;
    const s = fit === 'contain' ? Math.min(W / vw, H / vh) : Math.max(W / vw, H / vh);
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

  function draw(vw, vh, fit, tile) {
    const W = window.innerWidth, H = window.innerHeight;
    octx.clearRect(0, 0, W, H);
    const m = screenMap(vw, vh, fit);

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
    if (sourceKind === 'image') return { el: els.photo, vw: els.photo.naturalWidth, vh: els.photo.naturalHeight, fit: 'contain' };
    const v = els.video;
    return { el: v, vw: v.videoWidth, vh: v.videoHeight, fit: sourceKind === 'file' ? 'contain' : 'cover' };
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
      let dets = await detectRegion(src.el, 0, 0, src.vw, src.vh, thresh);
      let tile = null;
      if (settings.zoom) {
        tile = zoomTile(src.vw, src.vh);
        dets = dets.concat(await detectRegion(src.el, tile.sx, tile.sy, tile.sw, tile.sh, thresh));
      }
      dets = nms(dets);
      for (const d of dets) Object.assign(d, classifyColor(src.el, src.vw, src.vh, d));
      const t1 = performance.now();

      if (dets.length) { shownDets = dets; shownAt = t1; }
      updateState(frameVote(dets), t1);
      draw(src.vw, src.vh, src.fit, tile);

      const dt = t1 - last; last = t1;
      fpsAvg = fpsAvg ? fpsAvg * 0.9 + (1000 / dt) * 0.1 : 1000 / dt;
      inferAvg = inferAvg ? inferAvg * 0.9 + (t1 - t0) * 0.1 : t1 - t0;
      els.status.textContent = settings.debug
        ? `${fpsAvg.toFixed(1)} fps · ${inferAvg.toFixed(0)} ms · ${tf.getBackend()}\n${src.vw}×${src.vh} · thr ${thresh.toFixed(2)} · ${dets.length} box`
        : (dets.length ? `${dets.length} light${dets.length > 1 ? 's' : ''} in view` : 'Scanning…');

      await nextFrame();
    }
  }
  const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));

  // ---------- Camera & lenses ----------
  // iOS lists every rear lens as its own camera ("Back Camera", "Back Telephoto Camera",
  // "Back Ultra Wide Camera"); labels become readable once camera permission is granted.
  const LENS_LABEL = { ultra: '0.5×', wide: '1×', tele: 'Tele' };
  const LENS_NAME = { ultra: 'Ultra wide', wide: 'Wide', tele: 'Telephoto' };
  let lenses = {};            // kind → deviceId, rear lenses only
  let activeLens = null;

  function lensKind(label) {
    const l = label.toLowerCase();
    if (/front|facetime|avant|vorder|frontal|anterior/.test(l)) return 'front';
    if (/ultra/.test(l)) return 'ultra';
    if (/tele/.test(l)) return 'tele';
    if (/dual|triple|double|doble|dreifach|tripla/.test(l)) return 'virtual';   // multi-lens virtual cameras
    return 'wide';
  }

  async function findLenses() {
    const found = {};
    const devices = await navigator.mediaDevices.enumerateDevices();
    for (const d of devices) {
      if (d.kind !== 'videoinput' || !d.label) continue;
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

  async function startCamera(kind = settings.lens) {
    stopSource();
    sourceKind = 'camera';
    els.photo.hidden = true; els.video.hidden = false;
    els.stage.classList.remove('contain');

    if (!Object.keys(lenses).length) lenses = await findLenses().catch(() => ({}));
    const id = lenses[kind] || null;
    try {
      stream = await openStream(id);
    } catch (err) {
      if (!id) throw err;
      stream = await openStream(null);          // lens unavailable → default rear camera
    }
    if (!Object.keys(lenses).length) {
      // First run: permission was just granted, so the lens list is readable now.
      lenses = await findLenses().catch(() => ({}));
      const opened = lensKind(stream.getVideoTracks()[0].label);
      if (lenses[kind] && opened !== kind) {
        stream.getTracks().forEach((t) => t.stop());
        stream = await openStream(lenses[kind]);
      }
    }
    activeLens = lensKind(stream.getVideoTracks()[0].label);
    els.video.srcObject = stream;
    await els.video.play();
    renderLensBar();
  }

  async function switchLens(kind) {
    if (sourceKind !== 'camera' || kind === activeLens) return;
    settings.lens = kind; settings.camZoom = 1; saveSettings();
    history.length = 0;
    try { await startCamera(kind); } catch (err) { els.status.textContent = cameraErrorText(err); }
  }

  function renderLensBar() {
    const kinds = ['ultra', 'wide', 'tele'].filter((k) => lenses[k]);
    els.lensButtons.textContent = '';
    for (const k of kinds) {
      const b = document.createElement('button');
      b.textContent = LENS_LABEL[k];
      b.setAttribute('role', 'radio');
      b.setAttribute('aria-label', `${LENS_NAME[k]} lens`);
      b.setAttribute('aria-checked', String(k === activeLens));
      b.addEventListener('click', () => switchLens(k));
      els.lensButtons.appendChild(b);
    }
    els.lensButtons.hidden = kinds.length < 2;
    setupZoom();
    els.lensBar.hidden = els.lensButtons.hidden && els.zoomRow.hidden;
  }

  /** Shows a zoom slider when the browser exposes the camera's zoom (iOS 17+ Safari). */
  function setupZoom() {
    const track = stream && stream.getVideoTracks()[0];
    const z = track && track.getCapabilities ? track.getCapabilities().zoom : null;
    if (!z || !(z.max > z.min)) { els.zoomRow.hidden = true; return; }
    const max = Math.min(z.max, 10);            // beyond ~10× digital zoom the image is too soft to help
    els.optCamZoom.min = z.min;
    els.optCamZoom.max = max;
    els.optCamZoom.step = z.step > 0 ? Math.max(z.step, 0.1) : 0.1;
    els.optCamZoom.value = Math.min(max, Math.max(z.min, settings.camZoom));
    applyZoom(+els.optCamZoom.value);
    els.zoomRow.hidden = false;
  }

  function applyZoom(v) {
    els.camZoomVal.textContent = `${v.toFixed(1)}×`;
    const track = stream && stream.getVideoTracks()[0];
    if (track) track.applyConstraints({ advanced: [{ zoom: v }] }).catch(() => {});
  }

  function stopSource() {
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
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
  els.btnSettings.addEventListener('click', () => { els.settings.hidden = false; });
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
  document.addEventListener('pointerdown', (e) => {
    if (!els.volPanel.hidden && !els.volPanel.contains(e.target) && !els.btnVolume.contains(e.target)) showVolumePanel(false);
  });
  els.btnMuteToggle.addEventListener('click', () => {
    settings.muted = !settings.muted;
    if (!settings.muted && settings.volume === 0) { settings.volume = 50; els.optVolume.value = 50; }
    saveSettings(); applyVolume(); updateVolumeUI();
    if (settings.muted && 'speechSynthesis' in window) speechSynthesis.cancel();
  });
  let testColor = 0;
  els.btnTestSound.addEventListener('click', () => {
    unlockAudio();
    playAlertSound(['red', 'yellow', 'green'][testColor++ % 3]);
  });
  els.optCamZoom.addEventListener('input', () => {
    settings.camZoom = +els.optCamZoom.value; saveSettings();
    applyZoom(settings.camZoom);
  });

  document.addEventListener('visibilitychange', async () => {
    if (document.visibilityState !== 'visible' || !running) return;
    requestWakeLock();
    if (audioCtx && audioCtx.state !== 'running') audioCtx.resume().catch(() => {});
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
    audioState: () => ({ gain: master && master.gain.value, voices: Object.keys(voiceBuffers), ctx: audioCtx && audioCtx.state }),
  };

  // ---------- Offline support ----------
  // Skipped on localhost during development (stale caches) unless ?sw=1 is given.
  const isDevHost = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname) && !/[?&]sw=1/.test(location.search);
  if ('serviceWorker' in navigator && window.isSecureContext && !isDevHost) {
    navigator.serviceWorker.register('sw.js').catch((err) => console.warn('SW registration failed', err));
  }
})();
