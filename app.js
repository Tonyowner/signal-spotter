/* Signal Spotter — offline traffic light detection.
 *
 * Pipeline per frame:
 *   1. SSDLite MobileNet v2 (COCO, TensorFlow.js) finds "traffic light" boxes.
 *      Optionally a second, zoomed pass over the upper-center of the view
 *      catches distant lights that are too small at full-frame resolution.
 *   2. Each box is cropped from the full-resolution frame and logic.js finds its lit
 *      lamp (a round, emissive spot, not housing paint) and reads its color, checked
 *      against the lamp's position in the head.
 *   3. Each signal head is tracked over time (smoothing, flashing, US sequence rules),
 *      the heads are split into the through-lane signal and the left-turn signal, and
 *      the signal for the chosen lane is announced on screen, by voice and with tones.
 */
(() => {
  'use strict';

  const MODEL_URL = 'models/ssdlite_mobilenet_v2/model.json';
  const TRAFFIC_LIGHT_COL = 9;          // COCO id 10 → score column 9
  const MODEL_INPUT = 300;
  const COLORS = { red: '#ff3b30', yellow: '#ffcc00', green: '#30d158', unknown: '#a1a1a6' };
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
    laneBar: $('laneBar'), bannerRole: $('bannerRole'), bannerSub: $('bannerSub'),
  };
  const { PHRASES, baseColor } = SignalLogic;

  // ---------- Settings ----------
  const DEFAULTS = {
    voice: true, beep: true, repeat: false, zoom: true, sens: 6, debug: false,
    muted: false, volume: 80,          // alert volume, 0–100
    lens: 'wide', camZoom: 1,          // rear lens ('ultra' | 'wide' | 'tele') and its zoom factor
    lane: 'straight',                  // 'straight' | 'left' (US: turn-only lane on the left)
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
    els.optVolume.addEventListener('change', () => playAlert({ color: 'green', primaryKey: 'main-green' }));  // preview
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
  function contained(a, b) {
    const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
    const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    return (ix * iy) / Math.min(a.w * a.h, b.w * b.h);
  }
  function nms(dets, thr = 0.45) {
    dets.sort((a, b) => b.score - a.score);
    const keep = [];
    for (const d of dets) if (keep.every((k) => iou(k, d) < thr && contained(k, d) < 0.7)) keep.push(d);
    return keep.slice(0, 10);
  }

  /** Square tile over the upper-center of the frame, where traffic lights usually are. */
  function zoomTile(vw, vh) {
    const side = Math.min(vw, vh) * 0.5;
    const cx = vw / 2, cy = vh * (vh > vw ? 0.36 : 0.4);
    return { sx: cx - side / 2, sy: Math.max(0, cy - side / 2), sw: side, sh: side };
  }

  // ---------- Lamp classification ----------
  const clsCanvas = document.createElement('canvas');
  const clsCtx = clsCanvas.getContext('2d', { willReadFrequently: true });

  /** Crops box d of the source (long side 96 px) and finds its lit lamp (see logic.js). */
  function classifyHead(src, vw, vh, d) {
    const x = Math.max(0, Math.floor(d.x)), y = Math.max(0, Math.floor(d.y));
    const w = Math.min(vw - x, Math.ceil(d.w)), h = Math.min(vh - y, Math.ceil(d.h));
    if (w < 3 || h < 3) return { lit: false, color: 'unknown', conf: 0 };
    const scale = 96 / Math.max(w, h);
    const cw = Math.max(4, Math.round(w * scale)), ch = Math.max(4, Math.round(h * scale));
    clsCanvas.width = cw; clsCanvas.height = ch;
    clsCtx.drawImage(src, x, y, w, h, 0, 0, cw, ch);
    return SignalLogic.classifyLamp(clsCtx.getImageData(0, 0, cw, ch).data, cw, ch);
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
  const voicePcm = {};                      // phrase key (see PHRASES) → Float32Array at SR
  Object.keys(PHRASES).forEach((k) => {
    fetch(`audio/${k}.wav`).then((r) => (r.ok ? r.arrayBuffer() : null))
      .then((b) => { if (b) voicePcm[k] = parseWav(b); }).catch(() => {});
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

  /**
   * Renders one alert at the given gain: tones for the primary color, its phrase, then the
   * secondary phrase a little quieter (e.g. "Green light ... Left turn, red"). Returns PCM or null.
   * alert: {color, primaryKey, secondaryKey}; color/primaryKey are null for a mention only.
   */
  function renderAlert(alert, gain, withVoice) {
    const lead = 0.08;                       // iOS can clip the very start of a clip
    const tones = settings.beep && alert.color ? BEEPS[alert.color] : null;
    const beepLen = tones ? Math.max(...tones.map(([, at, dur]) => at + dur)) + 0.12 : 0;
    const v1 = withVoice && settings.voice && alert.primaryKey ? voicePcm[alert.primaryKey] : null;
    const v2 = withVoice && settings.voice && alert.secondaryKey ? voicePcm[alert.secondaryKey] : null;
    if (!tones && !v1 && !v2) return null;
    const start1 = Math.round((lead + beepLen) * SR);
    const start2 = start1 + (v1 ? v1.length + Math.round(0.25 * SR) : 0);
    const pcm = new Float32Array(start2 + (v2 ? v2.length : 0) + Math.round(0.05 * SR));
    if (tones) {
      for (const [freq, at, dur] of tones) {
        const start = Math.round((lead + at) * SR), n = Math.round(dur * SR);
        const peak = alert.color === 'red' ? 0.35 : 0.6;
        for (let i = 0; i < n; i++) {
          const t = i / SR;
          const env = t < 0.01 ? t / 0.01 : Math.exp(-6.9 * (t - 0.01) / (dur - 0.01));  // 10 ms attack, decay to −60 dB
          const ph = 2 * Math.PI * freq * t;
          const wave = alert.color === 'red' ? (Math.sin(ph) >= 0 ? 1 : -1) : Math.sin(ph);
          pcm[start + i] += wave * env * peak;
        }
      }
    }
    if (v1) pcm.set(v1, start1);
    if (v2) for (let i = 0; i < v2.length; i++) pcm[start2 + i] += v2[i] * 0.75;
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

  function speakFallback(alert) {
    if (!settings.voice || !('speechSynthesis' in window)) return;
    const text = [alert.primaryKey, alert.secondaryKey].filter(Boolean).map((k) => PHRASES[k]).join('. ');
    if (!text) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.05;
    speechSynthesis.speak(u);
  }

  const alertLog = [];                      // recent alerts, for diagnostics
  let pendingAlert = null;
  alertAudio.addEventListener('ended', () => {
    const next = pendingAlert;
    pendingAlert = null;
    if (next) playAlertNow(next);
  });
  function playAlert(alert) {
    alertLog.push([alert.primaryKey, alert.secondaryKey].filter(Boolean).join(' + '));
    if (alertLog.length > 12) alertLog.shift();
    if (!alertAudio.paused && !alert.primaryKey) { pendingAlert = alert; return; }   // let the current clip finish
    pendingAlert = null;
    playAlertNow(alert);
  }
  function playAlertNow(alert) {
    const gain = volumeGain();
    if (gain === 0) return;
    // Voice clips not loaded (yet)? Speak with the built-in voice and play only the tones.
    const clipsReady = [alert.primaryKey, alert.secondaryKey].every((k) => !k || voicePcm[k]);
    if (settings.voice && !clipsReady) speakFallback(alert);
    const pcm = renderAlert(alert, gain, clipsReady);
    if (!pcm) return;
    playPcm(pcm)
      .then(() => { audioStatus = 'ready (media playback)'; })
      .catch((err) => {
        if (err.name === 'AbortError') return;          // replaced by a newer alert
        audioStatus = `blocked: ${err.name} (using speech fallback)`;
        speakFallback(alert);
      });
  }

  function clearAnnouncement() {
    els.banner.className = 'none';
    els.edge.className = '';
    els.bannerRole.textContent = '';
    els.bannerText.textContent = 'Searching for traffic lights…';
    els.bannerSub.hidden = true;
  }

  // ---------- Signal heads, lanes & announcements ----------
  const tracker = new SignalLogic.HeadTracker();
  const STATE_LABEL = { red: 'RED', yellow: 'YELLOW', green: 'GREEN', 'flash-yellow': 'FLASHING YELLOW', 'flash-red': 'FLASHING RED' };
  const ROLE_CAPTION = { turn: 'LEFT TURN', straight: 'STRAIGHT' };
  const announcer = new SignalLogic.Announcer();

  function resetSignals() {
    tracker.reset();
    announcer.reset();
    clearAnnouncement();
  }

  function showBanner(r, pulse) {
    const base = baseColor(r.primary);
    els.banner.className = base + (r.primary.startsWith('flash') ? ' flash' : '');
    els.edge.className = base;
    els.bannerRole.textContent = ROLE_CAPTION[r.primaryRole] || '';
    els.bannerText.textContent = STATE_LABEL[r.primary];
    if (r.secondary) {
      els.bannerSub.hidden = false;
      els.bannerSub.className = baseColor(r.secondary);
      const who = r.secondaryRole === 'turn' ? 'Left turn' : 'Straight';
      els.bannerSub.textContent = `${who}: ${STATE_LABEL[r.secondary].toLowerCase()}`;
    } else {
      els.bannerSub.hidden = true;
    }
    if (pulse) {
      els.banner.classList.remove('pulse');
      void els.banner.offsetWidth;          // restart pulse animation
      els.banner.classList.add('pulse');
    }
  }

  function setLane(lane) {
    settings.lane = lane; saveSettings();
    els.laneBar.querySelectorAll('button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.lane === lane)));
    announcer.reset();                       // re-announce for the new lane
  }

  // ---------- Drawing ----------
  const octx = els.overlay.getContext('2d');

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

  function draw(vw, vh, fit, tile, z, tracks, roleMap, now) {
    const W = window.innerWidth, H = window.innerHeight;
    octx.clearRect(0, 0, W, H);
    const m = screenMap(vw, vh, fit, z);

    if (settings.debug && tile) {
      octx.setLineDash([6, 6]); octx.lineWidth = 1.5; octx.strokeStyle = 'rgba(255,255,255,0.5)';
      octx.strokeRect(m.dx + tile.sx * m.s, m.dy + tile.sy * m.s, tile.sw * m.s, tile.sh * m.s);
      octx.setLineDash([]);
    }

    for (const t of tracks) {
      if (now - t.lastSeen > BOX_HOLD_MS) continue;
      const d = t.box, pad = 6;
      const x = m.dx + d.x * m.s - pad, y = m.dy + d.y * m.s - pad;
      const w = d.w * m.s + pad * 2, h = d.h * m.s + pad * 2;
      const base = baseColor(t.state) || 'unknown';
      const role = roleMap.get(t.id);
      const col = COLORS[base];
      octx.save();
      if (role === 'other') octx.setLineDash([8, 6]);       // not part of the main / turn signals
      octx.shadowColor = col; octx.shadowBlur = t.state ? 18 : 0;
      octx.lineWidth = t.state ? 4 : 2; octx.strokeStyle = col;
      roundRect(octx, x, y, w, h, 10); octx.stroke();
      octx.restore();
      if (t.state) { octx.fillStyle = col + '22'; roundRect(octx, x, y, w, h, 10); octx.fill(); }

      let text = t.state ? STATE_LABEL[t.state] : 'LIGHT';
      if (role === 'turn') text = `TURN · ${text}`;
      if (settings.debug && t.lamp) {
        const L = t.lamp;
        text += ` ${Math.round(t.score * 100)}% ${L.lit ? `${L.color[0]}${Math.round(L.conf * 100)} c${L.contrast.toFixed(1)}` : 'off'}`;
        if (L.lamp) {                                     // where the lamp was found
          octx.strokeStyle = '#fff'; octx.lineWidth = 1.5;
          octx.beginPath();
          octx.arc(m.dx + (d.x + L.lamp.x * d.w) * m.s, m.dy + (d.y + L.lamp.y * d.h) * m.s, Math.max(3, L.lamp.r * Math.min(d.w, d.h) * m.s), 0, 2 * Math.PI);
          octx.stroke();
        }
      }
      octx.font = '700 15px -apple-system, system-ui, sans-serif';
      const tw = octx.measureText(text).width + 16;
      const ly = y - 28 < 4 ? y + h + 4 : y - 28;
      octx.fillStyle = col;
      roundRect(octx, x, ly, tw, 24, 8); octx.fill();
      octx.fillStyle = base === 'red' ? '#fff' : '#111';
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
      for (const d of dets) d.lamp = classifyHead(src.el, src.vw, src.vh, d);
      const t1 = performance.now();

      const tracks = tracker.update(dets, t1);
      const r = SignalLogic.assignRoles(tracks, t1, settings.lane, src.vw);
      const out = announcer.update(r, t1, settings.repeat ? 5000 : 0);
      if (out.banner === null) clearAnnouncement();
      else if (out.banner) showBanner(out.banner, out.pulse);
      if (out.alert) playAlert(out.alert);
      draw(src.vw, src.vh, src.fit, tile, z, tracks, r.roles, t1);

      const dt = t1 - last; last = t1;
      fpsAvg = fpsAvg ? fpsAvg * 0.9 + (1000 / dt) * 0.1 : 1000 / dt;
      inferAvg = inferAvg ? inferAvg * 0.9 + (t1 - t0) * 0.1 : t1 - t0;
      els.status.textContent = settings.debug
        ? `${fpsAvg.toFixed(1)} fps · ${inferAvg.toFixed(0)} ms · ${tf.getBackend()}\n${src.vw}×${src.vh} · thr ${thresh.toFixed(2)} · zoom ${z.toFixed(1)}× · ${dets.length} box\nmain ${r.main || '–'} · turn ${r.turn || '–'}`
        : (dets.length ? '' : 'Scanning…');

      await nextFrame();
    }
  }
  const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));

  // ---------- Camera, lenses & zoom ----------
  // Some iOS versions list each rear lens as its own camera ("Back Telephoto Camera", …), others
  // expose a single "Back Camera". Tele therefore uses the telephoto lens when it is listed and
  // otherwise zooms the main camera — with hardware zoom if the browser offers it, else a digital
  // crop. The crop is also what the detector sees, so distant lights become larger for it too.
  const TELE_ZOOM = 2, DIGITAL_MAX = 5, HW_MAX = 10;
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
    tracker.reset();
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
      `Version 1.3 · lane ${settings.lane} · ${tf.getBackend ? tf.getBackend() : '?'}`,
      `Cameras: ${cameraList.length ? cameraList.join(', ') : 'none listed yet'}`,
      `Rear lenses found: ${Object.keys(lenses).join(', ') || 'none'}`,
      `Active: ${track ? track.label || '(no label)' : 'no camera'}${s.width ? ` · ${s.width}×${s.height}` : ''}`,
      `Zoom: ${hwZoom ? `hardware ${hwZoom.min}–${hwZoom.max}×` : `digital (max ${DIGITAL_MAX}×)`} · now ${settings.camZoom}×`,
      `Recent alerts: ${alertLog.slice(-4).join(' | ') || 'none'}`,
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
    tracker.reset();
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
    resetSignals();
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
  // Test sound cycles through examples, including a left-turn mention.
  const TESTS = [
    { color: 'red', primaryKey: 'main-red' },
    { color: 'yellow', primaryKey: 'main-yellow' },
    { color: 'green', primaryKey: 'main-green', secondaryKey: 'turn-red' },
  ];
  let testIdx = 0;
  els.btnTestSound.addEventListener('click', () => playAlert(TESTS[testIdx++ % TESTS.length]));
  els.laneBar.addEventListener('click', (e) => {
    const b = e.target.closest('button[data-lane]');
    if (b) setLane(b.dataset.lane);
  });
  setLane(settings.lane);
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
    async analyze(el, { zoom = settings.zoom, digital = 1 } = {}) {
      await modelReady;
      const vw = el.naturalWidth || el.videoWidth || el.width, vh = el.naturalHeight || el.videoHeight || el.height;
      const cw = vw / digital, ch = vh / digital, cx = (vw - cw) / 2, cy = (vh - ch) / 2;   // same crop as Tele
      let dets = await detectRegion(el, cx, cy, cw, ch, minScore());
      if (zoom) { const t = zoomTile(cw, ch); dets = dets.concat(await detectRegion(el, cx + t.sx, cy + t.sy, t.sw, t.sh, minScore())); }
      dets = nms(dets);
      for (const d of dets) d.lamp = classifyHead(el, vw, vh, d);
      return { dets };
    },
    audioState: () => ({ status: audioStatus, voices: Object.keys(voicePcm), src: alertAudio.src.slice(0, 30), paused: alertAudio.paused }),
    diagnostics: () => diagnostics(),
    alerts: () => alertLog.slice(),
    tracks: () => tracker.tracks.map((t) => ({ id: t.id, x: Math.round(t.box.x), state: t.state, cand: t.cand, role: t.role || null,
      obs: t.obs.slice(-8).map((o) => o.kind[0] + (o.kind === 'on' || o.kind === 'weak' ? o.color[0] : '')).join('') })),
  };

  // ---------- Offline support ----------
  // Skipped on localhost during development (stale caches) unless ?sw=1 is given.
  const isDevHost = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname) && !/[?&]sw=1/.test(location.search);
  if ('serviceWorker' in navigator && window.isSecureContext && !isDevHost) {
    navigator.serviceWorker.register('sw.js').catch((err) => console.warn('SW registration failed', err));
  }
})();
