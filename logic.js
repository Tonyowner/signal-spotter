/* Signal Spotter — signal logic with no DOM dependencies, so it can be unit-tested in Node.
 *
 *   classifyLamp  finds the lit lamp in a crop of one signal head and reads its color.
 *   HeadTracker   follows each signal head across frames: smoothing, flashing detection and
 *                 US signal-sequence rules (green → yellow → red → green).
 *   assignRoles   splits the heads in view into the through-lane signal ("main") and the
 *                 left-turn signal ("turn"), US layout: the turn-only lane is on the left.
 *
 * Loaded as a classic script in the app (window.SignalLogic) and with require() in tests/tools.
 */
(function (root, factory) {
  const lib = factory();
  if (typeof module === 'object' && module.exports) module.exports = lib;
  else root.SignalLogic = lib;
})(typeof self !== 'undefined' ? self : globalThis, () => {
  'use strict';

  // Spoken alerts. tools/make-voice.js renders these to audio/<key>.wav.
  const PHRASES = {
    'main-red': 'Red light',
    'main-yellow': 'Yellow light',
    'main-green': 'Green light',
    'main-flash-yellow': 'Flashing yellow',
    'main-flash-red': 'Flashing red',
    'turn-red': 'Left turn red',
    'turn-yellow': 'Left turn yellow',
    'turn-green': 'Left turn green',
    'turn-flash-yellow': 'Left turn flashing yellow',
    'turn-flash-red': 'Left turn flashing red',
    'straight-red': 'Straight red',
    'straight-yellow': 'Straight yellow',
    'straight-green': 'Straight green',
    'straight-flash-yellow': 'Straight flashing yellow',
    'straight-flash-red': 'Straight flashing red',
  };

  /** 'flash-yellow' → 'yellow' */
  const baseColor = (state) => (state ? state.replace('flash-', '') : null);

  // ---------- Lamp classification ----------

  /**
   * Hue bands of signal lamps as phone cameras render them. Red LEDs sit at 340–10° (orange-ish
   * when overexposed), amber at 30–50°, and US green LEDs are blue-green at 150–180°. The gaps
   * (20–24° and 54–90°) are deliberately left out: those colors are ambiguous, not votes.
   */
  function hueColor(h) {
    if (h < 20 || h >= 335) return 'red';
    if (h >= 24 && h < 54) return 'yellow';
    if (h >= 90 && h <= 200) return 'green';
    if (h >= 54 && h < 90) return 'yg';   // yellow-green: decided by position, if at all
    return null;
  }

  const OFF = Object.freeze({ lit: false, color: 'unknown', conf: 0, contrast: 0, pos: null, lamp: null });
  const COLOR_ID = { red: 1, yellow: 2, green: 3, yg: 4 };

  /** 4-neighbour connected components of a 0/1 mask. */
  function components(mask, w, h, V) {
    const n = w * h, lab = new Int32Array(n), stack = new Int32Array(n), comps = [];
    for (let i = 0; i < n; i++) {
      if (lab[i] || !mask[i]) continue;
      const id = comps.length + 1;
      let sp = 0;
      stack[sp++] = i; lab[i] = id;
      let area = 0, sx = 0, sy = 0, sv = 0, x0 = w, y0 = h, x1 = -1, y1 = -1;
      while (sp) {
        const j = stack[--sp], x = j % w, y = (j / w) | 0;
        area++; sx += x; sy += y; sv += V[j];
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
        if (x > 0 && !lab[j - 1] && mask[j - 1]) { lab[j - 1] = id; stack[sp++] = j - 1; }
        if (x < w - 1 && !lab[j + 1] && mask[j + 1]) { lab[j + 1] = id; stack[sp++] = j + 1; }
        if (y > 0 && !lab[j - w] && mask[j - w]) { lab[j - w] = id; stack[sp++] = j - w; }
        if (y < h - 1 && !lab[j + w] && mask[j + w]) { lab[j + w] = id; stack[sp++] = j + w; }
      }
      const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
      comps.push({ id, area, cx: sx / area, cy: sy / area, mean: sv / area, fill: area / (bw * bh), aspect: bw / bh,
        touches: (x0 === 0) + (y0 === 0) + (x1 === w - 1) + (y1 === h - 1), rc: Math.sqrt(area / Math.PI) });
    }
    return { lab, comps };
  }

  function dilate(m, w, h) {
    const o = new Uint8Array(m.length);
    for (let i = 0; i < m.length; i++) {
      if (!m[i]) continue;
      const x = i % w;
      o[i] = 1;
      if (x > 0) o[i - 1] = 1;
      if (x < w - 1) o[i + 1] = 1;
      if (i >= w) o[i - w] = 1;
      if (i + w < m.length) o[i + w] = 1;
    }
    return o;
  }

  /** Ring R1..R2 around (cx, cy): pixel count, mean brightness and fraction of pixels set in mask. */
  function ring(cx, cy, R1, R2, w, h, V, mask) {
    let sum = 0, cnt = 0, hit = 0;
    for (let y = Math.max(0, Math.floor(cy - R2)); y <= Math.min(h - 1, Math.ceil(cy + R2)); y++) {
      for (let x = Math.max(0, Math.floor(cx - R2)); x <= Math.min(w - 1, Math.ceil(cx + R2)); x++) {
        const d2 = (x - cx) ** 2 + (y - cy) ** 2;
        if (d2 < R1 * R1 || d2 > R2 * R2) continue;
        const i = y * w + x;
        sum += V[i]; cnt++;
        if (mask && mask[i]) hit++;
      }
    }
    return { n: cnt, mean: cnt ? sum / cnt : 0, frac: cnt ? hit / cnt : 0 };
  }

  /**
   * Finds the lit lamp in an RGBA crop (w × h) of one signal head and reads its color.
   *
   * Two kinds of candidate lamp are considered:
   *  - bright spots: round, emissive spots clearly brighter than the ring around them (works for
   *    overexposed white cores and at night);
   *  - colored spots: a round red / yellow / green patch the size of a lens that is NOT
   *    surrounded by the same color (finds LED lamps no brighter than sunlit paint around them).
   * Painted housings (often yellow in the US), reflective backplate borders and sky fail both
   * tests: they are large, thin, touch the box edges, or are surrounded by more of themselves.
   *
   * The color is read inside the chosen lamp and, for a box that is clearly a whole signal head,
   * must agree with the lamp's position (top red, middle yellow, bottom green).
   *
   * Returns {lit, color: 'red'|'yellow'|'green'|'unknown', conf, contrast, pos, lamp, why}.
   */
  function classifyLamp(data, w, h) {
    const n = w * h;
    if (n < 16) return OFF;
    const V = new Uint8Array(n), S = new Float32Array(n), Hc = new Uint8Array(n), Hu = new Int16Array(n);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      const r = data[p], g = data[p + 1], b = data[p + 2];
      const max = r > g ? (r > b ? r : b) : (g > b ? g : b);
      const min = r < g ? (r < b ? r : b) : (g < b ? g : b);
      V[i] = max;
      S[i] = max ? (max - min) / max : 0;
      if (max === min) { Hu[i] = -1; continue; }
      const d = max - min;
      let hh;
      if (max === r) hh = 60 * (((g - b) / d) % 6);
      else if (max === g) hh = 60 * ((b - r) / d + 2);
      else hh = 60 * ((r - g) / d + 4);
      if (hh < 0) hh += 360;
      Hu[i] = hh;
      Hc[i] = COLOR_ID[hueColor(hh)] || 0;
    }

    // 3×3 mean of brightness, so single noisy pixels do not form blobs.
    const Vs = new Float32Array(n);
    let vmax = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let s = 0, c = 0;
        for (let yy = Math.max(0, y - 1); yy <= Math.min(h - 1, y + 1); yy++) {
          for (let xx = Math.max(0, x - 1); xx <= Math.min(w - 1, x + 1); xx++) { s += V[yy * w + xx]; c++; }
        }
        const v = s / c;
        Vs[y * w + x] = v;
        if (v > vmax) vmax = v;
      }
    }
    if (vmax < 110) return OFF;                 // nothing bright enough to be a lit lamp

    const hist = new Uint32Array(256);
    for (let i = 0; i < n; i++) hist[V[i]]++;
    let med = 0, p75 = 0;
    for (let v = 0, acc = 0; v < 256; v++) {
      acc += hist[v];
      if (!med && acc >= n / 2) med = v;
      if (acc >= n * 0.75) { p75 = v; break; }
    }
    const dark = p75 < 45;                      // night: everything but the lamps is near black
    const T = Math.max(120, med + 0.6 * (vmax - med));

    const short = Math.min(w, h), long = Math.max(w, h);
    const vertical = h >= w * 1.3, horizontal = w >= h * 1.3;
    const minArea = Math.max(3, n * 0.002);
    const vlo = vmax * 0.4;
    // Lamps sit on the head's centre line; spots off to the side are background or glare.
    const central = (c) => {
      const off = vertical ? Math.abs(c.cx / w - 0.5) : horizontal ? Math.abs(c.cy / h - 0.5) : 0;
      return 1 - Math.min(1, off * 2.5) * 0.7;
    };
    // In a tight box a lamp may touch the box's sides; a large region touching two sides is background.
    const shapeOk = (c, minFill = 0.4) => c.area >= minArea && c.fill >= minFill && c.aspect >= 0.45 && c.aspect <= 2.2 &&
      !(c.touches >= 2 && c.area > n * 0.3) && c.rc * 2 <= short * 1.15;
    const diskSat = (cx, cy, R) => {
      let s = 0, k = 0;
      for (let y = Math.max(0, Math.floor(cy - R)); y <= Math.min(h - 1, Math.ceil(cy + R)); y++) {
        for (let x = Math.max(0, Math.floor(cx - R)); x <= Math.min(w - 1, Math.ceil(cx + R)); x++) {
          const i = y * w + x;
          if ((x - cx) ** 2 + (y - cy) ** 2 <= R * R && V[i] >= vlo) { s += S[i]; k++; }
        }
      }
      return k ? s / k : 0;
    };
    const cands = [];

    // (1) Bright spots.
    const bright = new Uint8Array(n);
    for (let i = 0; i < n; i++) bright[i] = Vs[i] >= T ? 1 : 0;
    const B = components(bright, w, h, V);
    for (const c of B.comps) {
      if (!shapeOk(c)) continue;
      const R = Math.min(short * 0.55, Math.max(c.rc + 1, c.rc * 1.5));   // core + colored halo
      const rg = ring(c.cx, c.cy, R * 1.2, R * 2, w, h, V);
      if (rg.n < 8) continue;
      const contrast = c.mean / Math.max(25, rg.mean);
      if (contrast < 1.25) continue;                                     // not emissive: paint, sky
      const score = contrast * (c.mean / 255) * Math.min(1, c.fill + 0.3) * (c.touches ? 0.6 : 1) *
        central(c) * (0.4 + diskSat(c.cx, c.cy, R));
      cands.push({ ...c, R, contrast, ringMean: rg.mean, score, lab: B.lab });
    }

    // (2) Colored spots.
    const vmin = Math.max(90, 0.35 * vmax), vlamp = Math.max(110, 0.55 * vmax);
    for (const id of [1, 2, 3]) {
      const m = new Uint8Array(n);
      let any = 0;
      for (let i = 0; i < n; i++) if (Hc[i] === id && S[i] >= 0.35 && V[i] >= vmin) { m[i] = 1; any++; }
      if (any < minArea) continue;
      const C = components(dilate(m, w, h), w, h, V);                   // dilation joins LED dots
      for (const c of C.comps) {
        if (!shapeOk(c, 0.3) || c.rc * 2 < short * 0.2 || c.mean < vlamp) continue;   // 0.3: arrows are not discs
        const R = Math.min(short * 0.55, c.rc * 1.15);
        const rg = ring(c.cx, c.cy, R * 1.2, R * 2, w, h, V, m);
        if (rg.n < 8 || rg.frac > 0.25) continue;                       // surrounded by the same color: paint
        const contrast = c.mean / Math.max(25, rg.mean);
        if (contrast < 0.95) continue;
        const score = (1 - rg.frac) * 1.5 * (c.mean / 255) * Math.min(1, c.fill + 0.3) * (c.touches ? 0.6 : 1) *
          central(c) * (0.4 + diskSat(c.cx, c.cy, R));
        cands.push({ ...c, R, contrast, ringMean: rg.mean, score, lab: C.lab });
      }
    }
    if (!cands.length) return OFF;
    const best = cands.reduce((a, b) => (b.score > a.score ? b : a));

    // Color, read in the lamp: its core at full weight, the glow around it fading with distance
    // (so housing paint at the edge barely counts). At night the surroundings are dark and the
    // rim of the glow shows the true color best (an overexposed red core looks orange or white).
    const { cx, cy, R, rc } = best;
    const night = best.ringMean < 70;
    const Rmax = night ? R * 1.5 : R;
    const wt = { red: 0, yellow: 0, green: 0, yg: 0, other: 0 };
    for (let y = Math.max(0, Math.floor(cy - Rmax)); y <= Math.min(h - 1, Math.ceil(cy + Rmax)); y++) {
      for (let x = Math.max(0, Math.floor(cx - Rmax)); x <= Math.min(w - 1, Math.ceil(cx + Rmax)); x++) {
        const d = Math.hypot(x - cx, y - cy);
        if (d > Rmax) continue;
        const i = y * w + x, v = V[i], s = S[i];
        if (Hu[i] < 0 || v < (night ? Math.max(50, vmax * 0.2) : vlo)) continue;
        let k;
        if (s >= 0.25) k = s * v / 255;
        else if (s >= 0.08 && v >= T) k = 0.3 * s * v / 255;            // tinted white core
        else continue;
        if (best.lab[i] !== best.id) {
          if (night) k *= s >= 0.4 ? 1.5 : 0.5;
          else k *= 0.5 * Math.max(0, 1 - (d - rc) / Math.max(1, R - rc));
        }
        wt[hueColor(Hu[i]) || 'other'] += k;
      }
    }
    const total = wt.red + wt.yellow + wt.green + wt.yg + wt.other;
    let hue = null, top = 0;
    for (const k of ['red', 'yellow', 'green']) if (wt[k] > top) { top = wt[k]; hue = k; }
    const colorConf = total > 0 ? top / total : 0;
    const enough = total >= 0.02 * Math.PI * R * R;

    // Position along the head, only when the box is clearly a whole head (elongated, much longer
    // than the lamp) and the scene is not dark (at night the housing is invisible, so the box is
    // often just part of the head). Vertical: top red / middle yellow / bottom green; horizontal:
    // left to right.
    let pos = null, posColor = null;
    if (rc * 2 <= long * 0.45 && !dark) {
      if (h >= w * 1.6) pos = cy / h; else if (w >= h * 1.6) pos = cx / w;
    }
    if (pos !== null) posColor = pos < 0.36 ? 'red' : pos > 0.64 ? 'green' : 'yellow';
    // Where each color may be (tolerant of slightly mis-cropped boxes): red in the top part,
    // green in the bottom part, yellow in the middle.
    const fits = (c) => pos === null || (c === 'red' ? pos < 0.45 : c === 'green' ? pos > 0.55 : pos > 0.28 && pos < 0.72);

    const res = { lit: true, color: 'unknown', conf: 0, contrast: best.contrast, pos, lamp: { x: cx / w, y: cy / h, r: R / short },
      why: { hue, colorConf: +colorConf.toFixed(2), posColor, enough, night, wt: Object.fromEntries(Object.entries(wt).map(([k, v]) => [k, +v.toFixed(1)])) } };
    if (!enough) return res;                          // white or colorless glow (e.g. walk signal)
    if (hue && wt.yg <= top && colorConf >= 0.5) {
      if (!posColor) { res.color = hue; res.conf = colorConf * 0.85; }
      else if (fits(hue)) { res.color = hue; res.conf = Math.min(1, 0.3 + 0.7 * colorConf); }
      else if (long >= short * 3.2 && colorConf >= 0.9 && best.contrast >= 1.8) {
        res.color = hue; res.conf = colorConf * 0.7;    // 4–5 section heads: slots differ from 3-section
      }
      else if (hue === 'yellow' && posColor === 'green' && colorConf >= 0.8) {
        // US left-turn heads often show the flashing yellow arrow in the bottom section. Too
        // doubtful to announce as a steady yellow, but it counts towards detecting flashing.
        res.color = 'yellow'; res.conf = 0.45; res.weak = true;
      }
      // otherwise color and position disagree → stay 'unknown' rather than guess
    } else if (wt.yg > top && (posColor === 'yellow' || posColor === 'green')) {
      res.color = posColor; res.conf = 0.55;          // yellow-green glow: position decides
    }
    return res;
  }

  // ---------- Head tracking ----------

  const STEADY_OBS = 5;         // a steady color is judged on the last few observations …
  const STEADY_MAX_AGE = 2500;  // … that are at most this old (works at any frame rate)
  const FLASH_WINDOW = 4000;    // ms of observations used to detect flashing
  const DROP_AFTER = 1500;      // forget a head not seen for this long (longer at low frame rates)
  const MEMORY = 3000;          // a state this recent still counts for sequence rules

  const NEXT = { green: 'yellow', yellow: 'red', red: 'green' };   // US signal sequence

  function plausible(from, to) {
    if (!from || !to || from === to) return true;
    if (from.startsWith('flash') || to.startsWith('flash')) return true;
    return NEXT[from] === to;
  }

  /** How long a new candidate state must hold before it is accepted. */
  function holdTime(from, to) {
    if (to === null) return 1200;           // keep the last state through short dropouts
    if (from === null) return 300;
    return plausible(from, to) ? 400 : 1500;
  }

  function iou(a, b) {
    const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
    const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    const inter = ix * iy;
    return inter / (a.w * a.h + b.w * b.h - inter);
  }

  function matchScore(t, d) {
    const o = iou(t, d);
    if (o >= 0.1) return 1 + o;
    const dist = Math.hypot(t.x + t.w / 2 - d.x - d.w / 2, t.y + t.h / 2 - d.y - d.h / 2) / Math.max(t.w, t.h, d.w, d.h);
    const ratio = (d.w * d.h) / (t.w * t.h);
    return dist < 0.75 && ratio > 0.4 && ratio < 2.5 ? 1 - dist : 0;
  }

  /**
   * Flashing (US signals flash 50–60 times a minute): at least three regular on/off cycles in the
   * last 4 s, with the head still seen during the off periods, and still going on. Distant lights
   * that drop in and out of detection are irregular and fail this.
   */
  function flashState(obs, now) {
    const runs = [];
    for (const o of obs) {
      if (now - o.t > FLASH_WINDOW || o.kind === 'unsure') continue;
      const kind = o.kind === 'weak' ? 'on' : o.kind;              // doubtful colors still show on/off
      const last = runs[runs.length - 1];
      if (last && last.kind === kind) { last.n++; last.end = o.t; if (kind === 'on') last.colors.push(o.color); }
      else runs.push({ kind, n: 1, start: o.t, end: o.t, colors: kind === 'on' ? [o.color] : [] });
    }
    // A single "off" frame is a dropout, not an off period: merge it into its neighbours.
    const merged = [];
    for (const r of runs) {
      const last = merged[merged.length - 1];
      if (r.kind === 'off' && r.n < 2) continue;
      if (last && last.kind === r.kind) { last.n += r.n; last.end = r.end; last.colors.push(...r.colors); }
      else merged.push({ ...r, colors: [...r.colors] });
    }
    const offs = merged.filter((r) => r.kind === 'off');
    const ons = merged.filter((r) => r.kind === 'on');
    if (offs.length < 3 || ons.length < 3) return null;
    let onSum = 0, offSum = 0;
    for (let i = 1; i < merged.length - 1; i++) {                  // complete on / off periods
      const dur = merged[i + 1].start - merged[i].start;
      if (dur < 250 || dur > 1300) return null;
      if (merged[i].kind === 'on') onSum += dur; else offSum += dur;
    }
    const offFrac = offSum / (onSum + offSum);
    if (offFrac < 0.35 || offFrac > 0.65) return null;             // a flasher is off about half the time
    const starts = ons.map((r) => r.start);
    const periods = starts.slice(1).map((v, i) => v - starts[i]);
    if (periods.length >= 2 && Math.max(...periods) > Math.min(...periods) * 1.5) return null;   // irregular
    if (now - merged[merged.length - 1].end > 700) return null;
    const colors = ons.flatMap((r) => r.colors);
    const count = (c) => colors.filter((x) => x === c).length;
    if (count('yellow') >= colors.length * 0.6) return 'flash-yellow';
    if (count('red') >= colors.length * 0.6) return 'flash-red';
    return null;
  }

  /**
   * Color of the latest run of at least two matching observations, provided the lamp was seen
   * lit in most recent frames; otherwise null. holdTime() then demands that it persists.
   */
  function rawState(obs, now) {
    const flash = flashState(obs, now);
    if (flash) return flash;
    const win = obs.filter((o) => now - o.t <= STEADY_MAX_AGE).slice(-STEADY_OBS);
    if (win.length < 2) return null;
    const on = win.filter((o) => o.kind === 'on');
    if (on.length < 2 || on.length < win.length * 0.5) return null;
    const c = on[on.length - 1].color;
    let run = 0;
    for (let i = on.length - 1; i >= 0 && on[i].color === c; i--) run++;
    return run >= 2 ? c : null;
  }

  class HeadTracker {
    constructor() { this.tracks = []; this.nextId = 1; this.lastT = null; this.dt = 0; }

    reset() { this.tracks = []; this.lastT = null; this.dt = 0; }

    /** dets: [{x, y, w, h, score, lamp}] in source pixels. Returns the current tracks. */
    update(dets, now) {
      const pairs = [];
      this.tracks.forEach((t, ti) => dets.forEach((d, di) => {
        const s = matchScore(t.box, d);
        if (s > 0) pairs.push([s, ti, di]);
      }));
      pairs.sort((a, b) => b[0] - a[0]);
      const tUsed = new Set(), dUsed = new Set();
      for (const [, ti, di] of pairs) {
        if (tUsed.has(ti) || dUsed.has(di)) continue;
        tUsed.add(ti); dUsed.add(di);
        this._observe(this.tracks[ti], dets[di], now, 0.5);
      }
      dets.forEach((d, di) => {
        if (dUsed.has(di)) return;
        const t = { id: this.nextId++, box: { x: d.x, y: d.y, w: d.w, h: d.h }, obs: [], state: null, cand: null,
          candSince: now, stateSince: now, lastKnown: null, lastKnownAt: 0, lastSeen: now };
        this.tracks.push(t);
        this._observe(t, d, now, 1);
      });
      if (this.lastT != null) this.dt = this.dt ? this.dt * 0.8 + (now - this.lastT) * 0.2 : now - this.lastT;
      this.lastT = now;
      const dropAfter = Math.max(DROP_AFTER, 3.5 * (this.dt || 0));
      this.tracks = this.tracks.filter((t) => now - t.lastSeen <= dropAfter);
      for (const t of this.tracks) this._settle(t, now);
      return this.tracks;
    }

    _observe(t, d, now, a) {
      t.box.x += (d.x - t.box.x) * a; t.box.y += (d.y - t.box.y) * a;
      t.box.w += (d.w - t.box.w) * a; t.box.h += (d.h - t.box.h) * a;
      t.lastSeen = now;
      t.score = d.score;
      const L = d.lamp || OFF;
      t.lamp = L;
      const kind = !L.lit ? 'off' : L.color !== 'unknown' && L.conf >= 0.5 ? 'on' : L.weak ? 'weak' : 'unsure';
      t.obs.push({ t: now, kind, color: L.color });
      while (t.obs.length && now - t.obs[0].t > FLASH_WINDOW + 500) t.obs.shift();
    }

    _settle(t, now) {
      if (t.state) { t.lastKnown = t.state; t.lastKnownAt = now; }
      const raw = rawState(t.obs, now);
      if (raw !== t.cand) { t.cand = raw; t.candSince = now; }
      if (t.cand === t.state) return;
      // Sequence rules also apply across a short dropout (yellow … nothing … green).
      const from = t.state || (now - t.lastKnownAt <= MEMORY ? t.lastKnown : null);
      if (now - t.candSince >= holdTime(from, t.cand)) { t.state = t.cand; t.stateSince = now; }
    }
  }

  // ---------- Lane roles ----------

  const TURN_CONFIRM = 1500;

  /**
   * US layout: through lanes have one or more signal heads (usually at least two showing the
   * same thing); a turn-only lane has its own head to the left of them. The main group is the
   * largest group of heads showing the same state (ties → the group nearest slightly right of
   * centre); heads to its left showing something else are the left-turn signal.
   *
   * A head must look like the turn signal for TURN_CONFIRM ms before it is treated as one (heads
   * of one signal change a fraction of a second apart). Once confirmed it keeps that role while
   * tracked (t.role), so it is never mistaken for the through signal when the through heads are
   * briefly unreadable.
   *
   * lane: 'straight' | 'left'. Returns {primary, secondary, primaryRole, secondaryRole, main,
   * turn, roles: Map(trackId → 'main' | 'turn' | 'other')}.
   */
  function assignRoles(tracks, now, lane, frameW) {
    const roles = new Map();
    const heads = tracks.filter((t) => t.state && now - t.lastSeen <= 600);
    const res = { primary: null, secondary: null, primaryRole: 'main', secondaryRole: null, main: null, turn: null, roles };
    if (!heads.length) return res;
    const cx = (t) => t.box.x + t.box.w / 2;
    const target = frameW * 0.55;
    const groups = new Map();
    for (const t of heads) {
      if (t.role === 'turn') continue;
      if (!groups.has(t.state)) groups.set(t.state, []);
      groups.get(t.state).push(t);
    }
    let main = null;
    for (const [state, hs] of groups) {
      const xs = hs.map(cx);
      const g = { state, heads: hs, n: hs.length, meanX: xs.reduce((a, b) => a + b, 0) / xs.length, minX: Math.min(...xs) };
      if (!main || g.n > main.n || (g.n === main.n && Math.abs(g.meanX - target) < Math.abs(main.meanX - target))) main = g;
    }
    const avgW = heads.reduce((a, t) => a + t.box.w, 0) / heads.length;
    for (const t of heads) {
      if (t.role === 'turn') continue;
      const looksTurn = main && t.state !== main.state && cx(t) < main.minX - 0.5 * avgW;
      if (!looksTurn) t.turnSince = null;
      else if (t.turnSince == null) t.turnSince = now;
      else if (now - t.turnSince >= TURN_CONFIRM) t.role = 'turn';
    }
    const turnHeads = heads.filter((t) => t.role === 'turn');
    let turn = null;
    if (turnHeads.length) {
      const counts = {};
      for (const t of turnHeads) counts[t.state] = (counts[t.state] || 0) + 1;
      turn = Object.keys(counts).sort((a, b) => counts[b] - counts[a])[0];
    }
    for (const t of heads) roles.set(t.id, turnHeads.includes(t) ? 'turn' : main && main.heads.includes(t) ? 'main' : 'other');

    res.main = main ? main.state : null;
    res.turn = turn;
    if (lane === 'left' && turn) {
      Object.assign(res, { primary: turn, primaryRole: 'turn', secondary: res.main, secondaryRole: res.main ? 'straight' : null });
    } else {
      Object.assign(res, { primary: res.main, primaryRole: 'main', secondary: turn, secondaryRole: turn ? 'turn' : null });
    }
    if (res.secondary === res.primary) { res.secondary = null; res.secondaryRole = null; }
    return res;
  }

  // ---------- Announcements ----------

  const PRIMARY_HOLD = 400;       // the lane's signal must hold this long before it is announced
  const SECONDARY_HOLD = 400;
  const MENTION_GAP = 2500;       // at most one separate "left turn ..." mention this often
  const LOST_AFTER = 1500;        // clear the banner after this long without the signal
  const FORGET_SAID = 3000;       // other lane's signal gone this long → mention it again later
  const QUIET_RETURN = 8000;      // the same signal back within this long after a loss → no repeat

  /**
   * Decides what to show and say. The chosen lane's signal is announced whenever it changes; the
   * other lane's signal is mentioned briefly, and only when it is new (first seen or changed).
   * update() returns {banner, pulse, alert}: banner is the state to show, null to clear, or
   * undefined for no change; alert is {color, primaryKey, secondaryKey} or null.
   */
  class Announcer {
    constructor() { this.reset(); this.lostKey = null; this.lostAt = -1e9; }

    reset() {
      Object.assign(this, { pKey: null, pCand: null, pSince: 0, primary: null, role: null, lastSeen: 0,
        sKey: null, sCand: null, sSince: 0, secondary: null, secondaryRole: null, secSeen: 0,
        said: null, mentionAt: -1e9, at: 0 });
    }

    update(r, now, repeatMs = 0) {
      const out = { banner: undefined, pulse: false, alert: null };
      if (r.primary) this.lastSeen = now;
      else if (this.primary && now - this.lastSeen > LOST_AFTER) {
        this.lostKey = this.pKey; this.lostAt = now;
        this.reset(); out.banner = null; return out;
      }

      const pk = r.primary ? r.primaryRole + '|' + r.primary : null;
      if (pk !== this.pCand) { this.pCand = pk; this.pSince = now; }
      let changed = false;
      if (pk && pk !== this.pKey && now - this.pSince >= PRIMARY_HOLD) {
        Object.assign(this, { pKey: pk, primary: r.primary, role: r.primaryRole });
        changed = true;
      }
      const sk = r.secondary ? r.secondaryRole + '|' + r.secondary : null;
      if (sk !== this.sCand) { this.sCand = sk; this.sSince = now; }
      if (!this.primary) return out;

      let secChanged = false;
      if (sk !== this.sKey && now - this.sSince >= (sk ? SECONDARY_HOLD : LOST_AFTER)) {
        Object.assign(this, { sKey: sk, secondary: r.secondary, secondaryRole: r.secondaryRole });
        secChanged = true;
      }
      if (this.secondary && this.secondary === this.primary) {    // now the same as the main signal: drop it
        Object.assign(this, { sKey: null, secondary: null, secondaryRole: null });
        secChanged = true;
      }
      if (this.secondary) this.secSeen = now;
      else if (this.said && now - this.secSeen > FORGET_SAID) this.said = null;

      if (changed || secChanged) {
        out.banner = { primary: this.primary, primaryRole: this.role, secondary: this.secondary, secondaryRole: this.secondaryRole };
      }
      const mention = this.secondary && this.sKey !== this.said ? this.secondaryRole + '-' + this.secondary : null;
      const primaryKey = this.role + '-' + this.primary;
      if (changed) {
        out.pulse = true;
        this.at = now;
        if (mention) { this.said = this.sKey; this.mentionAt = now; }
        const quiet = this.pKey === this.lostKey && now - this.lostAt < QUIET_RETURN;   // same signal, briefly lost
        out.alert = quiet ? (mention ? { color: null, primaryKey: null, secondaryKey: mention } : null)
          : { color: baseColor(this.primary), primaryKey, secondaryKey: mention };
      } else if (mention && now - this.mentionAt >= MENTION_GAP) {
        this.said = this.sKey; this.mentionAt = now;
        out.alert = { color: null, primaryKey: null, secondaryKey: mention };
      } else if (repeatMs && now - this.at >= repeatMs) {
        this.at = now;
        out.alert = { color: baseColor(this.primary), primaryKey, secondaryKey: null };
      }
      return out;
    }
  }

  return { PHRASES, baseColor, hueColor, classifyLamp, HeadTracker, assignRoles, Announcer, plausible, holdTime, flashState, rawState };
});
