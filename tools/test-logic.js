// Unit tests for logic.js: node tools/test-logic.js
const assert = require('assert');
const L = require('../logic.js');

let failed = 0;
function test(name, fn) {
  try { fn(); console.log(`  ok   ${name}`); }
  catch (e) { failed++; console.log(`  FAIL ${name}\n       ${e.message}`); }
}

// ---------- synthetic signal heads ----------
// Renders a vertical 3-section head, w × h pixels, like a crop the app would classify.
function head({ w = 36, h = 96, housing = [58, 58, 62], bg = [120, 160, 210], lit = null, litRGB = null,
  border = null, sunPaint = false, sections = 3 }) {
  const px = new Uint8ClampedArray(w * h * 4);
  const set = (x, y, c) => { const i = (y * w + x) * 4; px[i] = c[0]; px[i + 1] = c[1]; px[i + 2] = c[2]; px[i + 3] = 255; };
  const pad = border ? 4 : 1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const inBack = x >= 0 && x < w && y >= 0 && y < h;
    let c = bg;
    if (inBack) c = housing;
    if (border && (x < 3 || x >= w - 3 || y < 3 || y >= h - 3)) c = border;   // retroreflective backplate border
    set(x, y, c);
  }
  const slot = (h - 2 * pad) / sections, r = Math.min((w - 2 * pad) / 2, slot / 2) * 0.82;
  for (let k = 0; k < sections; k++) {
    const cy = pad + slot * (k + 0.5), cx = w / 2;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy);
      // sunlit yellow visor arc above each lens
      if (sunPaint && d > r && d < r + 2.5 && y < cy - r * 0.4) set(x, y, [235, 200, 40]);
      if (d > r) continue;
      if (k === lit) {
        const t = d / r;                              // white core → colored halo
        const core = Math.max(0, 1 - t * 1.8);
        set(x, y, litRGB.map((v) => Math.round(v + (255 - v) * core)));
      } else {
        set(x, y, [18, 18, 18]);                      // unlit lens
      }
    }
  }
  return { px, w, h };
}
const classify = (img) => L.classifyLamp(img.px, img.w, img.h);
const YELLOW_HOUSING = [205, 170, 20];                // US "federal yellow" body in daylight
const RED = [255, 40, 30], AMBER = [255, 170, 0], GREEN = [20, 235, 170];

console.log('classifyLamp');
test('unlit yellow housing in sun is not a yellow light', () => {
  const r = classify(head({ housing: YELLOW_HOUSING, sunPaint: true }));
  assert.notStrictEqual(r.color, 'yellow', JSON.stringify(r));
});
test('unlit head with yellow reflective border is not a yellow light', () => {
  const r = classify(head({ border: [250, 215, 30] }));
  assert.notStrictEqual(r.color, 'yellow', JSON.stringify(r));
});
test('green lamp in a sunlit yellow housing reads green', () => {
  const r = classify(head({ housing: YELLOW_HOUSING, sunPaint: true, lit: 2, litRGB: GREEN }));
  assert.strictEqual(r.color, 'green', JSON.stringify(r));
  assert.ok(r.conf >= 0.5);
});
test('red lamp (top) reads red', () => {
  const r = classify(head({ lit: 0, litRGB: RED }));
  assert.strictEqual(r.color, 'red', JSON.stringify(r));
});
test('amber lamp (middle) reads yellow', () => {
  const r = classify(head({ lit: 1, litRGB: AMBER, housing: YELLOW_HOUSING }));
  assert.strictEqual(r.color, 'yellow', JSON.stringify(r));
});
test('green lamp with backplate border reads green', () => {
  const r = classify(head({ lit: 2, litRGB: GREEN, border: [250, 215, 30] }));
  assert.strictEqual(r.color, 'green', JSON.stringify(r));
});
test('yellow-green washed-out glow at the bottom is green, not yellow', () => {
  const r = classify(head({ lit: 2, litRGB: [170, 255, 60] }));     // hue ≈ 80°
  assert.strictEqual(r.color, 'green', JSON.stringify(r));
});
test('yellowish glow in the bottom slot is too doubtful for a steady yellow', () => {
  const r = classify(head({ lit: 2, litRGB: [255, 200, 40] }));     // amber-ish color where green belongs
  assert.ok(r.color !== 'yellow' || r.conf < 0.5, JSON.stringify(r));
});
test('4-section head: bottom green arrow position still reads green', () => {
  const r = classify(head({ h: 120, sections: 4, lit: 3, litRGB: GREEN }));
  assert.strictEqual(r.color, 'green', JSON.stringify(r));
});
test('4-section head: yellow arrow in an inner slot reads yellow', () => {
  const r = classify(head({ h: 120, sections: 4, lit: 2, litRGB: AMBER }));
  assert.strictEqual(r.color, 'yellow', JSON.stringify(r));
});

// A dotted LED red lamp no brighter than the sunlit yellow paint around it.
test('LED red lamp in a sunlit yellow housing reads red', () => {
  const img = head({ housing: [235, 200, 35], lit: 0, litRGB: [225, 30, 25] });
  const { px, w } = img;
  // punch a grid of dark gaps into the lit lamp, like individual LEDs
  for (let y = 0; y < 34; y++) for (let x = 0; x < w; x++) {
    const i = (y * w + x) * 4;
    if (px[i] > 200 && px[i + 1] < 120 && (x % 3 === 0 || y % 3 === 0)) { px[i] = 90; px[i + 1] = 15; px[i + 2] = 12; }
  }
  for (let i = 0; i < px.length; i += 4) if (px[i] === 255 && px[i + 1] > 200) { px[i] = 230; px[i + 1] = 60; px[i + 2] = 50; }  // no white core
  const r = classify(img);
  assert.strictEqual(r.color, 'red', JSON.stringify(r));
});

test('green left-arrow lamp (thin shape) in the bottom section reads green', () => {
  const w = 36, h = 96, px = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) { px[i * 4] = 40; px[i * 4 + 1] = 42; px[i * 4 + 2] = 45; px[i * 4 + 3] = 255; }
  const set = (x, y) => { const i = (y * w + x) * 4; px[i] = 40; px[i + 1] = 210; px[i + 2] = 160; };
  const cy = 80;
  for (let x = 8; x <= 28; x++) for (let y = cy - 1; y <= cy + 1; y++) set(x, y);           // shaft
  for (let k = 0; k <= 7; k++) for (let t = -1; t <= 1; t++) { set(8 + k, cy - k + t); set(8 + k, cy + k + t); }  // head
  const r = L.classifyLamp(px, w, h);
  assert.strictEqual(r.color, 'green', JSON.stringify(r));
});
test('red lamp in a tight box that touches both sides reads red', () => {
  const img = head({ w: 26, h: 80, lit: 0, litRGB: [220, 40, 60], bg: [200, 210, 220] });
  const r = classify(img);
  assert.strictEqual(r.color, 'red', JSON.stringify(r));
});

// Night: dark surroundings, box roughly around the glow only.
function nightGlow({ w = 60, h = 80, core = [255, 250, 230], mid, rim }) {
  const px = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const d = Math.hypot(x + 0.5 - w / 2, y + 0.5 - h / 2), i = (y * w + x) * 4;
    const c = d < 9 ? core : d < 15 ? mid : d < 22 ? rim : [12, 10, 14];
    px[i] = c[0]; px[i + 1] = c[1]; px[i + 2] = c[2]; px[i + 3] = 255;
  }
  return { px, w, h };
}
test('overexposed red lamp at night (yellow-white core, red rim) reads red', () => {
  const r = classify(nightGlow({ mid: [255, 190, 90], rim: [230, 30, 40] }));
  assert.strictEqual(r.color, 'red', JSON.stringify(r));
});
test('green glow at night with no visible housing reads green', () => {
  const r = classify(nightGlow({ core: [230, 255, 250], mid: [80, 255, 220], rim: [20, 200, 170] }));
  assert.strictEqual(r.color, 'green', JSON.stringify(r));
});

// ---------- tracker ----------
const box = (x, lamp, extra = {}) => ({ x, y: 100, w: 30, h: 80, score: 0.8, lamp, ...extra });
const on = (color) => ({ lit: true, color, conf: 0.9 });
const off = { lit: false, color: 'unknown', conf: 0 };

function run(tracker, frames, startT = 0, fps = 8) {
  let t = startT;
  for (const dets of frames) { t += 1000 / fps; tracker.update(dets, t); }
  return t;
}
const rep = (n, f) => Array.from({ length: n }, (_, i) => f(i));

console.log('HeadTracker');
test('steady green becomes the state within about half a second', () => {
  const tr = new L.HeadTracker();
  run(tr, rep(5, () => [box(500, on('green'))]));
  assert.strictEqual(tr.tracks.length, 1);
  assert.strictEqual(tr.tracks[0].state, 'green');
});
test('works at a low frame rate (2 fps): green, then yellow', () => {
  const tr = new L.HeadTracker();
  const t0 = run(tr, rep(6, () => [box(500, on('green'))]), 0, 2);
  assert.strictEqual(tr.tracks[0].state, 'green');
  run(tr, rep(4, () => [box(500, on('yellow'))]), t0, 2);
  assert.strictEqual(tr.tracks[0].state, 'yellow');
});
test('single-frame dropouts do not break a steady light', () => {
  const tr = new L.HeadTracker();
  run(tr, rep(40, (i) => [box(500, i % 7 === 3 ? off : on('green'))]));
  assert.strictEqual(tr.tracks[0].state, 'green');
});
test('flashing yellow arrow (1 Hz) is detected', () => {
  const tr = new L.HeadTracker();
  run(tr, rep(40, (i) => [box(500, Math.floor(i / 4) % 2 === 0 ? on('yellow') : off)]));   // 8 fps, 0.5 s on/off
  assert.strictEqual(tr.tracks[0].state, 'flash-yellow');
});
test('steady doubtful yellow in the bottom slot is never announced', () => {
  const tr = new L.HeadTracker();
  run(tr, rep(30, () => [box(500, { lit: true, color: 'yellow', conf: 0.45, weak: true })]));
  assert.strictEqual(tr.tracks[0].state, null);
});
test('flashing yellow arrow in the bottom slot (bimodal US head) is detected', () => {
  const tr = new L.HeadTracker();
  run(tr, rep(40, (i) => [box(500, Math.floor(i / 4) % 2 === 0 ? { lit: true, color: 'yellow', conf: 0.45, weak: true } : off)]));
  assert.strictEqual(tr.tracks[0].state, 'flash-yellow');
});
test('a distant light dropping in and out irregularly is not called flashing', () => {
  const tr = new L.HeadTracker();
  const pattern = 'oooxxoooooxxxoooxxooooooxxoooxxxxooooo';   // o = seen lit, x = not lit
  run(tr, [...pattern].map((c) => [box(500, c === 'o' ? on('red') : off)]));
  assert.notStrictEqual(tr.tracks[0].state, 'flash-red');
});
test('steady yellow is not called flashing', () => {
  const tr = new L.HeadTracker();
  run(tr, rep(30, () => [box(500, on('yellow'))]));
  assert.strictEqual(tr.tracks[0].state, 'yellow');
});
test('green → yellow (normal sequence) switches within 0.75 s', () => {
  const tr = new L.HeadTracker();
  const t = run(tr, rep(16, () => [box(500, on('green'))]));
  run(tr, rep(6, () => [box(500, on('yellow'))]), t);
  assert.strictEqual(tr.tracks[0].state, 'yellow');
});
test('red → yellow (impossible in the US) needs much longer', () => {
  const tr = new L.HeadTracker();
  let t = run(tr, rep(16, () => [box(500, on('red'))]));
  t = run(tr, rep(6, () => [box(500, on('yellow'))]), t);
  assert.strictEqual(tr.tracks[0].state, 'red', 'should not accept red → yellow after 0.75 s');
  run(tr, rep(14, () => [box(500, on('yellow'))]), t);
  assert.strictEqual(tr.tracks[0].state, 'yellow', 'accepted once it persists');
});
test('two heads are tracked separately', () => {
  const tr = new L.HeadTracker();
  run(tr, rep(8, () => [box(200, on('red')), box(600, on('green'))]));
  const states = tr.tracks.map((t) => t.state).sort();
  assert.deepStrictEqual(states, ['green', 'red']);
});

console.log('assignRoles (US: turn lane on the left)');
// Runs frames through the tracker and the lane roles (as the app does every frame).
function runRoles(tr, frames, lane = 'straight', startT = 0, fps = 8) {
  let t = startT, r = null;
  for (const dets of frames) { t += 1000 / fps; tr.update(dets, t); r = L.assignRoles(tr.tracks, t, lane, 1000); }
  return { t, r };
}
function scene(heads, lane = 'straight') {
  return runRoles(new L.HeadTracker(), rep(20, () => heads.map(([x, c]) => box(x, on(c)))), lane).r;
}
test('two through heads green + turn head red on the left', () => {
  const r = scene([[150, 'red'], [450, 'green'], [700, 'green']]);
  assert.strictEqual(r.primary, 'green');
  assert.strictEqual(r.secondary, 'red');
  assert.strictEqual(r.secondaryRole, 'turn');
});
test('one through head + one turn head: the right one is the through signal', () => {
  const r = scene([[300, 'green'], [600, 'red']]);
  assert.strictEqual(r.primary, 'red');
  assert.strictEqual(r.secondary, 'green');
});
test('left-turn lane mode announces the turn head first', () => {
  const r = scene([[150, 'green'], [450, 'red'], [700, 'red']], 'left');
  assert.strictEqual(r.primary, 'green');
  assert.strictEqual(r.primaryRole, 'turn');
  assert.strictEqual(r.secondary, 'red');
  assert.strictEqual(r.secondaryRole, 'straight');
});
test('all heads agree → no separate turn mention', () => {
  const r = scene([[150, 'red'], [450, 'red'], [700, 'red']]);
  assert.strictEqual(r.primary, 'red');
  assert.strictEqual(r.secondary, null);
});
test('turn head is not promoted to the through signal when the through heads become unreadable', () => {
  const tr = new L.HeadTracker();
  let { t, r } = runRoles(tr, rep(20, () => [box(150, on('red')), box(450, on('green')), box(700, on('green'))]));
  assert.strictEqual(r.primary, 'green');
  assert.strictEqual(r.turn, 'red');
  // through heads changing: unreadable for a moment, turn head still red
  ({ t, r } = runRoles(tr, rep(12, () => [box(150, on('red')), box(450, off), box(700, off)]), 'straight', t));
  assert.notStrictEqual(r.primary, 'red', JSON.stringify({ p: r.primary, s: r.secondary }));
  assert.strictEqual(r.turn, 'red');
});
test('heads changing a moment apart are not mistaken for a turn signal', () => {
  const tr = new L.HeadTracker();
  let { t } = runRoles(tr, rep(20, () => [box(200, on('red')), box(450, on('red')), box(700, on('red'))]));
  // the left head switches to green 0.5 s before the other two (staggered change)
  ({ t } = runRoles(tr, rep(4, () => [box(200, on('green')), box(450, on('red')), box(700, on('red'))]), 'straight', t));
  const { r } = runRoles(tr, rep(16, () => [box(200, on('green')), box(450, on('green')), box(700, on('green'))]), 'straight', t);
  assert.strictEqual(r.primary, 'green');
  assert.strictEqual(r.turn, null);
  assert.ok(tr.tracks.every((k) => k.role !== 'turn'));
});
test('a differing head on the right is not called a left turn', () => {
  const r = scene([[300, 'green'], [500, 'green'], [800, 'red']]);
  assert.strictEqual(r.primary, 'green');
  assert.strictEqual(r.secondary, null);
});

console.log('Announcer');
function announce(seq, lane = 'straight') {
  const a = new L.Announcer(), said = [];
  seq.forEach(([t, primary, secondary, pRole = 'main', sRole = 'turn']) => {
    const out = a.update({ primary, secondary, primaryRole: pRole, secondaryRole: secondary ? sRole : null }, t);
    if (out.alert) said.push([out.alert.primaryKey, out.alert.secondaryKey].filter(Boolean).join(' + '));
  });
  return said;
}
const every = (from, to, step, f) => { const r = []; for (let t = from; t <= to; t += step) r.push(f(t)); return r; };
test('announces once the signal holds, with a brief turn mention', () => {
  const said = announce(every(0, 2000, 100, (t) => [t, 'green', 'red']));
  assert.deepStrictEqual(said, ['main-green + turn-red']);
});
test('a flicker shorter than the hold time is not announced', () => {
  const said = announce([...every(0, 1000, 100, (t) => [t, 'red', null]), [1100, 'green', null], [1200, 'red', null], ...every(1300, 2000, 100, (t) => [t, 'red', null])]);
  assert.deepStrictEqual(said, ['main-red']);
});
test('the turn signal is not repeated when only the main signal changes', () => {
  const said = announce([...every(0, 2000, 100, (t) => [t, 'green', 'red']), ...every(2100, 4000, 100, (t) => [t, 'yellow', 'red'])]);
  assert.deepStrictEqual(said, ['main-green + turn-red', 'main-yellow']);
});
test('a change of the turn signal alone is mentioned', () => {
  const said = announce([...every(0, 3000, 100, (t) => [t, 'red', null]), ...every(3100, 6000, 100, (t) => [t, 'red', 'green'])]);
  assert.deepStrictEqual(said, ['main-red', 'turn-green']);
});

test('the turn line disappears at once when it matches the main signal', () => {
  const a = new L.Announcer();
  let last;
  for (let t = 0; t <= 2000; t += 200) a.update({ primary: 'green', secondary: 'red', primaryRole: 'main', secondaryRole: 'turn' }, t);
  for (let t = 2200; t <= 3000; t += 200) { const o = a.update({ primary: 'red', secondary: null, primaryRole: 'main', secondaryRole: null }, t); if (o.banner) last = o.banner; }
  assert.strictEqual(last.primary, 'red');
  assert.strictEqual(last.secondary, null);
});
test('the same signal coming back after a brief loss is not announced again', () => {
  const said = announce([...every(0, 2000, 100, (t) => [t, 'red', null]), ...every(2100, 4500, 100, (t) => [t, null, null]),
    ...every(4600, 6000, 100, (t) => [t, 'red', null]), ...every(6100, 8000, 100, (t) => [t, 'green', null])]);
  assert.deepStrictEqual(said, ['main-red', 'main-green']);
});

console.log(failed ? `\n${failed} test(s) failed` : '\nall tests passed');
process.exit(failed ? 1 : 0);
