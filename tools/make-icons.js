// Generates the app icons (PNG) with no dependencies: node tools/make-icons.js
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

function crcTable() {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
}
const CRC = crcTable();
function crc32(buf) {
  let c = 0xffffffff;
  for (const b of buf) c = CRC[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type), data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}
function png(size, rgba) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0); ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  const raw = Buffer.alloc((size * 4 + 1) * size);
  for (let y = 0; y < size; y++) {
    raw[y * (size * 4 + 1)] = 0;
    rgba.copy(raw, y * (size * 4 + 1) + 1, y * size * 4, (y + 1) * size * 4);
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0)),
  ]);
}

function render(size) {
  const buf = Buffer.alloc(size * size * 4);
  const S = 4; // supersampling
  const u = size / 100;
  // Housing: rounded rect centered, lamps stacked vertically.
  const hx0 = 32 * u, hx1 = 68 * u, hy0 = 12 * u, hy1 = 88 * u, hr = 10 * u;
  const lamps = [
    { cy: 26 * u, col: [255, 59, 48], on: true },
    { cy: 50 * u, col: [255, 204, 0], on: false },
    { cy: 74 * u, col: [48, 209, 88], on: false },
  ];
  const lr = 9.5 * u, cx = 50 * u;
  function inRR(x, y) {
    if (x < hx0 || x > hx1 || y < hy0 || y > hy1) return false;
    const qx = Math.max(hx0 + hr - x, 0, x - (hx1 - hr));
    const qy = Math.max(hy0 + hr - y, 0, y - (hy1 - hr));
    return qx * qx + qy * qy <= hr * hr;
  }
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let r = 0, g = 0, b = 0;
      for (let sy = 0; sy < S; sy++) for (let sx = 0; sx < S; sx++) {
        const px = x + (sx + 0.5) / S, py = y + (sy + 0.5) / S;
        // background: dark blue-gray gradient
        let c = [20 + 14 * (py / size), 24 + 16 * (py / size), 34 + 24 * (py / size)];
        // glow from red lamp
        const gd = Math.hypot(px - cx, py - lamps[0].cy) / (30 * u);
        if (gd < 1) { const k = (1 - gd) ** 2 * 0.55; c = c.map((v, i) => v + (lamps[0].col[i] - v) * k); }
        if (inRR(px, py)) c = [28, 28, 30];
        for (const l of lamps) {
          const d = Math.hypot(px - cx, py - l.cy);
          if (d <= lr) {
            const k = l.on ? 1 : 0.28;
            c = l.col.map((v) => 28 + (v - 28) * k);
            if (l.on && d < lr * 0.45) c = c.map((v) => v + (255 - v) * 0.35);
          }
        }
        r += c[0]; g += c[1]; b += c[2];
      }
      const i = (y * size + x) * 4, n = S * S;
      buf[i] = r / n; buf[i + 1] = g / n; buf[i + 2] = b / n; buf[i + 3] = 255;
    }
  }
  return png(size, buf);
}

const out = path.join(__dirname, '..', 'icons');
fs.mkdirSync(out, { recursive: true });
for (const s of [180, 192, 512]) fs.writeFileSync(path.join(out, `icon-${s}.png`), render(s));
console.log('icons written to', out);
