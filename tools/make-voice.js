// Generates the spoken alert clips with Windows text-to-speech: node tools/make-voice.js [voice]
// Clips are trimmed and normalised so the in-app volume slider controls them like the beeps
// (iOS Safari ignores the volume of speechSynthesis).
const { execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const VOICE = process.argv[2] || 'Microsoft Zira Desktop';
const RATE = 24000;
const { PHRASES } = require('../logic.js');   // single source of the spoken alerts
const outDir = path.join(__dirname, '..', 'audio');
fs.mkdirSync(outDir, { recursive: true });

function synth(text, file) {
  const ps = `
    Add-Type -AssemblyName System.Speech
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $s.SelectVoice('${VOICE}')
    $s.Rate = 0
    $fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(${RATE}, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
    $s.SetOutputToWaveFile('${file.replace(/'/g, "''")}', $fmt)
    $s.Speak('${text}')
    $s.Dispose()`;
  execFileSync('powershell', ['-NoProfile', '-Command', ps], { stdio: 'inherit' });
}

function readPcm(file) {
  const buf = fs.readFileSync(file);
  let off = 12;
  while (off < buf.length) {
    const id = buf.toString('ascii', off, off + 4), size = buf.readUInt32LE(off + 4);
    if (id === 'data') {
      const n = Math.floor(size / 2), pcm = new Float32Array(n);
      for (let i = 0; i < n; i++) pcm[i] = buf.readInt16LE(off + 8 + i * 2) / 32768;
      return pcm;
    }
    off += 8 + size + (size & 1);
  }
  throw new Error('no data chunk in ' + file);
}

function writeWav(file, pcm) {
  const data = Buffer.alloc(pcm.length * 2);
  for (let i = 0; i < pcm.length; i++) data.writeInt16LE(Math.round(Math.max(-1, Math.min(1, pcm[i])) * 32767), i * 2);
  const h = Buffer.alloc(44);
  h.write('RIFF', 0); h.writeUInt32LE(36 + data.length, 4); h.write('WAVE', 8);
  h.write('fmt ', 12); h.writeUInt32LE(16, 16); h.writeUInt16LE(1, 20); h.writeUInt16LE(1, 22);
  h.writeUInt32LE(RATE, 24); h.writeUInt32LE(RATE * 2, 28); h.writeUInt16LE(2, 32); h.writeUInt16LE(16, 34);
  h.write('data', 36); h.writeUInt32LE(data.length, 40);
  fs.writeFileSync(file, Buffer.concat([h, data]));
}

function trimAndNormalise(pcm) {
  const thr = 0.01, pad = Math.round(RATE * 0.02);
  let a = 0, b = pcm.length - 1;
  while (a < b && Math.abs(pcm[a]) < thr) a++;
  while (b > a && Math.abs(pcm[b]) < thr) b--;
  const out = pcm.slice(Math.max(0, a - pad), Math.min(pcm.length, b + pad));
  let peak = 0;
  for (const v of out) peak = Math.max(peak, Math.abs(v));
  const g = 0.95 / (peak || 1);
  const fade = Math.round(RATE * 0.01);
  for (let i = 0; i < out.length; i++) {
    const f = Math.min(1, i / fade, (out.length - 1 - i) / fade);
    out[i] *= g * f;
  }
  return out;
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'voice-'));
for (const [key, text] of Object.entries(PHRASES)) {
  const raw = path.join(tmp, `${key}.wav`);
  synth(text, raw);
  const pcm = trimAndNormalise(readPcm(raw));
  writeWav(path.join(outDir, `${key}.wav`), pcm);
  console.log(`${key}.wav  ${(pcm.length / RATE).toFixed(2)} s`);
}
fs.rmSync(tmp, { recursive: true, force: true });
