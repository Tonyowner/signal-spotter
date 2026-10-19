# Signal Spotter — offline traffic light detector for iPhone

A web app (PWA) that uses the iPhone camera to find traffic lights continuously. It outlines each
light on screen and announces its color (**red**, **yellow** or **green**) on screen, by voice and
with tones. All processing runs on the phone. After the first visit it works with no internet
connection.

> Assistive aid only. It can miss lights or misread them, for example in glare, fog, at long
> range, or with arrow and pedestrian signals. Never rely on it alone for safety decisions.

## How it works

| Step | What happens |
|---|---|
| Detect | **SSDLite MobileNet v2** (COCO-trained, TensorFlow.js on the phone's GPU through WebGL) scores every frame for the *traffic light* class. A second pass zooms 2× into the upper-centre of the view to catch distant lights (Long-range mode). |
| Classify | Each box is cropped from the full-resolution frame. Bright, saturated pixels are scored by hue (red / amber / green). The lamp's position in the housing (top / middle / bottom, or left / right) breaks ties. |
| Stabilise | A light must have the same color in 3 of the last 5 frames before it is announced. A change is announced at once, and the banner clears 1.5 s after the light leaves view. |
| Announce | Full-color banner, glowing screen edge and an outlined box. A voice clip says "Red light", "Yellow light" or "Green light". Tones: red = two low beeps, yellow = three medium, green = rising chime. All audio goes through one volume control. |

## Files

```
index.html, style.css, app.js     the app
sw.js, manifest.webmanifest        offline cache + "Add to Home Screen" support
vendor/tf.min.js                   TensorFlow.js 4.22.0 (bundled, no CDN)
models/ssdlite_mobilenet_v2/       detector weights (~18 MB, bundled)
icons/                             app icons (regenerate: node tools/make-icons.js)
audio/                             spoken alerts (regenerate on Windows: node tools/make-voice.js)
tools/serve.js                     local test server
```

## Put it on your iPhone

iOS only allows camera access on **https** pages. Host the folder once on any free static HTTPS
host. The model never leaves your phone; the host only delivers the files.

**Option A: Netlify Drop (no account needed to try)**
1. Go to <https://app.netlify.com/drop> on your computer.
2. Drag the whole `Cam` folder onto the page. You get an `https://….netlify.app` link.

**Option B: GitHub Pages**
1. Create a repository, upload the contents of this folder and push.
2. Repository → Settings → Pages → deploy from the `main` branch. You get `https://<user>.github.io/<repo>/`.

**Then on the iPhone:**
1. Open the link in **Safari** and wait for "Ready · WEBGL · works offline". The first load
   downloads about 20 MB and caches it.
2. Tap **Share → Add to Home Screen**.
3. Open **Signals** from the Home Screen, tap **Start camera** and allow camera access.
4. From now on it works in Airplane Mode.

Tips:
- Hold the phone upright with the light in the upper half of the screen. That area gets the extra
  long-range scan.
- **Lenses:** on iPhones with more than one rear lens, buttons above the banner switch between
  **0.5×** (ultra wide), **1×** and **Tele** (telephoto). Tele works best for lights far down the
  road. The app remembers your choice. When iOS allows web pages to zoom the camera (iOS 17+),
  a zoom slider appears under the lens buttons.
- **Volume:** tap 🔊 to open the volume panel. It has a slider, **Test sound** and **Mute**. The
  slider controls both the tones and the voice. The phone's volume buttons set the overall level
  as usual.
- Speech and tones also play with the silent switch on (iOS 17+).
- The screen stays awake while the app runs.
- ⚙︎ Settings: voice, tones, repeat every 5 s, long-range mode, sensitivity, diagnostics overlay.
- **Test with a photo or video** on the start screen runs the same detector on a file from your
  photo library.

## Test on your computer

```bash
node tools/serve.js 8080
```

Open <http://localhost:8080> in Chrome, Edge or Safari. A webcam works, or use *Test with a photo
or video*. The offline service worker is disabled on `localhost` so edits show up immediately. Add
`?sw=1` to test offline caching.

After changing files on a hosted copy, bump `VERSION` in `sw.js` so phones pick up the new shell.
