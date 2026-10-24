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
| Find the lamp | In each box, [logic.js](logic.js) looks for the *lit lamp*: a round spot that glows (much brighter than the ring around it), or a round red / yellow / green patch that is **not** surrounded by the same color. Painted housings (US yellow), reflective backplate borders, signs and sky fail these tests, so they are never read as a light. |
| Read the color | Color is read only inside the lamp. Hue bands match signal LEDs (US green is blue-green), and yellow-green glows are treated as uncertain rather than forced into yellow or green. In a whole 3-lamp head the color must agree with the lamp's position (top red, middle yellow, bottom green), or the head stays unknown. |
| Track each head | Every signal head is followed across frames with its own smoothed state. A color must persist (≈0.3–0.4 s) before it counts. Changes that break the US sequence (green → yellow → red → green), such as red → yellow, must persist 1.5 s. Flashing is recognized only as a regular ≈1-per-second on/off rhythm (so a distant light blinking in and out of view is not "flashing"), including the **flashing yellow arrow** in the bottom lamp of US left-turn heads. |
| Lanes | US layout: the through lanes' heads usually agree; a **turn-only lane's head is to their left**. The app groups the heads into *main* (through) and *turn*. A head must differ from the others for 1.5 s to count as the turn signal (heads of one signal change a moment apart). Once identified it stays the turn signal, so it is never announced as your light when the through heads are briefly unreadable. |
| Announce | Your lane's signal must hold for 0.4 s, then it is announced whenever it changes: full-color banner, glowing screen edge, outlined boxes (**TURN** label on the turn head), tones and voice. The other lane's signal is mentioned briefly ("Left turn red" / "Straight green") only when it is new. |

## Files

```
index.html, style.css, app.js     the app
logic.js                           lamp finding, head tracking, lane roles (no DOM; unit-tested)
sw.js, manifest.webmanifest        offline cache + "Add to Home Screen" support
vendor/tf.min.js                   TensorFlow.js 4.22.0 (bundled, no CDN)
models/ssdlite_mobilenet_v2/       detector weights (~18 MB, bundled)
icons/                             app icons (regenerate: node tools/make-icons.js)
audio/                             spoken alerts (regenerate on Windows: node tools/make-voice.js)
tools/serve.js                     local test server
tools/test-logic.js                unit tests for logic.js: node tools/test-logic.js
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
- **Lane switch (top left):** **Straight** (default) announces the through-lane signal and briefly
  mentions the left-turn signal. **◀ Left turn** announces the turn signal and briefly mentions
  the straight signal. If no separate turn signal is visible, the main signal is announced.
- Hold the phone upright with the lights in the upper half of the screen. That area gets the extra
  long-range scan.
- **Use Tele when the signals are far away.** At 1× a signal 100 m away is only a few pixels tall;
  Tele (2× by default) makes it readable while keeping the left-turn head in view. Switch back to
  1× when you are close to the intersection, or the signals leave the zoomed view.
- **Lenses & zoom:** buttons above the banner switch between **1×** and **Tele**, plus **0.5×**
  when iOS exposes the ultra-wide lens. Tele uses the telephoto lens if iOS lists it. Otherwise it
  zooms the main camera 3× (digitally, unless the browser offers hardware zoom). The detector then
  sees the zoomed view, so distant lights get easier to find. The slider sets any zoom from 1× to
  5×. The app remembers your choice.
- **Volume:** tap 🔊 to open the volume panel. It has a slider, **Test sound** and **Mute**. The
  slider controls both the tones and the voice. The phone's volume buttons set the overall level
  as usual (alerts use the *media* volume).
- Alerts play as a media clip, so they are audible even with the ring/silent switch on.
- **Settings › Diagnostics** shows which cameras iOS reports and the audio status. This helps if
  something doesn't work.
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
