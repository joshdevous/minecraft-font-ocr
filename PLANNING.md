# Minecraft Font OCR — Planning Document

## Why Traditional OCR Fails

Tesseract and AI-based OCR tools are trained on natural, anti-aliased, variable fonts with
typographic variation. The Minecraft font is the opposite: it's a **fixed, pixel-perfect bitmap**
with no anti-aliasing, a shadow baked in, arbitrary foreground colors, and potentially any
background. These properties break the assumptions of general OCR — but they also make a
purpose-built solution very achievable.

---

## 1. The Default Minecraft Font — Everything to Know

### 1.1 Font Type: Bitmap / Pixel Art

The default font is stored as a **PNG sprite sheet** in the game's assets:

```
assets/minecraft/textures/font/ascii.png   →  128×128px, 16×16 grid of 8×8 character cells
assets/minecraft/textures/font/accented.png / nonlatin_european.png / etc.
assets/minecraft/font/default.json         →  provider definition JSON
```

Each character lives in a fixed **8×8 pixel cell**. The *actual* rendered width per character is
determined by scanning the rightmost non-transparent column within that cell — so the font is
**variable-width but deterministic**.

Typical widths at 1× scale:
| Characters          | Rendered width (px) |
|---------------------|---------------------|
| `i`, `!`, `'`, `;`  | 2                   |
| `l`, `|`            | 3                   |
| Most lowercase      | 5                   |
| Most uppercase      | 5–6                 |
| Space               | 4                   |
| `@`, `~`, `m`, `w`  | 6                   |

There is always **1 pixel of horizontal gap** appended after each character.

### 1.2 GUI Scale & Pixel Scaling

Minecraft renders text using **integer pixel scaling only** — no sub-pixel rendering, no
anti-aliasing. This is very helpful for OCR.

| GUI Scale | Pixels per font-pixel |
|-----------|-----------------------|
| 1×        | 1 px                  |
| 2×        | 2 px (most common)    |
| 3×        | 3 px                  |
| 4×        | 4 px                  |

At GUI Scale 2×, each character cell is 16×16 real pixels (8×8 font pixels × 2).
The OCR must detect which scale is in use and downsample accordingly before matching.

### 1.3 Shadow Rendering

By default, Minecraft renders **every character twice**:

1. Shadow pass: same character, offset **(+1px right, +1px down)** (at 1× scale), rendered in a
   darkened version of the text color (each RGB channel divided by 4, then floored).
2. Normal pass: the actual character on top.

This means every colour has a predictable shadow colour:

```
shadow_r = text_r >> 2   (i.e. floor(r / 4))
shadow_g = text_g >> 2
shadow_b = text_b >> 2
```

The shadow is always behind-and-below, so it's identifiable geometrically. Shadow removal is an
early pre-processing step.

Some contexts do **not** draw shadows:
- Signs (wooden/stone/etc.) — text is drawn flat, no shadow
- Book & Quill pages
- Some GUI labels depending on the draw call

### 1.4 Text Colour

Minecraft's 16 named colours (the `§` / section-sign codes):

| Code | Name          | Hex       | RGB            |
|------|---------------|-----------|----------------|
| `§0` | Black         | `#000000` | 0, 0, 0        |
| `§1` | Dark Blue     | `#0000AA` | 0, 0, 170      |
| `§2` | Dark Green    | `#00AA00` | 0, 170, 0      |
| `§3` | Dark Aqua     | `#00AAAA` | 0, 170, 170    |
| `§4` | Dark Red      | `#AA0000` | 170, 0, 0      |
| `§5` | Dark Purple   | `#AA00AA` | 170, 0, 170    |
| `§6` | Gold          | `#FFAA00` | 255, 170, 0    |
| `§7` | Gray          | `#AAAAAA` | 170, 170, 170  |
| `§8` | Dark Gray     | `#555555` | 85, 85, 85     |
| `§9` | Blue          | `#5555FF` | 85, 85, 255    |
| `§a` | Green         | `#55FF55` | 85, 255, 85    |
| `§b` | Aqua          | `#55FFFF` | 85, 255, 255   |
| `§c` | Red           | `#FF5555` | 255, 85, 85    |
| `§d` | Light Purple  | `#FF55FF` | 255, 85, 255   |
| `§e` | Yellow        | `#FFFF55` | 255, 255, 85   |
| `§f` | White         | `#FFFFFF` | 255, 255, 255  |

Additionally, **arbitrary RGB** colours are possible via JSON text components in modern Minecraft
(1.16+), which means coloured text isn't always one of the 16 exact values.

The shadow for each colour is deterministic — always `(r>>2, g>>2, b>>2)`. Exception: black text
(`§0`) has a black shadow, making it invisible on dark backgrounds.

Colour output strategy:
- For a given character region, take a histogram of non-background, non-shadow pixels and
  identify the dominant colour — that's the foreground RGB.
- Always output the **raw detected hex** value, regardless of whether it matches a named colour.
  Callers can snap to the nearest named colour themselves if they want to.

### 1.5 Text Formatting / Styling

#### Bold (`§l`)
Bold is achieved by rendering the character **twice**: once normally, then again shifted **1px to
the right**. This effectively widens each character by 1 pixel and gives a "thicker" appearance.
Key implication: a bold character's bounding box is 1px wider than the standard template.

#### Italic (`§o`)
A **shear (skew) transform** is applied: the top of the character is shifted ~1–2px to the right
relative to the bottom. Exact shear: each row is shifted right by `floor((baseline_row - row) / 2)`
pixels. Italic characters bleed into adjacent cells horizontally.

#### Underline (`§n`)
A **1px-tall horizontal line** drawn at `baseline + 1px` (just below the text), spanning the full
rendered width of the character (including inter-character gap).

#### Strikethrough (`§m`)
A **1px-tall horizontal line** drawn at approximately `baseline - 3px` (through the middle of the
character height), spanning the full rendered width.

#### Obfuscated (`§k`)
Characters randomly cycle through other characters of the **same pixel width** at ~20 FPS. This
cannot be OCR'd — it must be detected as "obfuscated" and output as a placeholder.

#### Reset (`§r`)
Resets all active formatting. Not visually observable, but affects the output encoding.

### 1.6 Where Text Appears (Rendering Contexts)

| Context              | Has shadow | Typical background         | Notes                              |
|----------------------|------------|----------------------------|------------------------------------|
| Chat                 | Yes        | Translucent dark gray      | Most common OCR target             |
| Signs                | No         | Wood/stone/crimson texture | Complex background, no shadow      |
| Books & Quills       | No         | Paper texture              | Dark text on light bg              |
| Item tooltips        | Yes        | Dark translucent purple    | Gradient border                    |
| HUD (hotbar labels)  | Yes        | Transparent / game world   | Very noisy background              |
| Scoreboards          | Yes        | Translucent dark           | Structured layout                  |
| Title / Subtitle     | Yes        | Transparent / game world   | Large text, easier to detect scale |
| Name tags            | Yes        | Transparent 3D world       | Perspective distortion possible    |
| Boss bar             | Yes        | Boss bar texture           |                                    |
| Tab list             | Yes        | Translucent dark           |                                    |

### 1.7 Background Contrast & Camouflaged Text

This is a real concern — even to the human eye, certain colour combinations are nearly unreadable
against certain in-world backgrounds (e.g. dark gray text `§8` over a dark stone wall, or white
text over a snow/quartz background). Here is how we handle each class of problem:

#### The Shadow as a Contrast Anchor (the key technique)

Even when the **text colour** is similar to the background, the **shadow** creates a *different*
contrast relationship. The shadow is always at 1/4 brightness, so:

- Bright text on bright background → shadow is dark → **shadow contrasts strongly**
- Dark text on dark background → shadow is near-black → **genuinely hard case** (see below)
- Bright text on dark background → text contrasts strongly → easy
- Dark text on bright background → text contrasts; shadow is harder to see but not needed

Strategy: **detect shadows first**, then infer text location from shadow position.

1. Scan for pixel pairs `(x, y)` and `(x+1, y+1)` where `pixel[x+1,y+1] ≈ pixel[x,y] >> 2`.
   This is a very specific geometric and photometric relationship — false positives from natural
   backgrounds are rare.
2. Once shadow pixels are identified, we know the text pixel is exactly at `(x, y - 1) → (x-1, y-1)`
   relative to the shadow, i.e. **(-1, -1)** offset from shadow.
3. Even if the text pixel itself is invisible against the background, the shadow anchors the
   character position — we can then match templates against the *expected* text position.

This means **highly camouflaged text is still locatable as long as its shadow contrasts**.

#### Chat Window Background — The Backing Rectangle

The chat window always draws a **uniform semi-transparent fill** behind each line of text before
rendering it. The fill colour is typically `rgba(0, 0, 0, ~100)` (dark, semi-transparent) blended
over the game world via alpha compositing.

To exploit this:
1. Detect the chat backing rectangle by looking for a rectangular region of consistent
   semi-transparent dark color.
2. Sample the mean background color of the backing rect in areas we know to be text-free
   (between lines, before the first character column).
3. Use this sampled background as the reference for foreground detection — subtract it from
   the region before running the pipeline.

This eliminates most of the world-background bleed-through problem for chat.

#### The Genuine Hard Case: Dark Text on Dark Background

The only situation where we genuinely struggle:
- Text is `§0` (black), `§1` (dark blue), `§4` (dark red), or `§8` (dark gray)
- The chat backing rect is set to low opacity, and the world behind it is also dark
- Shadow is also dark — neither the text nor its shadow contrast with the background

Mitigations:
- The shadow at `(r>>2, g>>2, b>>2)` of an already-dark colour is extremely close to `(0,0,0)`,
  so even a tiny luminance difference can be amplified by boosting contrast on the crop before
  processing.
- If the chat background is detectable, we can account for the compositing and work on a
  background-subtracted version.
- If none of this works, the character is reported with low confidence and flagged as uncertain.
  This is the honest outcome — if a human also can't read it, our OCR shouldn't claim to.

#### Colour-Background Matching for Non-Chat Contexts

For contexts without a definite backing rect (HUD labels, nametags, floating text):
- Use the **inter-character gap pixels** as background samples (the 1px gap between chars is
  always background-coloured — this is reliable even over complex scenes).
- Use **inter-line pixels** as additional background samples.
- Build a small background colour model per text region and subtract before binarization.
---

## 2. The OCR Problem Decomposed

```
Input: Full screenshot OR pre-cropped region
        │
        ▼
[0] Text Region Detection
        │  Locate all text regions in the image (shadow-pattern scan + context detection)
        │  Each detected region is passed through the rest of the pipeline independently
        ▼
[1] Scale Detection
        │  Detect GUI scale (1×–4×), downsample to 1× font pixels
        ▼
[2] Background Estimation
        │  Sample per-region background colour; detect chat/tooltip backing rects
        ▼
[3] Shadow-First Foreground Extraction
        │  Locate shadows (pixel-pair photometric test); derive text position from shadow;
        │  remove shadow layer to isolate clean foreground pixels
        ▼
[4] Line Detection
        │  Find horizontal bands containing text (8px tall at 1×)
        ▼
[5] Colour Segmentation
        │  Within each line, identify runs of consistent foreground colour
        ▼
[6] Character Segmentation
        │  Split each colour-run into individual character bounding boxes
        │  (using known inter-character 1px gap + variable width logic)
        ▼
[7] Template Matching
        │  Compare each character candidate to font atlas templates
        │  (Hamming distance on binary pixel grid)
        ▼
[8] Formatting Detection
        │  Bold (width), italic (shear), underline (extra row), strikethrough
        ▼
[9] Output Assembly
           Produce plain text, or text with § codes, or structured JSON
```

### Step 1 — Scale Detection

**Try-all-scales**: Attempt matching at scales 1–4 and pick the scale with the highest confidence
match score. There are only 4 possible integer scales, so this is fast and robust — no need for
frequency analysis or anchor detection.

Alternatively, the user can supply the scale directly, which is valid for batch processing of
same-source screenshots.

### Step 2 — Background Estimation

- **Chat / tooltip / scoreboard**: detect the backing rectangle; sample mean background from
  text-free rows (inter-line gaps) within it.
- **HUD / nametags / unknown**: sample the **1px inter-character gap columns** (which are always
  background-coloured) and **inter-line rows** as background reference points.
- Build a per-region background colour estimate. All foreground detection is relative to this
  estimate, not a global threshold.

### Step 3 — Shadow-First Foreground Extraction

For contexts with shadows:

1. Scan pixel pairs `(x,y)` + `(x+1,y+1)`: if `pixel[x+1,y+1] ≈ background + (pixel[x,y] - background) >> 2`,
   it's a shadow pair. (This formulation handles the fact that shadow is composited *over* the
   background, not drawn into a vacuum.)
2. Mark all shadow pixels. The text pixels are then located at the (-1,-1) offset from shadow.
3. Even where text pixels are indistinguishable from the background, their *positions* are now
   known via the shadow anchors — use these positions for template matching.
4. Remove shadow pixels from the working image to leave clean foreground.

**Compositing tolerance note:** The shadow formula above is exact only on solid or known
backgrounds. For translucent contexts (chat backing rect with alpha composited over the game
world), the final pixel values depend on what's behind the UI element. In practice this means:
- Work on the difference image (subtract estimated background) before shadow detection, or
- Use a **tolerance parameter** for the shadow-pair test (e.g., allow ±5 per channel)

The tolerance should be calibrated against real screenshots. Start with ±4 per channel.

For contexts without shadow: skip steps 1–3, go straight to foreground detection by contrast
against the estimated background.

### Step 4 — Line Detection

At 1× scale, each text line occupies exactly **8 pixels of height**. Lines are separated by
0–1px vertical gaps (no baseline descenders in the Minecraft font — it's strictly 8px tall).

Strategy:
- Project pixel density onto the Y axis (sum of foreground pixels per row).
- Find bands of ~8 consecutive rows with non-zero density.
- Cluster into individual lines.

### Step 0 — Text Region Detection (Optional — Full Screenshot Only)

> This step runs **before Step 1** when a full screenshot is provided. It is skipped entirely
> for pre-cropped regions. It is deferred to Phase 5 since the core pipeline works without it.

Text region detection is **optional** and only needed when given a full screenshot with no
coordinates. For most real use cases (chat OCR, tooltip reading, scoreboard parsing), the user
will either crop the region or provide coordinates. If given a pre-cropped region, the pipeline
processes all text in the crop directly starting at Step 1.

When enabled, two complementary approaches:

**Shadow-pattern scan (primary):**
The shadow relationship `pixel[x+1,y+1] ≈ pixel[x,y] >> 2` is a very specific photometric
signature that rarely occurs in natural pixel art backgrounds. Scan the image for neighbourhoods
that satisfy this relationship above a density threshold to identify candidate text regions.

**Context-template detection (secondary):**
For known UI contexts, detect their bounding frames:
- Chat: look for the semi-transparent backing rectangle (dark uniform region in bottom-left,
  containing horizontal bands of the same background colour).
- Tooltip: look for the characteristic double-pixel border (outer dark purple, inner lighter).
- Scoreboard: look for the right-side backing rect pattern.

Output: a list of bounding boxes, each tagged with a likely context type (chat / tooltip /
scoreboard / unknown). Unknown regions are processed with the generic shadow-scan pipeline.

### Step 5 — Colour Segmentation

Within a line, each character has a single dominant foreground colour. Horizontally scan for
colour transitions. Group consecutive columns sharing the same dominant colour into "colour runs."
This also helps with bold detection (bold pixels are the same colour as normal).

### Step 6 — Character Segmentation

The character segmentation is the trickiest part because:
- Characters are **variable width** (not fixed)
- Bold is 1px wider
- Italic bleeds into adjacent cells
- The 1px gap between characters is always the same colour as the background

Strategy: **sliding window with 1px gap validation.**

1. Use the known font atlas to know all possible character widths (2–6px at 1×).
2. At the current X position, try matching each template (all widths) against the pixel data.
3. Verify that the column immediately after the candidate character's width is a **gap column**
   (all background-coloured pixels). This is the 1px inter-character gap — it acts as a hard
   boundary and eliminates most ambiguity.
4. Accept the best-scoring match whose gap column validates, advance X by `width + 1`, repeat.

The 1px gap constraint makes this much simpler than general variable-width segmentation — DP or
beam search should not be needed unless real-world testing reveals ambiguity the gap check
doesn't resolve.

### Step 7 — Template Matching

Data source: extract `ascii.png` directly from the Minecraft JAR (it's a zip file). Parse
`default.json` for provider metadata.

For each character in the atlas:
- Extract its 8px-tall slice (from column 0 to its known width).
- Binarize (set pixel = 1 if any channel > threshold, else 0).
- Store as a `numpy` binary array indexed by character.

At match time:
- Binarize the candidate region the same way.
- Compute **Hamming distance** (XOR + popcount) between candidate and each template.
- The template with the lowest Hamming distance wins.
- Confidence = `1 - (hamming / total_pixels)`.

Handle bold by pre-computing **bold templates**: `template | (template << 1)` (OR with itself
shifted right by 1 column).

Handle italic by pre-computing **italic templates**: apply the known shear transform to each
template.

### Step 8 — Formatting Detection

| Feature       | Detection method                                                              |
|---------------|-------------------------------------------------------------------------------|
| Bold          | Character region is 1px wider than standard template width                   |
| Italic        | Pixel columns in character are sheared (top cols offset right vs bottom)      |
| Underline     | Row at `baseline + 1` is fully occupied across the character's width          |
| Strikethrough | Row at `baseline - 3` (mid-height) is fully occupied across character width   |
| Obfuscated    | Very low confidence match across all templates; random-looking pixel pattern  |
| Colour        | Dominant non-background colour of the character's pixel region                |

---

## 3. Data Sources

### 3.1 Extracting the Font Atlas

The Minecraft JAR is a ZIP archive:

```python
import zipfile
from PIL import Image
import io

with zipfile.ZipFile("minecraft-1.21.jar", "r") as jar:
    with jar.open("assets/minecraft/textures/font/ascii.png") as f:
        atlas = Image.open(io.BytesIO(f.read())).convert("RGBA")
```

The atlas is 128×128px. Characters are laid out in a 16×16 grid:
- Character `n` (0-indexed) is at cell `(n % 16, n // 16)`.
- Each cell is 8×8px.
- The Unicode code point is the same as the cell index for the ASCII range (code points 0–255).
- The `chars` array in `default.json` maps cell positions to Unicode strings for non-ASCII chars.

**Assumption:** This project targets the vanilla Minecraft font atlas (128×128px, 8×8 cells).
Some resource packs use higher-resolution font textures (e.g., 16×16 cells in a 256×256 atlas).
Custom resource pack fonts are **out of scope** — the atlas loader should validate dimensions on
load and reject non-standard atlases with a clear error message.

### 3.2 Character Width Calculation

```python
def get_char_width(atlas, char_index):
    col = char_index % 16
    row = char_index // 16
    cell = atlas.crop((col * 8, row * 8, (col + 1) * 8, (row + 1) * 8))
    pixels = cell.load()
    for x in range(7, -1, -1):  # scan right to left
        for y in range(8):
            if pixels[x, y][3] > 0:  # non-transparent
                return x + 2  # +1 for the pixel, +1 for the inter-char gap
    return 4  # space fallback
```

---

## 4. Implementation Plan

### Phase 1 — Font Atlas & Synthetic Test Harness

- [ ] Extract `ascii.png` and `default.json` from Minecraft JAR
- [ ] Build character template library (binary arrays per character)
- [ ] Compute and store character widths
- [ ] Generate bold and italic variants of each template
- [ ] Build a **synthetic renderer** that uses the extracted atlas to render known strings at
      all scales, colours, and formatting combos → generates test images with perfect ground truth
- [ ] Unit tests: verify template extraction, verify synthetic renderer output

### Phase 2 — End-to-End Pipeline on Synthetic Images

Get a working system on easy, controlled inputs first (known scale, known background, perfect
conditions).

- [ ] Scale detection (try-all-scales)
- [ ] Downsampling to 1× font resolution
- [ ] Shadow detection and removal
- [ ] Line detection (Y-axis projection)
- [ ] Character segmentation (sliding window + gap validation)
- [ ] Template matching with confidence scoring
- [ ] Plain text output
- [ ] Validate end-to-end against synthetic ground truth

### Phase 3 — Robustness on Real Screenshots

- [ ] Background estimation (chat/tooltip backing rects, inter-gap sampling)
- [ ] Shadow compositing tolerance tuning against real screenshots
- [ ] Handle noisy/translucent backgrounds
- [ ] Confidence threshold calibration (start at 0.85, tune against corpus)

### Phase 4 — Formatting, Colour & Output

- [ ] Bold / italic variant matching
- [ ] Formatting detection (underline, strikethrough)
- [ ] Colour identification per character (raw hex output)
- [ ] § code-annotated output
- [ ] Structured JSON output (character, colour, bold, italic, underline, strikethrough)
- [ ] Confidence reporting per character

### Phase 5 — Context Specialisation & Full-Screenshot Detection

- [ ] Chat window (shadow, translucent background)
- [ ] Signs (no shadow, textured background — bundle vanilla sign background templates)
- [ ] Tooltips (shadow, purple gradient border)
- [ ] Books (no shadow, paper background)
- [ ] Full-screenshot text region detection (Step 0 — shadow-pattern scan + context templates)

---

## 5. Technical Stack (Proposed)

```
Python 3.11+
├── Pillow (PIL)      — image loading, cropping, basic manipulation
├── NumPy             — binary array operations, Hamming distance, template math
├── OpenCV (cv2)      — binarization cleanup, morphological ops, connected components,
│                        matchTemplate for scale detection
├── SciPy             — 2D correlation for sub-pixel template matching (optional)
└── zipfile (stdlib)  — JAR extraction
```

For testing / benchmarking:
```
pytest                — unit and integration tests
Hypothesis            — property-based testing on font rendering edge cases
```

For a potential CLI / library interface:
```
Click or argparse     — CLI wrapper
```

---

## 6. Known Hard Cases & Mitigations

| Problem                                   | Mitigation                                                                          |
|-------------------------------------------|-------------------------------------------------------------------------------------|
| JPEG compression artifacts                | Threshold tuning; slight blur before binarization; warn user to use PNG             |
| Bright text on bright background          | Shadow is dark → shadow-first detection still locates text reliably                 |
| Dark text on medium background            | Text contrasts fine; shadow-first still helps anchor positions                      |
| Dark text on dark background (worst case) | Contrast-boost the crop first; if unreadable, report as uncertain; see §1.7         |
| Black `§0` text (shadow also black)       | Shadow-first is blind here; fall back to inter-gap background sampling + edge detect|
| Text over noisy background (HUD)          | Sample background from inter-char/inter-line gaps; shadow-first for location        |
| Arbitrary RGB colour (1.16+)              | Histogram dominant foreground colour per character; output raw hex                  |
| Italic + bold combined                    | Pre-compute combined italic-bold templates                                          |
| Very small text (GUI scale 1×)            | Works natively; no downsampling needed                                              |
| Obfuscated text                           | Detect low confidence uniformly → output `§k[N chars]` placeholder                  |
| RGBA composited UI                        | Flatten against sampled background colour before processing                         |
| Perspective-warped nametags               | Out of scope for v1 — flag as unsupported context                                   |

---

## 7. Confidence & Validation Strategy

- **Synthetic ground truth**: use a font renderer (Python script that uses the extracted atlas
  to render known strings) to generate test images at all scales, all colours, all formatting
  combos → perfect ground truth.
- **Real screenshot corpus**: collect screenshots from in-game across contexts, annotate
  manually, test against those.
- **Confidence threshold**: if best-match Hamming distance is above a threshold, report the
  character as uncertain (e.g., `[?]`).
- **Output confidence per character** so callers can choose how to handle uncertain regions.

---

## 8. Decisions Made

1. **Arbitrary RGB colour**: output raw hex for every character — callers can snap to named
   colours if they want. Detection via dominant-colour histogram on non-background pixels.
2. **Input format**: accept full screenshots OR pre-cropped regions. Full-screenshot region
   detection (Step 0) is optional and deferred to Phase 5. Pre-cropped regions are the primary
   input path — the pipeline processes all text found in the crop.
3. **Resource pack support**: out of scope. Only the vanilla 128×128 `ascii.png` (8×8 cells) is
   supported. The atlas loader validates dimensions on load and rejects non-standard atlases.
4. **Sign backgrounds**: bundle known background templates for vanilla sign types (oak, spruce,
   birch, jungle, acacia, dark oak, crimson, warped, mangrove, cherry, bamboo, stone). Users
   do not need to provide their own background crops. Note: sign text position varies depending
   on line count, and sign textures tile — Phase 5 will need **per-sign-type region templates**
   (defining where text can appear within the sign, not just the background colour/texture).
5. **Obfuscated text output**: use count notation `§k[5]` — output should be machine-parseable,
   not visual.
6. **Confidence thresholds**: start at 0.85, calibrate against synthetic corpus first, then tune
   on real screenshots. Ship as a configurable parameter so callers can adjust.
7. **Scale detection**: use try-all-scales (match at 1×–4×, pick best score). Only 4 options, so
   it's fast and robust. No frequency analysis needed.
8. **Character segmentation**: sliding window with 1px gap validation. DP/beam search deferred
   unless real-world testing reveals ambiguity the gap check doesn't resolve.

## 9. Resolved Questions

*(Moved from open questions — these are now decided.)*

1. ~~**Sign backgrounds**~~ → Bundle vanilla templates. (Decision 4 above.)
2. ~~**Obfuscated text width**~~ → `§k[5]` count notation. (Decision 5 above.)
3. ~~**Confidence thresholds**~~ → Start at 0.85, tune, ship as configurable. (Decision 6 above.)
