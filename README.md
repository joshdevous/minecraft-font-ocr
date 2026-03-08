# minecraft-font-ocr

A purpose-built OCR engine for reading text rendered in the default Minecraft bitmap font.

---

## The Problem

General OCR tools (Tesseract, cloud vision APIs, AI models) perform poorly on Minecraft screenshots. The Minecraft font is a fixed, pixel-perfect bitmap with no anti-aliasing, baked-in drop shadows, arbitrary foreground colours, and integer-only pixel scaling — none of which matches the assumptions those tools are built around.

This project solves it the right way: by exploiting the font's deterministic properties directly.

## How It Works

The Minecraft font is stored as a PNG sprite sheet inside the game's JAR file. Every character is a known pixel pattern with a known width. This means OCR reduces to **template matching on a binary pixel grid** — no training, no models, just efficient exact comparison.

The pipeline:

1. **Scale detection** — Minecraft renders at integer GUI scales (1×–4×). The image is downsampled to 1× before processing.
2. **Background estimation** — Per-region background colour is sampled from inter-character gaps and inter-line rows, or from the chat/tooltip backing rectangle when present.
3. **Shadow-first foreground extraction** — Every Minecraft character casts a shadow offset (+1px right, +1px down) at exactly 1/4 the text brightness. Shadows are detected first using this photometric relationship, which anchors character positions even when the text colour blends into the background.
4. **Line detection** — Text lines are exactly 8px tall at 1×. Detected via Y-axis pixel density projection.
5. **Character segmentation** — Sliding window with 1px inter-character gap validation. The mandatory gap column between every character acts as a hard boundary, making segmentation straightforward without dynamic programming.
6. **Template matching** — Hamming distance on binarized pixel grids against pre-built templates extracted from the font atlas. Bold and italic templates are precomputed (bold = `template | template >> 1`; italic = shear transform applied to template).
7. **Formatting detection** — Bold, italic, underline, strikethrough, and obfuscated are all detectable geometrically or by confidence signature.
8. **Output** — Plain text, `§`-code annotated text, or structured JSON with per-character colour (raw hex), formatting flags, and confidence scores.

## What It Handles

- All GUI scales (1×–4×)
- All 16 named Minecraft colours + arbitrary RGB (1.16+)
- Bold, italic, underline, strikethrough, obfuscated formatting
- Text camouflaged against similar-coloured backgrounds (via shadow-anchor detection)
- Chat windows, tooltips, scoreboards, signs, books, HUD labels
- Full screenshots (automatic text region detection) or pre-cropped regions

## Why Not Just Use an AI?

Because this is a **solved problem given the constraints**. The font is deterministic, the rendering rules are documented, and the shadow geometry is exact. A template matcher with correct pre-processing will outperform a general vision model on this specific task — and it runs fast, offline, with no API costs.

## Stack

```
Python 3.11+  |  Pillow  |  NumPy  |  OpenCV
```

## Status

Planning phase. See [PLANNING.md](PLANNING.md) for the full design.

**Planned phases:**
- [ ] Phase 1 — Font atlas extraction & synthetic test harness
- [ ] Phase 2 — End-to-end pipeline on synthetic images
- [ ] Phase 3 — Robustness on real screenshots
- [ ] Phase 4 — Formatting, colour & output
- [ ] Phase 5 — Context specialisation & full-screenshot detection
