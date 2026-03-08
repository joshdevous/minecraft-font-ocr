"""
OCR pipeline — Phase 2: end-to-end recognition on clean synthetic images.

Pipeline steps (all at 1× font-pixel resolution after downsampling):
  0. Background estimation   — sample image corners
  1. Scale detection         — try 1×–4×, pick highest mean confidence
  2. Downsample to 1×        — stride-based, exact for integer scaling
  3. Foreground extraction   — shadow-first photometric test; or plain threshold
  4. Line detection          — Y-axis density projection, 8-px bands
  5. Character recognition   — sliding window + 1px gap validation + Hamming match
  6. Output assembly         — plain text + per-character detail + confidence

Phase 4 will add bold/italic variant matching and formatting detection.
Phase 3 will add robustness for real screenshots (JPEG noise, translucent BG).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .atlas import FontAtlas, CharTemplate, CELL_SIZE

_Color = tuple[int, int, int]

# Tolerance for "is this pixel the background colour?"
_BG_TOLERANCE: int = 8
# Per-channel tolerance for the shadow photometric test.
_SHADOW_TOLERANCE: int = 6
# Minimum run of blank columns to be considered a space character.
_SPACE_ADVANCE: int = 4


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

@dataclass
class CharResult:
    """Recognition result for a single character.

    Attributes:
        char:       The recognised Unicode character.
        confidence: Match quality in [0, 1]; 1 = perfect Hamming match.
        color:      Median RGB of the character's foreground pixels.
        x:          Left edge of the character in font-pixels within its line.
    """
    char: str
    confidence: float
    color: _Color
    x: int


@dataclass
class LineResult:
    """Recognition result for one horizontal line of text.

    Attributes:
        chars: Ordered list of per-character results.
        y:     Top of the line in font-pixels from the top of the image.
    """
    chars: list[CharResult] = field(default_factory=list)
    y: int = 0

    @property
    def text(self) -> str:
        return "".join(c.char for c in self.chars)

    @property
    def mean_confidence(self) -> float:
        if not self.chars:
            return 0.0
        return sum(c.confidence for c in self.chars) / len(self.chars)


@dataclass
class OCRResult:
    """Top-level OCR result for a whole image.

    Attributes:
        lines: One :class:`LineResult` per detected text line, top-to-bottom.
        scale: The GUI scale that was used (detected or supplied).
    """
    lines: list[LineResult] = field(default_factory=list)
    scale: int = 1

    @property
    def text(self) -> str:
        """Plain text with lines joined by newlines."""
        return "\n".join(line.text for line in self.lines)

    @property
    def mean_confidence(self) -> float:
        all_chars = [c for line in self.lines for c in line.chars]
        if not all_chars:
            return 0.0
        return sum(c.confidence for c in all_chars) / len(all_chars)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ocr(
    image: Image.Image,
    atlas: FontAtlas,
    scale: int | None = None,
    shadow: bool = True,
    background: _Color | None = None,
) -> OCRResult:
    """Run the OCR pipeline on *image* and return an :class:`OCRResult`.

    Args:
        image:      Input image (any PIL mode; converted to RGB internally).
        atlas:      Pre-built :class:`~minecraft_ocr.atlas.FontAtlas`.
        scale:      GUI scale 1–4.  If *None*, auto-detected by trying all four
                    and picking the scale with the highest mean confidence.
        shadow:     Whether to expect and remove a drop shadow.  Default True.
        background: RGB background colour.  If *None*, estimated from corners.

    Returns:
        :class:`OCRResult` exposing ``.text`` and per-character detail.
    """
    img_rgb = np.array(image.convert("RGB"))

    if background is None:
        background = _estimate_background(img_rgb)

    # Pre-group templates by bitmap width once — reused across all candidates.
    templates_by_width = _group_templates_by_width(atlas)

    if scale is None:
        scale = _detect_scale(img_rgb, atlas, background, shadow, templates_by_width)

    font_arr = _downsample(img_rgb, scale)
    fg = _extract_foreground(font_arr, background, shadow)
    lines_y = _detect_lines(fg)

    result = OCRResult(scale=scale)
    for y in lines_y:
        y_end = min(y + CELL_SIZE, fg.shape[0])
        line = _recognize_line(
            font_arr[y:y_end],
            fg[y:y_end],
            templates_by_width,
        )
        line.y = y
        # Strip trailing spaces: Minecraft text never has meaningful trailing
        # whitespace; trailing blanks arise from canvas padding in wide images.
        while line.chars and line.chars[-1].char == " ":
            line.chars.pop()
        result.lines.append(line)

    return result


# ---------------------------------------------------------------------------
# Pipeline steps (internal)
# ---------------------------------------------------------------------------

def _estimate_background(img: np.ndarray) -> _Color:
    """Estimate the background colour by taking the median of the four corners."""
    corners = np.array([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]])
    return tuple(int(v) for v in np.median(corners, axis=0))


def _downsample(img: np.ndarray, scale: int) -> np.ndarray:
    """Reduce to 1× font-pixel resolution via stride-based sampling.

    At scale N every font-pixel is an N×N block of identical real pixels
    (Minecraft uses integer-only scaling with no anti-aliasing), so taking
    every Nth pixel starting at (0, 0) gives the exact font-pixel data.
    """
    if scale == 1:
        return img
    return img[::scale, ::scale]


def _extract_foreground(
    img: np.ndarray,
    bg: _Color,
    shadow: bool,
    bg_tol: int = _BG_TOLERANCE,
    shadow_tol: int = _SHADOW_TOLERANCE,
) -> np.ndarray:
    """Return a boolean mask of clean text pixels at 1× font resolution.

    Pixels that are background-coloured or identified as shadow are excluded.
    When *shadow* is False, returns all non-background pixels.

    Shadow identification: for every non-background pixel T at (y, x), the
    pixel at (y+1, x+1) is considered a shadow if its colour is within
    *shadow_tol* of ``T >> 2`` channel-wise.  This is Minecraft's exact
    shadow formula on a dark (near-black) background.
    """
    bg_arr = np.array(bg, dtype=np.int32)
    arr = img.astype(np.int32)

    # Non-background: any channel deviates from background by more than tolerance.
    non_bg = (np.abs(arr - bg_arr) > bg_tol).any(axis=2)  # (H, W)

    if not shadow:
        return non_bg

    # Vectorised shadow detection.
    # For each (y, x) in [:-1, :-1]: expected shadow at (y+1, x+1) = T >> 2.
    text_region   = arr[:-1, :-1]   # (H-1, W-1, 3) candidate text pixels
    shadow_region = arr[1:,  1:]    # (H-1, W-1, 3) candidate shadow pixels
    expected      = text_region >> 2

    is_shadow_pair = (
        non_bg[:-1, :-1] &                                           # text pixel is fg
        (np.abs(shadow_region - expected) <= shadow_tol).all(axis=2) # shadow matches
    )

    shadow_mask = np.zeros(img.shape[:2], dtype=bool)
    shadow_mask[1:, 1:] = is_shadow_pair

    return non_bg & ~shadow_mask


def _detect_lines(fg: np.ndarray) -> list[int]:
    """Return the y-coordinate of the top of each detected text line.

    Projects foreground pixel counts onto the Y axis.  Each run of occupied
    rows is assumed to be one 8-px text line.  Once a line start is found,
    the scan jumps ahead by CELL_SIZE to avoid double-counting.
    """
    density = fg.sum(axis=1)   # (H,) non-background pixel count per row
    occupied = density > 0

    lines_y: list[int] = []
    i = 0
    h = len(occupied)
    while i < h:
        if occupied[i]:
            lines_y.append(i)
            i += CELL_SIZE
        else:
            i += 1
    return lines_y


def _group_templates_by_width(
    atlas: FontAtlas,
) -> dict[int, list[CharTemplate]]:
    """Group all atlas templates by bitmap width (``advance - 1``)."""
    groups: dict[int, list[CharTemplate]] = {}
    for tmpl in atlas.all_templates():
        w = tmpl.advance - 1
        groups.setdefault(w, []).append(tmpl)
    return groups


def _dominant_color(region: np.ndarray, fg_mask: np.ndarray) -> _Color:
    """Return the median colour of the foreground pixels in *region*."""
    if not fg_mask.any():
        return (255, 255, 255)
    pixels = region[fg_mask]  # (N, 3)
    return tuple(int(v) for v in np.median(pixels, axis=0))


def _best_match(
    candidate: np.ndarray,
    templates: list[CharTemplate],
) -> tuple[CharTemplate, float] | None:
    """Find the template with the lowest Hamming distance to *candidate*.

    Args:
        candidate: Boolean array (CELL_SIZE, W).
        templates: All templates with the same bitmap width W.

    Returns:
        ``(best_template, confidence)`` or *None* if no template fits.
    """
    best_tmpl: CharTemplate | None = None
    best_dist = float("inf")

    for tmpl in templates:
        if tmpl.bitmap.shape != candidate.shape:
            continue
        dist = int(np.sum(candidate != tmpl.bitmap))
        if dist < best_dist:
            best_dist = dist
            best_tmpl = tmpl

    if best_tmpl is None:
        return None

    n_pixels = candidate.size
    confidence = 1.0 - best_dist / max(n_pixels, 1)
    return best_tmpl, max(0.0, confidence)


def _recognize_line(
    line_band: np.ndarray,                     # (≤8, W, 3) RGB at 1× scale
    fg_band: np.ndarray,                       # (≤8, W) bool foreground mask
    templates_by_width: dict[int, list[CharTemplate]],
) -> LineResult:
    """Recognise all characters in a single 8-px-tall line band.

    Algorithm:
        Left-to-right sliding window.  At each non-empty column *x*:
          1. For every template width W, check that the column at *x+W* is
             empty (the mandatory 1px inter-character gap).
          2. Among widths that pass the gap check, find the template with the
             minimum normalised Hamming distance.
          3. Advance *x* by the winning template's advance (W+1).

        Blank column runs of ≥ _SPACE_ADVANCE pixels are emitted as spaces.
        Shorter blank runs (the 1px inter-character gap) are skipped silently.
    """
    result = LineResult()
    band_w = fg_band.shape[1]
    x = 0
    blank_run = 0

    while x < band_w:
        # --- Blank column ---
        if not fg_band[:, x].any():
            blank_run += 1
            x += 1
            continue

        # Coming out of a blank run — emit a space if the run was wide enough.
        if blank_run >= _SPACE_ADVANCE:
            result.chars.append(CharResult(
                char=" ",
                confidence=1.0,
                color=(255, 255, 255),
                x=x - blank_run,
            ))
        blank_run = 0

        # --- Template matching ---
        best_tmpl: CharTemplate | None = None
        best_conf = -1.0
        best_advance = 1  # fallback: advance 1 to avoid infinite loop

        for char_w, templates in templates_by_width.items():
            if x + char_w > band_w:
                continue

            # Gap validation: the column immediately after the glyph must be
            # background.  For the very last character in a line the gap may
            # fall at or beyond band_w — treat that as valid.
            gap_x = x + char_w
            if gap_x < band_w and fg_band[:, gap_x].any():
                continue  # column is occupied → wrong width

            candidate = fg_band[:CELL_SIZE, x:x + char_w]
            match = _best_match(candidate, templates)
            if match is None:
                continue

            tmpl, conf = match
            if conf > best_conf:
                best_conf = conf
                best_tmpl = tmpl
                best_advance = tmpl.advance

        if best_tmpl is None:
            x += 1
            continue

        char_w = best_tmpl.advance - 1
        color = _dominant_color(
            line_band[:, x:x + char_w],
            fg_band[:, x:x + char_w],
        )
        result.chars.append(CharResult(
            char=best_tmpl.char,
            confidence=best_conf,
            color=color,
            x=x,
        ))
        x += best_advance

    # Handle a trailing blank run at end of line.
    if blank_run >= _SPACE_ADVANCE:
        result.chars.append(CharResult(
            char=" ",
            confidence=1.0,
            color=(255, 255, 255),
            x=x - blank_run,
        ))

    return result


def _detect_scale(
    img: np.ndarray,
    atlas: FontAtlas,
    background: _Color,
    shadow: bool,
    templates_by_width: dict[int, list[CharTemplate]],
) -> int:
    """Auto-detect GUI scale by trying 1–4 and returning the best-scoring one.

    Score = mean confidence of the first line's characters at each scale.
    Falls back to scale 1 if no line is detected at any scale.
    """
    best_scale = 1
    best_score = -1.0

    for s in (1, 2, 3, 4):
        font_arr = _downsample(img, s)
        fg = _extract_foreground(font_arr, background, shadow)
        lines_y = _detect_lines(fg)
        if not lines_y:
            continue

        y = lines_y[0]
        y_end = min(y + CELL_SIZE, fg.shape[0])
        line = _recognize_line(font_arr[y:y_end], fg[y:y_end], templates_by_width)

        if not line.chars:
            continue

        # Exclude spaces from confidence scoring (they're empty, always 1.0).
        scored = [c for c in line.chars if c.char != " "]
        if not scored:
            continue

        score = sum(c.confidence for c in scored) / len(scored)
        if score > best_score:
            best_score = score
            best_scale = s

    return best_scale
