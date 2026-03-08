"""
Synthetic Minecraft font renderer.

Produces pixel-perfect images using the extracted font atlas, matching
Minecraft's actual rendering behaviour.  This is the ground-truth generator
for Phase 1–2 of the test harness.

Rendering rules reproduced here:
  - Two-pass rendering: all shadows first, then all text on top.
  - Shadow offset: (+scale, +scale) real pixels (i.e. +1 font-pixel).
  - Shadow colour:  (r >> 2, g >> 2, b >> 2)  per channel.
  - Bold: advance width is tmpl.advance + 1 (one extra font-pixel column).
  - Italic: same advance as normal; top rows are sheared right (see atlas.py).
  - Underline: 1 font-pixel tall line at row 8 (just below the 8-row cell).
  - Strikethrough: 1 font-pixel tall line at row 3 (mid-height of the cell).
  - Decorations (underline/strikethrough) span the full advance width including
    the inter-character gap, matching Minecraft's behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from PIL import Image

from .atlas import FontAtlas, CharTemplate, _make_italic, CELL_SIZE

# Type alias for an RGB colour triple.
_Color = tuple[int, int, int]

# Vertical position (in font-pixels, 0 = top of cell) for decorations.
_STRIKETHROUGH_ROW: int = 3   # middle of the 8-pixel cell
_UNDERLINE_ROW: int = CELL_SIZE  # one row below the cell (row 8)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class TextSpan:
    """A run of text sharing a single colour and set of style flags.

    Attributes:
        text:          The characters to render.
        color:         Foreground RGB colour.
        bold:          Whether to apply bold (double-draw shifted 1 px right).
        italic:        Whether to apply italic (shear transform).
        underline:     Whether to draw an underline.
        strikethrough: Whether to draw a strikethrough.
    """
    text: str
    color: _Color = field(default=(255, 255, 255))
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_text(
    spans: Sequence[TextSpan],
    atlas: FontAtlas,
    scale: int = 2,
    shadow: bool = True,
    background: _Color = (0, 0, 0),
) -> Image.Image:
    """Render a sequence of *spans* and return an RGB PIL Image.

    The canvas is sized exactly to contain the rendered text:
      width  = total advance (font-px) × scale  [+ scale if shadow]
      height = 8 (font-px) × scale              [+ scale if shadow]

    Args:
        spans:      Ordered list of :class:`TextSpan` objects to render.
        atlas:      Pre-built :class:`FontAtlas` to source glyph bitmaps from.
        scale:      Integer GUI scale factor (1–4).  Defaults to 2.
        shadow:     Whether to render the drop shadow.  Defaults to True.
        background: RGB background fill colour.  Defaults to black.

    Returns:
        An RGB :class:`PIL.Image.Image`.

    Raises:
        ValueError: If *scale* is not in the range 1–4.
    """
    if not (1 <= scale <= 4):
        raise ValueError(f"Scale must be 1–4, got {scale}")

    total_font_w = _measure_width(spans, atlas)
    shadow_pad = scale if shadow else 0

    # Underline is drawn one row below the 8-px cell (at font-row 8), so the
    # canvas must be at least (CELL_SIZE + 1) rows tall when underline is used.
    underline_extra = scale if any(s.underline for s in spans) else 0

    img_w = total_font_w * scale + shadow_pad
    img_h = CELL_SIZE * scale + max(shadow_pad, underline_extra)

    canvas = np.full((img_h, img_w, 3), background, dtype=np.uint8)

    # --- Pass 1: draw all shadows ---
    if shadow:
        x_font = 0
        for span in spans:
            shadow_color: _Color = (
                span.color[0] >> 2,
                span.color[1] >> 2,
                span.color[2] >> 2,
            )
            for ch in span.text:
                tmpl = atlas.get(ch)
                if tmpl is None:
                    x_font += 4
                    continue
                bitmap = _select_bitmap(tmpl, span.bold, span.italic)
                advance = _advance(tmpl, span.bold)
                # Blit shadow at (+scale, +scale) offset
                _blit(canvas, bitmap, x_font * scale + scale, scale, scale, shadow_color)
                # Decoration shadows
                if span.underline:
                    _draw_decoration(
                        canvas, x_font * scale + scale, scale + _UNDERLINE_ROW * scale,
                        advance * scale, scale, shadow_color, img_w, img_h,
                    )
                if span.strikethrough:
                    _draw_decoration(
                        canvas, x_font * scale + scale, scale + _STRIKETHROUGH_ROW * scale,
                        advance * scale, scale, shadow_color, img_w, img_h,
                    )
                x_font += advance

    # --- Pass 2: draw all text on top ---
    x_font = 0
    for span in spans:
        for ch in span.text:
            tmpl = atlas.get(ch)
            if tmpl is None:
                x_font += 4
                continue
            bitmap = _select_bitmap(tmpl, span.bold, span.italic)
            advance = _advance(tmpl, span.bold)
            _blit(canvas, bitmap, x_font * scale, 0, scale, span.color)
            if span.underline:
                _draw_decoration(
                    canvas, x_font * scale, _UNDERLINE_ROW * scale,
                    advance * scale, scale, span.color, img_w, img_h,
                )
            if span.strikethrough:
                _draw_decoration(
                    canvas, x_font * scale, _STRIKETHROUGH_ROW * scale,
                    advance * scale, scale, span.color, img_w, img_h,
                )
            x_font += advance

    return Image.fromarray(canvas, mode="RGB")


def measure_width(spans: Sequence[TextSpan], atlas: FontAtlas) -> int:
    """Return the total advance width (in font-pixels) of *spans*.

    This is the canvas width before applying scale and before shadow padding.
    Public wrapper around the internal helper, useful for callers that need to
    know dimensions before rendering.
    """
    return _measure_width(spans, atlas)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _advance(tmpl: CharTemplate, bold: bool) -> int:
    """Advance width in font-pixels for a character, accounting for bold."""
    return tmpl.advance + (1 if bold else 0)


def _select_bitmap(tmpl: CharTemplate, bold: bool, italic: bool) -> np.ndarray:
    """Pick the correct pre-computed bitmap for the requested style combo."""
    if bold and italic:
        # Bold-italic: apply italic shear to the bold bitmap.
        # _make_italic expects (8, W); bold bitmap is (8, advance), which is fine.
        return _make_italic(tmpl.bold)
    if bold:
        return tmpl.bold
    if italic:
        return tmpl.italic
    return tmpl.bitmap


def _measure_width(spans: Sequence[TextSpan], atlas: FontAtlas) -> int:
    total = 0
    for span in spans:
        for ch in span.text:
            tmpl = atlas.get(ch)
            total += _advance(tmpl, span.bold) if tmpl is not None else 4
    return total


def _blit(
    canvas: np.ndarray,
    bitmap: np.ndarray,
    x: int,
    y: int,
    scale: int,
    color: _Color,
) -> None:
    """Paint the True-pixels of *bitmap* onto *canvas* at (*x*, *y*) scaled by *scale*.

    Uses np.repeat for integer up-scaling (fast, no interpolation).
    Out-of-bounds regions are silently clipped.
    """
    H, W = canvas.shape[:2]
    # Scale up the bitmap: True cells become (scale×scale) blocks.
    scaled = np.repeat(np.repeat(bitmap, scale, axis=0), scale, axis=1)
    sh, sw = scaled.shape

    dst_y0 = max(0, y)
    dst_x0 = max(0, x)
    dst_y1 = min(H, y + sh)
    dst_x1 = min(W, x + sw)

    if dst_y0 >= dst_y1 or dst_x0 >= dst_x1:
        return

    src_y0 = dst_y0 - y
    src_x0 = dst_x0 - x
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    src_x1 = src_x0 + (dst_x1 - dst_x0)

    mask = scaled[src_y0:src_y1, src_x0:src_x1]
    canvas[dst_y0:dst_y1, dst_x0:dst_x1][mask] = color


def _draw_decoration(
    canvas: np.ndarray,
    x: int,
    y: int,
    width_px: int,
    height_px: int,
    color: _Color,
    img_w: int,
    img_h: int,
) -> None:
    """Draw a horizontal decoration line (underline or strikethrough)."""
    x0 = max(0, x)
    x1 = min(img_w, x + width_px)
    y0 = max(0, y)
    y1 = min(img_h, y + height_px)
    if x0 < x1 and y0 < y1:
        canvas[y0:y1, x0:x1] = color
