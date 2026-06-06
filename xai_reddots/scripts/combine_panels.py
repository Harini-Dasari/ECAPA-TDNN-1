"""
combine_panels.py
=================
Combines the 4 individual panel PNGs for a speaker into a single
publication-ready 2×2 grid image with a shared title header.

Layout:
  ┌──────────────────┬──────────────────┐
  │  Panel A         │  Panel B         │
  │  RMS + Attention │  Mel Spec+Attn   │
  ├──────────────────┼──────────────────┤
  │  Panel C         │  Panel D         │
  │  Attn+Boundaries │  Phoneme Ranking │
  └──────────────────┴──────────────────┘

Usage:
    python3 xai_reddots/scripts/combine_panels.py m0004
    python3 xai_reddots/scripts/combine_panels.py m0001
    python3 xai_reddots/scripts/combine_panels.py m0002
    python3 xai_reddots/scripts/combine_panels.py all
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont

# ── CONFIG ────────────────────────────────────────────────────
PHRASE_CLEAN = "my_voice_is_my_password"
SRC_DIR      = "xai_reddots/plots/individual_panels"
OUT_DIR      = "xai_reddots/plots"
SPEAKERS     = ["m0001", "m0002", "m0004"]

PANEL_KEYS = [
    "panel_a_rms_attention",
    "panel_b_mel_spectrogram",
    "panel_c_attention_boundaries",
    "panel_d_phoneme_ranking",
]

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]

# Header bar colours (white text on dark background)
HEADER_BG  = (30, 30, 40)      # near-black
HEADER_TXT = (255, 255, 255)
DIVIDER    = (220, 220, 220)    # light gray divider between panels
BORDER     = (200, 200, 200)


def load_panels(speaker_id):
    imgs = []
    for key in PANEL_KEYS:
        path = os.path.join(SRC_DIR,
                            f"{speaker_id}_{PHRASE_CLEAN}_{key}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Panel not found: {path}\n"
                f"Run:  python3 xai_reddots/scripts/individual_panels.py {speaker_id}")
        imgs.append(Image.open(path).convert("RGB"))
    return imgs


def make_combined(speaker_id, imgs, phrase="My voice is my password"):
    """
    Stitch 4 panel images into a 2×2 grid with a header bar.
    All panels are resized to the same dimensions before stitching.
    """
    # Determine target panel size (use smallest to avoid upscaling)
    min_w = min(im.width  for im in imgs)
    min_h = min(im.height for im in imgs)

    # Resize all panels to identical size (keep aspect ratio via padding)
    panels = []
    for im in imgs:
        # Scale to fit min_w × min_h exactly (slight distortion acceptable
        # since all panels have the same matplotlib figsize)
        panels.append(im.resize((min_w, min_h), Image.LANCZOS))

    GAP        = 8    # px gap between panels
    HEADER_H   = 80   # px for title bar at top
    LABEL_H    = 32   # px for panel-letter badge
    BORDER_W   = 3    # outer border

    total_w = min_w * 2 + GAP * 3 + BORDER_W * 2
    total_h = min_h * 2 + GAP * 3 + HEADER_H + BORDER_W * 2

    canvas = Image.new("RGB", (total_w, total_h), color=(245, 245, 248))
    draw   = ImageDraw.Draw(canvas)

    # ── Header bar ──────────────────────────────────────────
    draw.rectangle([0, 0, total_w, HEADER_H], fill=HEADER_BG)
    try:
        # Try to use a nicer font if available
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_sub   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_badge = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub   = font_title
        font_badge = font_title

    title_txt = f"ECAPA-TDNN Speaker XAI Analysis  •  Speaker: {speaker_id}"
    sub_txt   = f'Phrase: "{phrase}"  |  4-Panel Individual View'

    # Centre title
    bbox = draw.textbbox((0, 0), title_txt, font=font_title)
    tw   = bbox[2] - bbox[0]
    draw.text(((total_w - tw) // 2, 10), title_txt, fill=HEADER_TXT, font=font_title)

    bbox = draw.textbbox((0, 0), sub_txt, font=font_sub)
    sw   = bbox[2] - bbox[0]
    draw.text(((total_w - sw) // 2, 46), sub_txt, fill=(180, 200, 220), font=font_sub)

    # ── Outer border ─────────────────────────────────────────
    draw.rectangle([0, 0, total_w - 1, total_h - 1],
                   outline=BORDER, width=BORDER_W)

    # ── Paste panels in 2×2 grid ─────────────────────────────
    positions = [
        (GAP + BORDER_W,                   HEADER_H + GAP),          # top-left  (A)
        (min_w + GAP * 2 + BORDER_W,       HEADER_H + GAP),          # top-right (B)
        (GAP + BORDER_W,                   HEADER_H + min_h + GAP*2),# bot-left  (C)
        (min_w + GAP * 2 + BORDER_W,       HEADER_H + min_h + GAP*2),# bot-right (D)
    ]

    for idx, (panel, (px, py), lbl) in enumerate(
            zip(panels, positions, PANEL_LABELS)):
        # Thin border around each panel
        draw.rectangle([px - 1, py - 1, px + min_w, py + min_h],
                       outline=BORDER, width=1)
        canvas.paste(panel, (px, py))

        # Panel letter badge (top-left corner overlay)
        badge_w, badge_h = 52, 32
        draw.rectangle([px, py, px + badge_w, py + badge_h],
                       fill=(30, 30, 40))
        bbox = draw.textbbox((0, 0), lbl, font=font_badge)
        bw   = bbox[2] - bbox[0]
        draw.text((px + (badge_w - bw) // 2, py + 4),
                  lbl, fill=(255, 255, 255), font=font_badge)

    # ── Divider lines ─────────────────────────────────────────
    mid_x = GAP + BORDER_W + min_w + GAP // 2
    mid_y = HEADER_H + GAP + min_h + GAP // 2
    draw.line([(mid_x, HEADER_H), (mid_x, total_h)], fill=DIVIDER, width=2)
    draw.line([(BORDER_W, mid_y), (total_w - BORDER_W, mid_y)], fill=DIVIDER, width=2)

    return canvas


def combine(speaker_id):
    print(f"Loading panels for {speaker_id}…")
    imgs   = load_panels(speaker_id)
    canvas = make_combined(speaker_id, imgs)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR,
                       f"{speaker_id}_{PHRASE_CLEAN}_combined_panels.png")
    canvas.save(out, "PNG", dpi=(300, 300))
    print(f"✓  Combined → {out}  ({canvas.width}×{canvas.height} px)")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    speakers = SPEAKERS if target == "all" else [target]

    for sid in speakers:
        try:
            combine(sid)
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")


if __name__ == "__main__":
    main()
