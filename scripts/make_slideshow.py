"""Build a polished 20-second demo video from the captured dashboard screenshots."""

import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
W, H = 1440, 900
FPS = 30


def font(size, bold=True):
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


TITLE_FONT = font(78, True)
SUB_FONT = font(28, False)
CAP_FONT = font(24, True)
SMALL_FONT = font(18, False)


def gradient_bg():
    y = np.linspace(0, 1, H)[:, None]
    x = np.linspace(0, 1, W)[None, :]
    base = np.zeros((H, W, 3), dtype=np.float32)
    base[..., 0] = 7 + 20 * y + 10 * x
    base[..., 1] = 11 + 18 * y + 12 * x
    base[..., 2] = 20 + 35 * (1 - y) + 25 * x
    # neon glows
    for cx, cy, radius, color in [
        (0.15, 0.1, 0.55, (34, 211, 238)),
        (0.9, 0.2, 0.6, (129, 140, 248)),
        (0.55, 1.0, 0.5, (251, 113, 133)),
    ]:
        d = np.sqrt(((x - cx) * 1.2) ** 2 + (y - cy) ** 2)
        glow = np.clip(1 - d / radius, 0, 1) ** 2 * 0.35
        for i in range(3):
            base[..., i] += glow * color[i]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def title_card():
    img = gradient_bg()
    draw = ImageDraw.Draw(img)
    # brain orb
    draw.ellipse([640, 220, 800, 380], fill=(18, 30, 55), outline=(34, 211, 238), width=3)
    draw.text((720, 300), "🧠", font=font(80, True), anchor="mm", fill=(255, 255, 255))
    draw.text((W // 2, 460), "NeuroPulse AI", font=TITLE_FONT, anchor="mm",
              fill=(170, 210, 255))
    draw.text((W // 2, 530), "Real-Time Neural Signal Classifier & Anomaly Detector",
              font=SUB_FONT, anchor="mm", fill=(180, 200, 235))
    draw.text((W // 2, 600), "6-channel synthetic EEG  ·  IsolationForest + RandomForest",
              font=SMALL_FONT, anchor="mm", fill=(130, 150, 190))
    # accent line
    draw.rounded_rectangle([560, 650, 880, 656], radius=3, fill=(34, 211, 238))
    draw.text((W // 2, 700), "Live demo", font=SMALL_FONT, anchor="mm", fill=(120, 220, 200))
    return img


def screenshot_card(path, caption):
    img = Image.open(path).convert("RGB")
    img = img.resize((W, H), Image.LANCZOS)
    # darken bottom for caption
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([0, H - 96, W, H], fill=(5, 9, 18, 210))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text((40, H - 62), caption, font=CAP_FONT, fill=(210, 230, 255))
    draw.text((W - 40, H - 58), "NeuroPulse AI", font=SMALL_FONT, anchor="ra",
              fill=(90, 200, 220))
    return img


def crossfade(a, b, steps):
    for i in range(1, steps + 1):
        alpha = i / (steps + 1)
        yield Image.blend(a, b, alpha)


def main():
    scenes = [
        (title_card(), 3.0),
        (screenshot_card(os.path.join(ASSETS, "dashboard_hero.png"),
                         "Real-time brain-state classification & anomaly risk"), 5.0),
        (screenshot_card(os.path.join(ASSETS, "dashboard_analytics.png"),
                         "Band power · scalp topography · channel heatmap"), 5.0),
        (screenshot_card(os.path.join(ASSETS, "dashboard_anomaly.png"),
                         "IsolationForest flags abnormal spikes in real time"), 5.0),
        (screenshot_card(os.path.join(ASSETS, "dashboard_events.png"),
                         "Live event timeline · alert feed · CSV session export"), 4.0),
    ]

    fade = 0.6
    frames = []
    for idx, (img, dur) in enumerate(scenes):
        frames.extend([img] * max(1, int(dur * FPS)))
        if idx < len(scenes) - 1:
            frames.extend(crossfade(img, scenes[idx + 1][0], int(fade * FPS)))

    out = os.path.join(ASSETS, "neuropulse_demo.mp4")
    writer = imageio.get_writer(
        out, fps=FPS, codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=16,
    )
    for img in frames:
        writer.append_data(np.asarray(img))
    writer.close()
    print(f"Slideshow saved: {out} ({len(frames)/FPS:.1f}s)")


if __name__ == "__main__":
    main()