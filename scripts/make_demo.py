"""Capture NeuroPulse AI screenshots and a demo video with Playwright."""

import glob
import os
import subprocess
import time

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

BASE_URL = os.environ.get("NEUROPULSE_URL", "http://127.0.0.1:8516")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
VIDEO_DIR = os.path.join(OUT, "_video")
os.makedirs(OUT, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

SCROLL_JS = "document.querySelector('main[data-testid=\"stMain\"]').scrollTo({{top:{top}, behavior:'instant'}})"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--force-device-scale-factor=1", "--disable-gpu"],
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1,
            record_video_dir=VIDEO_DIR,
            record_video_size={"width": 1440, "height": 900},
        )
        page = context.new_page()
        page.on("pageerror", lambda e: print("PAGEERROR:", str(e)[:300]))
        page.goto(BASE_URL, wait_until="domcontentloaded", timeout=90000)
        page.wait_for_selector("text=NeuroPulse AI", timeout=90000)
        # Wait for the live fragment (models train per session) to fully render.
        page.wait_for_selector("text=Multi-Channel EEG Waveform", timeout=120000)
        time.sleep(5)

        # Hero / KPI / waveform
        page.evaluate(SCROLL_JS.format(top=0))
        time.sleep(2)
        page.screenshot(path=os.path.join(OUT, "dashboard_hero.png"))

        # Analytics row
        page.evaluate(SCROLL_JS.format(top=640))
        time.sleep(3)
        page.screenshot(path=os.path.join(OUT, "dashboard_analytics.png"))

        # Event row
        page.evaluate(SCROLL_JS.format(top=1480))
        time.sleep(3)
        page.screenshot(path=os.path.join(OUT, "dashboard_events.png"))

        # Anomaly state
        page.evaluate(SCROLL_JS.format(top=0))
        time.sleep(2)
        try:
            page.get_by_text("Anomaly Injection (continuous)").click(timeout=10000)
        except Exception:
            pass
        time.sleep(6)
        page.screenshot(path=os.path.join(OUT, "dashboard_anomaly.png"))
        time.sleep(5)

        context.close()
        browser.close()

        webms = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.webm")), key=os.path.getmtime)
        if webms:
            webm_path = os.path.join(OUT, "neuropulse_demo.webm")
            os.replace(webms[-1], webm_path)
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            mp4_path = os.path.join(OUT, "neuropulse_demo.mp4")
            subprocess.run(
                [ffmpeg, "-y", "-ss", "10", "-i", webm_path,
                 "-filter:v", "setpts=PTS/1.35",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", mp4_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("Video saved:", mp4_path)

    print("Screenshots saved in:", OUT)


if __name__ == "__main__":
    main()