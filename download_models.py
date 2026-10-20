"""
Helper script to download required AI model weights for Clarity AI Studio.
Run: python download_models.py
"""

import os
import urllib.request
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = [
    {
        "name": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "desc": "Real-ESRGAN General Super-Resolution 4x Model (~67MB)"
    },
    {
        "name": "RealESRGAN_x4plus_anime_6B.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "desc": "Real-ESRGAN Anime & 2D Illustration 4x Model (~18MB)"
    },
    {
        "name": "GFPGANv1.4.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "desc": "GFPGAN v1.4 Face Restoration & Portrait Enhancer (~348MB)"
    },
    {
        "name": "big-lama.pt",
        "url": "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
        "desc": "LaMa Neural Network for Generative AI Outpainting (~205MB)"
    },
    {
        "name": "scunet_color_real_gan.pth",
        "url": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth",
        "desc": "SCUNet Swin-Conv-UNet Blind Restoration & Denoise 1X Model (~31MB)"
    }
]


def download_progress_hook(count, block_size, total_size):
    if total_size <= 0:
        return
    percent = int(count * block_size * 100 / total_size)
    percent = min(100, percent)
    downloaded_mb = (count * block_size) / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    bar = ('=' * (percent // 4)).ljust(25)
    sys.stdout.write(f"\r  [{bar}] {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
    sys.stdout.flush()


def main():
    print("=" * 65)
    print("  Clarity AI Studio - Model Weights Downloader")
    print("=" * 65)

    for item in MODELS:
        dest_path = os.path.join(MODELS_DIR, item["name"])
        print(f"\nChecking: {item['desc']}")
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000000:
            print(f"  [ALREADY EXISTS] {item['name']} ({os.path.getsize(dest_path) / 1024 / 1024:.1f} MB)")
            continue

        print(f"  Downloading from: {item['url']} ...")
        try:
            urllib.request.urlretrieve(item["url"], dest_path, reporthook=download_progress_hook)
            print(f"\n  [OK] Successfully downloaded {item['name']}!")
        except Exception as e:
            print(f"\n  [FAILED] Could not download {item['name']}: {e}")
            print(f"  Please manually download from: {item['url']}")
            print(f"  and place it in: {MODELS_DIR}")

    print("\n" + "=" * 65)
    print("  All model checks completed! You can now run: python server.py")
    print("=" * 65)


if __name__ == '__main__':
    main()
