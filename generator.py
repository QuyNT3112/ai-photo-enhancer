"""
AI Image Generator Engine (T2I & I2I) for Clarity AI Studio
Powered by Hugging Face Diffusers (SDXL + SD 1.5) on NVIDIA CUDA (FP16)
Supports Text-to-Image, Image-to-Image, Long-Prompt Chunking (>77 tokens + Attention Weights),
FP16-safe SDXL VAE Decode, Aspect-Preserving I2I, and Clean Snapshot Caching.
"""

import os
import sys
import gc
import re
import time
import math
import json
import random
import threading
import urllib.request
import urllib.parse
from typing import Optional, Dict, Any, List, Tuple, cast

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        getattr(sys.stdout, 'reconfigure')(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        getattr(sys.stderr, 'reconfigure')(encoding='utf-8', errors='replace')
    except Exception:
        pass

import torch
from PIL import Image, ImageOps
from huggingface_hub import snapshot_download, constants as hf_constants  # type: ignore
from huggingface_hub.file_download import are_symlinks_supported  # type: ignore
from diffusers import (  # type: ignore
    AutoencoderKL,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)


def safe_print(msg: str):
    """Safely print message without throwing UnicodeEncodeError on Windows terminals."""
    try:
        print(msg, flush=True)
    except Exception:
        try:
            print(msg.encode('ascii', 'replace').decode('ascii'), flush=True)
        except Exception:
            pass


DEFAULT_MODEL_ID = "RunDiffusion/Juggernaut-XL-v9"

# Supported Models List (SOTA SDXL models first, followed by legacy SD 1.5 compatibility)
SUPPORTED_MODELS: List[Dict[str, Any]] = [
    {
        "id": "RunDiffusion/Juggernaut-XL-v9",
        "name": "🏆 Juggernaut XL v9 (SDXL · Đa Năng Đỉnh Cao · Bám Prompt Nhất)",
        "badge": "Khuyên dùng - Toàn diện",
        "type": "sdxl",
        "is_distilled": False,
        "variant": "fp16",
        "use_fp16_vae_fix": True,
        "scheduler": "dpmsolver_2m_karras",
        "scheduler_name": "DPM++ 2M Karras",
        "default_steps": 24,
        "min_steps": 12,
        "max_steps": 50,
        "default_cfg": 6.0,
        "min_cfg": 3.0,
        "max_cfg": 12.0,
        "default_strength": 0.55,
        "eta_hint": "~18-24 giây",
        "allow_patterns": ["model_index.json", "*/*.json", "*/*.txt", "*/*.fp16.safetensors"],
        "ignore_patterns": ["*.index.json"],
        "desc": "Mô hình SDXL toàn diện hàng đầu (1024px gốc): bám sát từng chi tiết Prompt, xuất sắc cho cả nhiếp ảnh thực tế, điện ảnh, phong cảnh, 3D và Img2Img."
    },
    {
        "id": "SG161222/RealVisXL_V5.0_Lightning",
        "name": "⚡ RealVisXL v5.0 Lightning (SDXL · Ảnh Thật DSLR · Siêu Tốc ~7s)",
        "badge": "Siêu Tốc 6 Steps",
        "type": "sdxl",
        "is_distilled": True,
        "variant": "fp16",
        "use_fp16_vae_fix": True,
        "scheduler": "euler_trailing",
        "scheduler_name": "Euler Trailing",
        "default_steps": 6,
        "min_steps": 4,
        "max_steps": 12,
        "default_cfg": 2.0,
        "min_cfg": 1.0,
        "max_cfg": 4.0,
        "default_strength": 0.60,
        "eta_hint": "~6-8 giây",
        "allow_patterns": ["model_index.json", "*/*.json", "*/*.txt", "*/*.fp16.safetensors"],
        "ignore_patterns": ["*.index.json"],
        "desc": "SDXL Lightning chưng cất thế hệ 5 chuyên ảnh chụp thực tế DSLR, ánh sáng tự nhiên & da người chân thực chỉ với 6 bước (~7 giây)."
    },
    {
        "id": "Lykon/dreamshaper-xl-v2-turbo",
        "name": "🎨 DreamShaper XL v2 Turbo (SDXL · 3D Pixar, Viễn Tưởng & Nghệ Thuật)",
        "badge": "Sáng tạo ~7s",
        "type": "sdxl",
        "is_distilled": True,
        "variant": "fp16",
        "use_fp16_vae_fix": True,
        "scheduler": "euler_trailing",
        "scheduler_name": "Euler Trailing",
        "default_steps": 7,
        "min_steps": 4,
        "max_steps": 15,
        "default_cfg": 2.0,
        "min_cfg": 1.0,
        "max_cfg": 4.5,
        "default_strength": 0.60,
        "eta_hint": "~6-8 giây",
        "allow_patterns": ["model_index.json", "*/*.json", "*/*.txt", "*/*.fp16.safetensors"],
        "ignore_patterns": ["*.index.json"],
        "desc": "SDXL Turbo đa phong cách siêu nhanh: nổi bật cho hoạt hình 3D Pixar, cyberpunk, giả tưởng fantasy, concept art và poster."
    },
    {
        "id": "cagliostrolab/animagine-xl-4.0",
        "name": "🌸 Animagine XL 4.0 (SDXL · Chuyên Anime & Illustration Đỉnh Cao)",
        "badge": "Anime SOTA",
        "type": "sdxl",
        "is_distilled": False,
        "variant": None,
        "use_fp16_vae_fix": False,
        "scheduler": "euler_a",
        "scheduler_name": "Euler a",
        "default_steps": 24,
        "min_steps": 15,
        "max_steps": 45,
        "default_cfg": 5.0,
        "min_cfg": 3.0,
        "max_cfg": 9.0,
        "default_strength": 0.55,
        "eta_hint": "~18-24 giây",
        "allow_patterns": ["model_index.json", "*/*.json", "*/*.txt", "*/*.safetensors"],
        "ignore_patterns": ["*.index.json"],
        "desc": "Mô hình Anime SDXL thế hệ 4.0 mới nhất: giải phẫu nhân vật chuẩn xác, nét vẽ sắc sảo, màu sắc rực rỡ và hiểu sâu các mô tả 2D."
    },
    {
        "id": "stabilityai/stable-diffusion-xl-base-1.0",
        "name": "🏛️ Stable Diffusion XL 1.0 Base (SDXL · Chuẩn Gốc Stability AI)",
        "badge": "SDXL Gốc",
        "type": "sdxl",
        "is_distilled": False,
        "variant": "fp16",
        "use_fp16_vae_fix": True,
        "scheduler": "dpmsolver_2m_karras",
        "scheduler_name": "DPM++ 2M Karras",
        "default_steps": 25,
        "min_steps": 15,
        "max_steps": 50,
        "default_cfg": 7.0,
        "min_cfg": 4.0,
        "max_cfg": 12.0,
        "default_strength": 0.55,
        "eta_hint": "~18-25 giây",
        "allow_patterns": ["model_index.json", "*/*.json", "*/*.txt", "*/*.fp16.safetensors"],
        "ignore_patterns": ["*.index.json"],
        "desc": "Mô hình nền tảng SDXL 1.0 nguyên bản từ Stability AI, độ cân bằng trung tính cho nhiều thể loại."
    },
    {
        "id": "SG161222/Realistic_Vision_V5.1_noVAE",
        "name": "📷 Realistic Vision v5.1 (SD 1.5 · Cũ/Nhẹ ~3s)",
        "badge": "SD 1.5 Cũ",
        "type": "sd15",
        "is_distilled": False,
        "variant": None,
        "use_fp16_vae_fix": False,
        "scheduler": "dpmsolver_2m_karras",
        "scheduler_name": "DPM++ 2M Karras",
        "default_steps": 25,
        "min_steps": 12,
        "max_steps": 50,
        "default_cfg": 7.0,
        "min_cfg": 3.0,
        "max_cfg": 12.0,
        "default_strength": 0.50,
        "eta_hint": "~3-5 giây",
        "desc": "Mô hình SD 1.5 cũ (512px) chuyên chân dung nhẹ, ít tốn RAM hơn SDXL nhưng độ chi tiết và bám prompt kém hơn."
    },
    {
        "id": "Lykon/dreamshaper-8",
        "name": "🖌️ DreamShaper v8 (SD 1.5 · Cũ/Nhẹ ~3s)",
        "badge": "SD 1.5 Cũ",
        "type": "sd15",
        "is_distilled": False,
        "variant": None,
        "use_fp16_vae_fix": False,
        "scheduler": "dpmsolver_2m_karras",
        "scheduler_name": "DPM++ 2M Karras",
        "default_steps": 25,
        "min_steps": 12,
        "max_steps": 50,
        "default_cfg": 7.0,
        "min_cfg": 3.0,
        "max_cfg": 12.0,
        "default_strength": 0.50,
        "eta_hint": "~3-5 giây",
        "desc": "Mô hình SD 1.5 cổ điển cho tranh minh họa nhẹ (512px)."
    },
    {
        "id": "runwayml/stable-diffusion-v1-5",
        "name": "⚙️ Stable Diffusion v1.5 (SD 1.5 · Cơ Bản ~2.5s)",
        "badge": "SD 1.5 Cơ Bản",
        "type": "sd15",
        "is_distilled": False,
        "variant": None,
        "use_fp16_vae_fix": False,
        "scheduler": "dpmsolver_2m_karras",
        "scheduler_name": "DPM++ 2M Karras",
        "default_steps": 25,
        "min_steps": 12,
        "max_steps": 50,
        "default_cfg": 7.5,
        "min_cfg": 3.0,
        "max_cfg": 12.0,
        "default_strength": 0.50,
        "eta_hint": "~2.5-4 giây",
        "desc": "Mô hình SD 1.5 nền tảng cũ (512px), giữ lại để tương thích các preset cũ."
    },
]

MODEL_CONFIG_MAP: Dict[str, Dict[str, Any]] = {m["id"]: m for m in SUPPORTED_MODELS}


def get_model_config(model_id: Optional[str]) -> Dict[str, Any]:
    """Looks up model metadata safely, inferring SDXL/SD1.5 defaults if custom/unknown."""
    mid = (model_id or DEFAULT_MODEL_ID).strip()
    if mid in MODEL_CONFIG_MAP:
        return MODEL_CONFIG_MAP[mid]
    low = mid.lower()
    is_sdxl = ("xl" in low or "juggernaut" in low or "realvis" in low or "animagine" in low)
    return {
        "id": mid,
        "name": mid,
        "badge": "Custom",
        "type": "sdxl" if is_sdxl else "sd15",
        "is_distilled": ("lightning" in low or "turbo" in low),
        "variant": "fp16" if is_sdxl else None,
        "use_fp16_vae_fix": is_sdxl and ("animagine" not in low),
        "scheduler": "euler_trailing" if ("lightning" in low or "turbo" in low) else "dpmsolver_2m_karras",
        "scheduler_name": "Euler Trailing" if ("lightning" in low or "turbo" in low) else "DPM++ 2M Karras",
        "default_steps": 7 if ("lightning" in low or "turbo" in low) else 24,
        "min_steps": 4 if ("lightning" in low or "turbo" in low) else 12,
        "max_steps": 50,
        "default_cfg": 2.0 if ("lightning" in low or "turbo" in low) else 6.0,
        "min_cfg": 1.0,
        "max_cfg": 12.0,
        "default_strength": 0.55,
        "eta_hint": "~7-22 giây",
    }


# Smart Bilingual Translation & Semantic Understanding
_VI_DIACRITICS = set(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
)

_UNACCENTED_VI_PHRASES = [
    "co gai", "chang trai", "ao dai", "ho guom", "pho co", "hoi an",
    "con meo", "con cho", "con rong", "sieu xe", "thanh pho", "bau troi",
    "hoang hon", "binh minh", "buoi sang", "buoi toi", "troi mua", "khu rung",
    "mot nguoi", "cua so", "chan dung", "phong canh", "phi hanh gia",
    "mac ao", "ngoi doc", "anh sang", "mat troi", "ngoai pho", "giua rung",
    "de thuong", "sieu net", "hien dai", "co tich", "bac si", "benh vien",
    "canh sat", "phi cong", "vu tru", "dai duong", "ngon nui", "khung canh"
]


def _needs_vietnamese_translation(text: str) -> bool:
    if any(c in _VI_DIACRITICS for c in text):
        return True
    low = f" {text.lower()} "
    hits = 0
    for phrase in _UNACCENTED_VI_PHRASES:
        if f" {phrase} " in low or phrase in low:
            hits += 1
            if hits >= 2 or phrase in {"co gai", "chang trai", "ao dai", "ho guom", "chan dung", "phong canh", "phi hanh gia", "de thuong", "benh vien"}:
                return True
    return False


def smart_translate_prompt(text: str) -> str:
    """Translates Vietnamese prompt to English accurately with multi-tier fallback."""
    text_clean = text.strip()
    if not text_clean:
        return ""
    if not _needs_vietnamese_translation(text_clean):
        return text_clean

    # Engine 1: Chrome Extension Google Translate endpoint (fast, preserves punctuation & multi-clause prompts)
    try:
        url = (
            "https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl=vi&tl=en&q="
            + urllib.parse.quote(text_clean)
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], list) and len(data[0]) > 0:
                    t = data[0][0]
                    if isinstance(t, str) and t.strip():
                        return t.strip()
                elif isinstance(data[0], str) and data[0].strip():
                    return data[0].strip()
    except Exception as e:
        safe_print(f"[ImageGenerator] Translation Engine 1 fallback: {e}")

    # Engine 2: MyMemory API fallback
    try:
        url = "https://api.mymemory.translated.net/get?q=" + urllib.parse.quote(text_clean) + "&langpair=vi|en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data and "responseData" in data and "translatedText" in data["responseData"]:
                t = data["responseData"]["translatedText"]
                if t and t.strip() and not t.startswith("MYMEMORY WARNING"):
                    return t.strip()
    except Exception as e:
        safe_print(f"[ImageGenerator] Translation Engine 2 fallback: {e}")

    # Engine 3: Local semantic concept replacement
    try:
        from llm_engine import VIETNAMESE_CONCEPT_MAP
        t_low = text_clean.lower()
        replacements = []
        for pat, en in VIETNAMESE_CONCEPT_MAP.items():
            if re.search(pat, t_low):
                replacements.append(en)
        if replacements:
            return ", ".join(replacements)
    except Exception:
        pass

    return text_clean


def detect_style_from_prompt(prompt_en: str) -> str:
    """Detects artistic style from English prompt keywords without overriding explicit intent."""
    p = prompt_en.lower()
    if any(re.search(r'\b' + re.escape(k) + r'\b', p) for k in ["anime", "manga", "waifu", "shinkai", "chibi", "cel shaded", "pixiv"]):
        return "anime"
    if any(re.search(r'\b' + re.escape(k) + r'\b', p) for k in ["pixar", "disney", "3d render", "3d character", "octane render", "unreal engine", "cgi"]):
        return "pixar3d"
    if any(re.search(r'\b' + re.escape(k) + r'\b', p) for k in ["cyberpunk", "neon", "sci-fi", "futuristic", "mecha", "blade runner"]):
        return "cyberpunk"
    if any(re.search(r'\b' + re.escape(k) + r'\b', p) for k in ["photo", "portrait", "dslr", "photograph", "real life", "skin texture", "raw photo", "headshot", "selfie", "candid"]):
        return "photorealistic"
    if any(re.search(r'\b' + re.escape(k) + r'\b', p) for k in ["fantasy", "dragon", "magic", "mythical", "wizard", "elf", "sorcerer"]):
        return "fantasy"
    if any(re.search(r'\b' + re.escape(k) + r'\b', p) for k in ["vintage", "retro", "1970s", "1980s", "film grain", "analog", "kodak"]):
        return "vintage"
    if any(re.search(r'\b' + re.escape(k) + r'\b', p) for k in ["cinematic", "film still", "movie scene", "dramatic lighting", "anamorphic"]):
        return "cinematic"
    return "general"


# Clean Style Presets (Subject-First, Minimal Quality Bias so Prompt Intent Leads)
STYLE_PRESETS = {
    "auto": {
        "name": "✨ Tự động bám sát Prompt & Phong cách ảnh gốc (Auto)",
        "prompt_prefix": "",
        "prompt_suffix": ", highly detailed, sharp focus, high quality",
        "negative": "worst quality, low quality, blurry, lowres, bad anatomy, deformed, watermark"
    },
    "none": {
        "name": "🎯 Nguyên bản theo Prompt (Không thêm từ khóa Style)",
        "prompt_prefix": "",
        "prompt_suffix": "",
        "negative": "blurry, low quality, watermark"
    },
    "original": {
        "name": "🖼️ Giữ nguyên phong cách & bố cục ảnh gốc (I2I Preserve)",
        "prompt_prefix": "",
        "prompt_suffix": ", high quality, detailed, sharp focus",
        "negative": "worst quality, low quality, blurry, noise, artifacts, distorted features, watermark"
    },
    "anime": {
        "name": "🌸 Anime Nhật Bản (Vibrant Illustration)",
        "prompt_prefix": "",
        "prompt_suffix": ", anime style illustration, vibrant colors, crisp clean lineart, detailed anime artwork, best quality",
        "negative": "photorealistic, real life photo, photography, western cartoon, 3d render, low quality, bad anatomy, deformed"
    },
    "photorealistic": {
        "name": "📸 Nhiếp Ảnh Đời Thực (Photorealistic DSLR)",
        "prompt_prefix": "",
        "prompt_suffix": ", realistic photography, shot on DSLR, natural lighting, sharp focus, intricate real-world details",
        "negative": "painting, drawing, illustration, anime, cartoon, 3d render, cgi, plastic skin, deformed, blurry"
    },
    "cinematic": {
        "name": "🎬 Điện Ảnh 8K (Cinematic Film Still)",
        "prompt_prefix": "",
        "prompt_suffix": ", cinematic film still, dramatic atmospheric lighting, volumetric light, rich contrast, sharp focus, 8k uhd",
        "negative": "cartoon, anime, blurry, low resolution, amateur, flat lighting, oversaturated"
    },
    "pixar3d": {
        "name": "🧸 3D Hoạt Hình Pixar / Disney",
        "prompt_prefix": "",
        "prompt_suffix": ", stylized 3D character animation render, Pixar Disney style, expressive eyes, soft studio rim lighting, octane render",
        "negative": "realistic human photo, creepy, ugly, deformed, grainy, 2d sketch"
    },
    "cyberpunk": {
        "name": "🌃 Cyberpunk Neon (Khoa Học Viễn Tưởng)",
        "prompt_prefix": "",
        "prompt_suffix": ", cyberpunk sci-fi aesthetic, vibrant glowing neon lights, futuristic reflections, volumetric mist, dramatic night lighting",
        "negative": "vintage, rustic, daylight, blurry, lowres, poor composition"
    },
    "fantasy": {
        "name": "🏰 Thần Thoại & Giả Tưởng (Epic Fantasy)",
        "prompt_prefix": "",
        "prompt_suffix": ", epic fantasy concept art, mystical glowing ambiance, intricate ethereal details, dramatic composition, masterpiece",
        "negative": "modern, cars, blurry, poor quality, bad proportions, watermark"
    },
    "vintage": {
        "name": "🎞️ Phim Cổ Điển Retro (Vintage 1970s)",
        "prompt_prefix": "",
        "prompt_suffix": ", vintage 1970s analog Kodak film photograph, warm retro color tones, subtle film grain, classic lens look",
        "negative": "modern digital, cgi, 3d render, futuristic, oversaturated"
    }
}

DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, low quality, lowres, blurry, bad anatomy, bad hands, "
    "missing fingers, extra digits, deformed, disfigured, poorly drawn face, "
    "mutation, extra limbs, watermark, signature, out of focus"
)

PROMPT_SUGGESTIONS = [
    {
        "prompt": "Phi hành gia mèo mặc bộ giáp vũ trụ phát sáng neon xanh tím, lơ lửng giữa tinh vân dải ngân hà rực rỡ, chi tiết phản chiếu trên kính mũ",
        "style": "cyberpunk",
        "ratio": "1:1"
    },
    {
        "prompt": "Cô gái anime ngồi đọc sách bên cửa sổ quán cà phê ngắm mưa rơi ngoài phố Tokyo về đêm, ánh đèn vàng ấm áp phản chiếu trên mặt kính",
        "style": "anime",
        "ratio": "9:16"
    },
    {
        "prompt": "Siêu xe thể thao tương lai màu đen nhám chạy trên đường đèo ven biển lúc hoàng hôn đỏ rực, mặt đường ướt phản chiếu ánh mặt trời",
        "style": "cinematic",
        "ratio": "16:9"
    },
    {
        "prompt": "Chú rồng con dễ thương vảy xanh ngọc bích ôm viên pha lê phát sáng trong hang động thạch nhũ kỳ ảo, hoạt hình 3D",
        "style": "pixar3d",
        "ratio": "1:1"
    },
    {
        "prompt": "Chân dung cô gái Việt Nam mặc áo dài trắng truyền thống đứng bên hồ gươm buổi sớm có sương mù nhẹ, ánh nắng dịu xuyên qua tán cây",
        "style": "photorealistic",
        "ratio": "9:16"
    },
    {
        "prompt": "Lâu đài cổ tích bay lơ lửng giữa những đám mây rực rỡ lúc bình minh với những dòng thác đổ xuống từ rìa đá",
        "style": "fantasy",
        "ratio": "16:9"
    }
]


# ============================================================================
# Attention Weighting & Long-Prompt (>77 tokens) CLIP Encoder Utilities
# ============================================================================

_RE_ATTENTION = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def _parse_prompt_attention(text: str) -> List[Tuple[str, float]]:
    """
    Parses A1111-style attention weighting syntax such as `(keyword:1.25)`, `(boost)`, `[weaken]`.
    Returns a list of (substring, weight) tuples while gracefully handling unbalanced brackets.
    """
    if not text:
        return [("", 1.0)]

    res: List[Tuple[str, float]] = []
    round_brackets: List[int] = []
    square_brackets: List[int] = []
    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1.0 / 1.1

    def multiply_range(start_position: int, multiplier: float):
        for p in range(start_position, len(res)):
            t, w = res[p]
            res[p] = (t, w * multiplier)

    for m in _RE_ATTENTION.finditer(text):
        tok = m.group(0)
        weight_group = m.group(1)

        if tok.startswith("\\"):
            res.append((tok[1:], 1.0))
        elif tok == "(":
            round_brackets.append(len(res))
        elif tok == "[":
            square_brackets.append(len(res))
        elif weight_group is not None and round_brackets:
            try:
                w = max(0.1, min(2.5, float(weight_group)))
            except ValueError:
                w = round_bracket_multiplier
            multiply_range(round_brackets.pop(), w)
        elif tok == ")" and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif tok == "]" and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            # Handle stray ':' or unmatched bracket as plain text
            parts = re.split(r"\\(.)", tok)
            res.append(("".join(parts), 1.0))

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)
    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if not res:
        return [("", 1.0)]

    # Coalesce adjacent spans with identical weights
    merged: List[Tuple[str, float]] = []
    for t, w in res:
        if not t:
            continue
        if merged and abs(merged[-1][1] - w) < 1e-6:
            merged[-1] = (merged[-1][0] + t, w)
        else:
            merged.append((t, w))

    return merged if merged else [("", 1.0)]


def _tokenize_to_chunks(tokenizer: Any, text: str, max_chunks: int = 6) -> List[Tuple[List[int], List[float]]]:
    """
    Tokenizes arbitrarily long weighted text into 77-token CLIP chunks ([BOS] + <=75 tokens + [EOS] + [PAD...]).
    Supports both CLIP-L (pad==eos==49407) and OpenCLIP-G (pad==0, eos==49407).
    """
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 49406
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 49407
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    max_len = getattr(tokenizer, "model_max_length", 77) or 77
    chunk_body_cap = max_len - 2  # 75 tokens per chunk

    parsed = _parse_prompt_attention(text)
    all_tokens: List[int] = []
    all_weights: List[float] = []

    for subtext, weight in parsed:
        if not subtext:
            continue
        orig_max_len = getattr(tokenizer, "model_max_length", max_len)
        try:
            tokenizer.model_max_length = 100000
            tok_ids = tokenizer.encode(subtext, add_special_tokens=False)
        finally:
            tokenizer.model_max_length = orig_max_len
        for tid in tok_ids:
            all_tokens.append(int(tid))
            all_weights.append(float(weight))

    chunks: List[Tuple[List[int], List[float]]] = []
    if not all_tokens:
        ids = [bos, eos] + [pad] * chunk_body_cap
        wts = [1.0] * max_len
        chunks.append((ids, wts))
        return chunks

    for idx in range(0, len(all_tokens), chunk_body_cap):
        if len(chunks) >= max_chunks:
            break
        slice_ids = all_tokens[idx : idx + chunk_body_cap]
        slice_wts = all_weights[idx : idx + chunk_body_cap]
        rem = chunk_body_cap - len(slice_ids)
        chunk_ids = [bos] + slice_ids + [eos] + [pad] * rem
        chunk_wts = [1.0] + slice_wts + [1.0] + [1.0] * rem
        chunks.append((chunk_ids, chunk_wts))

    return chunks


def _pad_chunk_list(tokenizer: Any, chunks: List[Tuple[List[int], List[float]]], target_count: int) -> List[Tuple[List[int], List[float]]]:
    """Pads a list of chunks with empty [BOS, EOS, PAD...] chunks up to target_count."""
    if len(chunks) >= target_count:
        return chunks[:target_count]
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 49406
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 49407
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    max_len = getattr(tokenizer, "model_max_length", 77) or 77
    empty_ids = [bos, eos] + [pad] * (max_len - 2)
    empty_wts = [1.0] * max_len
    return list(chunks) + [(empty_ids, empty_wts)] * (target_count - len(chunks))


def _apply_chunk_weights(embeds: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Applies per-token weights to CLIP hidden states while preserving chunk mean norm
    so emphasized prompts stay sharp without oversaturating or burning.
    embeds: (N, 77, D), weights: (N, 77, 1)
    """
    if torch.all(weights == 1.0):
        return embeds
    orig_mean = embeds.mean(dim=(-2, -1), keepdim=True)
    weighted = embeds * weights
    new_mean = weighted.mean(dim=(-2, -1), keepdim=True)
    scale = torch.where(new_mean.abs() > 1e-6, orig_mean / new_mean, torch.ones_like(orig_mean))
    return weighted * scale


# ============================================================================
# Main Generator Class
# ============================================================================

class ImageGenerator:
    """
    Manages Diffusion models (SDXL + SD 1.5) for Text-to-Image (T2I) and Image-to-Image (I2I).
    Uses shared weights between T2I and I2I pipelines and optimal CPU offload on RTX 3060 12GB WDDM.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.current_model_id: Optional[str] = None
        self.current_model_type: str = "sdxl"
        self.t2i_pipe: Any = None
        self.i2i_pipe: Any = None
        self._active_offload_mode: Optional[str] = None
        self._lock = threading.Lock()
        safe_print(f"[ImageGenerator] Initialized on device: {self.device.upper()} (FP16: {self.torch_dtype == torch.float16})")

    def _resolve_local_snapshot(self, repo_id: str) -> Optional[str]:
        """Checks if a clean Diffusers snapshot exists locally in Hugging Face Hub cache."""
        cache_dir = hf_constants.HF_HUB_CACHE
        storage = os.path.join(cache_dir, f"models--{repo_id.replace('/', '--')}")
        if not os.path.isdir(storage):
            return None

        commit_hash = None
        ref_main = os.path.join(storage, "refs", "main")
        if os.path.isfile(ref_main):
            try:
                with open(ref_main, "r", encoding="utf-8") as f:
                    commit_hash = f.read().strip()
            except Exception:
                commit_hash = None

        snaps_dir = os.path.join(storage, "snapshots")
        if not os.path.isdir(snaps_dir):
            return None

        candidates = []
        if commit_hash and os.path.isdir(os.path.join(snaps_dir, commit_hash)):
            candidates.append(os.path.join(snaps_dir, commit_hash))
        for entry in sorted(os.listdir(snaps_dir), reverse=True):
            p = os.path.join(snaps_dir, entry)
            if os.path.isdir(p) and p not in candidates:
                candidates.append(p)

        for cand in candidates:
            if not os.path.isfile(os.path.join(cand, "model_index.json")):
                continue
            unet_dir = os.path.join(cand, "unet")
            if os.path.isdir(unet_dir):
                files = os.listdir(unet_dir)
                if any(fn.endswith((".safetensors", ".bin")) and not fn.endswith(".index.json") for fn in files):
                    return cand
        return None

    def _download_clean_snapshot(self, model_cfg: Dict[str, Any]) -> str:
        """Downloads only required FP16/Diffusers files without sharded FP32 UNets or root checkpoints."""
        repo_id = model_cfg["id"]
        storage = os.path.join(hf_constants.HF_HUB_CACHE, f"models--{repo_id.replace('/', '--')}")
        os.makedirs(storage, exist_ok=True)
        # Fix Windows HuggingFace Hub symlink concurrency bug before worker threads start
        try:
            are_symlinks_supported(storage)
        except Exception:
            pass

        kwargs: Dict[str, Any] = {"max_workers": 4}
        if model_cfg.get("allow_patterns"):
            kwargs["allow_patterns"] = model_cfg["allow_patterns"]
        if model_cfg.get("ignore_patterns"):
            kwargs["ignore_patterns"] = model_cfg["ignore_patterns"]

        safe_print(f"[ImageGenerator] Downloading clean FP16 snapshot for {repo_id}...")
        return snapshot_download(repo_id, **kwargs)

    def _load_sdxl_fp16_vae(self) -> Optional[AutoencoderKL]:
        """Loads madebyollin/sdxl-vae-fp16-fix so FP16 VAE decode never produces NaNs or slows down with FP32 upcast."""
        vae_repo = "madebyollin/sdxl-vae-fp16-fix"
        try:
            raw_vae = AutoencoderKL.from_pretrained(vae_repo, torch_dtype=self.torch_dtype, local_files_only=True)
            return cast(AutoencoderKL, raw_vae[0] if isinstance(raw_vae, tuple) else raw_vae)
        except Exception:
            try:
                storage = os.path.join(hf_constants.HF_HUB_CACHE, f"models--{vae_repo.replace('/', '--')}")
                os.makedirs(storage, exist_ok=True)
                try:
                    are_symlinks_supported(storage)
                except Exception:
                    pass
                raw_vae = AutoencoderKL.from_pretrained(vae_repo, torch_dtype=self.torch_dtype)
                return cast(AutoencoderKL, raw_vae[0] if isinstance(raw_vae, tuple) else raw_vae)
            except Exception as e:
                safe_print(f"[ImageGenerator] Warning: Could not load sdxl-vae-fp16-fix ({e}), using model default VAE.")
                return None

    def _configure_scheduler(self, pipe: Any, scheduler_key: str, model_id: str):
        """Configures optimal scheduler per model architecture without clashing config keys."""
        try:
            sched_cfg = dict(pipe.scheduler.config)
            if scheduler_key == "euler_a":
                sched_cfg.pop("final_sigmas_type", None)
                sched_cfg.pop("algorithm_type", None)
                sched_cfg.pop("use_karras_sigmas", None)
                pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(sched_cfg)
                safe_print(f"[ImageGenerator] Configured EulerAncestralDiscreteScheduler (Euler a) for {model_id}")
            elif scheduler_key == "dpmsolver_sde_karras":
                if sched_cfg.get("algorithm_type") == "deis" and sched_cfg.get("final_sigmas_type", "zero") == "zero":
                    sched_cfg["final_sigmas_type"] = "sigma_min"
                sched_cfg.pop("final_sigmas_type", None)
                sched_cfg["algorithm_type"] = "sde-dpmsolver++"
                sched_cfg["use_karras_sigmas"] = True
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(sched_cfg)
                safe_print(f"[ImageGenerator] Configured DPMSolverMultistepScheduler (DPM++ SDE Karras) for {model_id}")
            elif scheduler_key == "euler_trailing":
                sched_cfg["timestep_spacing"] = "trailing"
                sched_cfg["steps_offset"] = 0
                sched_cfg.pop("algorithm_type", None)
                sched_cfg.pop("use_karras_sigmas", None)
                sched_cfg.pop("final_sigmas_type", None)
                pipe.scheduler = EulerDiscreteScheduler.from_config(sched_cfg)
                safe_print(f"[ImageGenerator] Configured EulerDiscreteScheduler (Euler Trailing) for {model_id}")
            elif scheduler_key == "euler":
                pipe.scheduler = EulerDiscreteScheduler.from_config(sched_cfg)
                safe_print(f"[ImageGenerator] Configured EulerDiscreteScheduler for {model_id}")
            else:
                if sched_cfg.get("algorithm_type") == "deis" and sched_cfg.get("final_sigmas_type", "zero") == "zero":
                    sched_cfg["final_sigmas_type"] = "sigma_min"
                sched_cfg["algorithm_type"] = "dpmsolver++"
                sched_cfg["use_karras_sigmas"] = True
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(sched_cfg)
                safe_print(f"[ImageGenerator] Configured DPMSolverMultistepScheduler (DPM++ 2M Karras) for {model_id}")
        except Exception as e:
            safe_print(f"[ImageGenerator] Warning: Could not set scheduler '{scheduler_key}' ({e}), keeping default.")

    def _ensure_offload_mode(self, mode: str):
        """Ensures accelerate CPU offload hooks match the currently executing SDXL pipeline (T2I or I2I)."""
        if self.device != "cuda" or self.current_model_type != "sdxl":
            return
        if self._active_offload_mode == mode:
            return
        target_pipe = self.i2i_pipe if mode == "i2i" else self.t2i_pipe
        if target_pipe is not None:
            target_pipe.enable_model_cpu_offload(device=self.device)
            self._active_offload_mode = mode

    def _ensure_pipeline(self, model_id: Optional[str]):
        """Loads or switches T2I + shared I2I pipelines cleanly with optimal memory layout."""
        mid = (model_id or DEFAULT_MODEL_ID).strip()
        if self.current_model_id == mid and self.t2i_pipe is not None and self.i2i_pipe is not None:
            return

        model_cfg = get_model_config(mid)
        model_type = model_cfg["type"]

        safe_print(f"[ImageGenerator] Loading {model_type.upper()} model: {mid} ...")
        t0 = time.time()

        # Free previous pipelines completely before loading a ~7GB SDXL checkpoint
        if self.t2i_pipe is not None or self.i2i_pipe is not None:
            try:
                if self.t2i_pipe is not None and hasattr(self.t2i_pipe, "remove_all_hooks"):
                    self.t2i_pipe.remove_all_hooks()
                if self.i2i_pipe is not None and hasattr(self.i2i_pipe, "remove_all_hooks"):
                    self.i2i_pipe.remove_all_hooks()
            except Exception:
                pass
            self.t2i_pipe = None
            self.i2i_pipe = None
            self._active_offload_mode = None
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

        snap_dir = self._resolve_local_snapshot(mid)
        load_source = snap_dir if snap_dir else mid
        variant = model_cfg.get("variant")

        if model_type == "sdxl":
            if not snap_dir and model_cfg.get("allow_patterns"):
                try:
                    load_source = self._download_clean_snapshot(model_cfg)
                except Exception as e:
                    safe_print(f"[ImageGenerator] Snapshot download notice ({e}), attempting direct from_pretrained...")

            load_kwargs: Dict[str, Any] = {
                "torch_dtype": self.torch_dtype,
                "use_safetensors": True,
                "local_files_only": bool(snap_dir or os.path.isdir(load_source)),
            }
            if variant:
                load_kwargs["variant"] = variant

            vae = self._load_sdxl_fp16_vae() if model_cfg.get("use_fp16_vae_fix") else None
            if vae is not None:
                load_kwargs["vae"] = vae

            try:
                raw_pipe = StableDiffusionXLPipeline.from_pretrained(load_source, **load_kwargs)
            except Exception:
                load_kwargs["local_files_only"] = False
                raw_pipe = StableDiffusionXLPipeline.from_pretrained(mid, **load_kwargs)

            pipe: Any = raw_pipe[0] if isinstance(raw_pipe, tuple) else raw_pipe
            self._configure_scheduler(pipe, model_cfg.get("scheduler", "dpmsolver_2m_karras"), mid)

            # Build shared I2I pipeline with zero duplicate weights
            i2i_pipe: Any = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)

            # Enable VAE slicing & tiling (keeps VAE memory < 500MB without slowing down SDPA attention!)
            try:
                pipe.enable_vae_slicing()
                pipe.enable_vae_tiling()
            except Exception:
                pass

            if self.device == "cuda":
                # IMPORTANT: use model CPU offload + native PyTorch 2.5 SDPA (NO enable_attention_slicing!)
                # Keeps peak VRAM ~5.6 GB (fits comfortably inside RTX 3060 WDDM headroom) at ~1.0s/step.
                pipe.enable_model_cpu_offload(device=self.device)
                self._active_offload_mode = "t2i"
            else:
                pipe.to("cpu")
                self._active_offload_mode = None

        else:
            # Legacy SD 1.5 path
            try:
                raw_pipe = StableDiffusionPipeline.from_pretrained(
                    load_source,
                    torch_dtype=self.torch_dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=bool(snap_dir)
                )
            except Exception:
                raw_pipe = StableDiffusionPipeline.from_pretrained(
                    mid,
                    torch_dtype=self.torch_dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                )

            pipe: Any = raw_pipe[0] if isinstance(raw_pipe, tuple) else raw_pipe
            self._configure_scheduler(pipe, model_cfg.get("scheduler", "dpmsolver_2m_karras"), mid)
            pipe.to(self.device)
            if self.device == "cuda":
                try:
                    pipe.enable_vae_slicing()
                except Exception:
                    pass

            i2i_pipe: Any = StableDiffusionImg2ImgPipeline.from_pipe(pipe)
            i2i_pipe.to(self.device)
            self._active_offload_mode = None

        self.t2i_pipe = pipe
        self.i2i_pipe = i2i_pipe
        self.current_model_id = mid
        self.current_model_type = model_type
        safe_print(f"[ImageGenerator] {model_type.upper()} pipeline {mid} ready in {time.time() - t0:.2f}s!")

    def _encode_sdxl_prompt(
        self,
        pipe: Any,
        positive_prompt: str,
        negative_prompt: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes unlimited-length weighted prompts for SDXL across both CLIP-L and OpenCLIP-G.
        Matches positive and negative chunk counts, preserves OpenCLIP-G pad=0 token handling,
        and releases text encoders from GPU right after encoding.
        """
        exec_device = torch.device(self.device if self.device == "cuda" else "cpu")
        dtype = pipe.unet.dtype if pipe.unet is not None else self.torch_dtype

        pos_c1 = _tokenize_to_chunks(pipe.tokenizer, positive_prompt)
        neg_c1 = _tokenize_to_chunks(pipe.tokenizer, negative_prompt)
        pos_c2 = _tokenize_to_chunks(pipe.tokenizer_2, positive_prompt)
        neg_c2 = _tokenize_to_chunks(pipe.tokenizer_2, negative_prompt)

        num_chunks = max(len(pos_c1), len(neg_c1), len(pos_c2), len(neg_c2), 1)
        pos_c1 = _pad_chunk_list(pipe.tokenizer, pos_c1, num_chunks)
        neg_c1 = _pad_chunk_list(pipe.tokenizer, neg_c1, num_chunks)
        pos_c2 = _pad_chunk_list(pipe.tokenizer_2, pos_c2, num_chunks)
        neg_c2 = _pad_chunk_list(pipe.tokenizer_2, neg_c2, num_chunks)

        # Encode with Text Encoder 1 (CLIP-L -> penultimate hidden_states[-2], dim=768)
        ids_1 = torch.tensor([c[0] for c in (pos_c1 + neg_c1)], dtype=torch.long, device=exec_device)
        wts_1 = torch.tensor([c[1] for c in (pos_c1 + neg_c1)], dtype=dtype, device=exec_device).unsqueeze(-1)
        with torch.inference_mode():
            out_1 = pipe.text_encoder(ids_1, output_hidden_states=True)
            hid_1 = out_1.hidden_states[-2].to(dtype=dtype)
        hid_1_pos = _apply_chunk_weights(hid_1[:num_chunks], wts_1[:num_chunks]).reshape(1, num_chunks * hid_1.shape[1], -1)
        hid_1_neg = _apply_chunk_weights(hid_1[num_chunks:], wts_1[num_chunks:]).reshape(1, num_chunks * hid_1.shape[1], -1)

        # Encode with Text Encoder 2 (OpenCLIP-G -> hidden_states[-2] dim=1280 + pooled dim=1280 from chunk 0)
        ids_2 = torch.tensor([c[0] for c in (pos_c2 + neg_c2)], dtype=torch.long, device=exec_device)
        wts_2 = torch.tensor([c[1] for c in (pos_c2 + neg_c2)], dtype=dtype, device=exec_device).unsqueeze(-1)
        with torch.inference_mode():
            out_2 = pipe.text_encoder_2(ids_2, output_hidden_states=True)
            hid_2 = out_2.hidden_states[-2].to(dtype=dtype)
            pooled_all = out_2[0].to(dtype=dtype)

        hid_2_pos = _apply_chunk_weights(hid_2[:num_chunks], wts_2[:num_chunks]).reshape(1, num_chunks * hid_2.shape[1], -1)
        hid_2_neg = _apply_chunk_weights(hid_2[num_chunks:], wts_2[num_chunks:]).reshape(1, num_chunks * hid_2.shape[1], -1)

        prompt_embeds = torch.cat([hid_1_pos, hid_2_pos], dim=-1).to(device=exec_device, dtype=dtype)
        negative_prompt_embeds = torch.cat([hid_1_neg, hid_2_neg], dim=-1).to(device=exec_device, dtype=dtype)
        pooled_prompt_embeds = pooled_all[0:1].to(device=exec_device, dtype=dtype)
        negative_pooled_prompt_embeds = pooled_all[num_chunks : num_chunks + 1].to(device=exec_device, dtype=dtype)

        # Ensure text encoders are returned to CPU before UNet/VAE encode touch CUDA VRAM
        if self.device == "cuda" and hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()
            torch.cuda.empty_cache()

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def _encode_sd15_prompt(
        self,
        pipe: Any,
        positive_prompt: str,
        negative_prompt: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes unlimited-length weighted prompts for legacy SD 1.5 models."""
        exec_device = torch.device(self.device if self.device == "cuda" else "cpu")
        dtype = pipe.unet.dtype if pipe.unet is not None else self.torch_dtype

        pos_c = _tokenize_to_chunks(pipe.tokenizer, positive_prompt)
        neg_c = _tokenize_to_chunks(pipe.tokenizer, negative_prompt)
        num_chunks = max(len(pos_c), len(neg_c), 1)
        pos_c = _pad_chunk_list(pipe.tokenizer, pos_c, num_chunks)
        neg_c = _pad_chunk_list(pipe.tokenizer, neg_c, num_chunks)

        ids = torch.tensor([c[0] for c in (pos_c + neg_c)], dtype=torch.long, device=exec_device)
        wts = torch.tensor([c[1] for c in (pos_c + neg_c)], dtype=dtype, device=exec_device).unsqueeze(-1)
        with torch.inference_mode():
            out = pipe.text_encoder(ids)
            hid = out[0].to(dtype=dtype)

        pos_emb = _apply_chunk_weights(hid[:num_chunks], wts[:num_chunks]).reshape(1, num_chunks * hid.shape[1], -1)
        neg_emb = _apply_chunk_weights(hid[num_chunks:], wts[num_chunks:]).reshape(1, num_chunks * hid.shape[1], -1)
        return pos_emb, neg_emb

    def _resolve_base_dimensions(
        self,
        ratio_str: str,
        custom_w: Optional[int] = None,
        custom_h: Optional[int] = None,
        model_type: str = "sdxl"
    ) -> Tuple[int, int]:
        """Calculates native diffusion resolution aligned to multiples of 64."""
        if custom_w and custom_h and custom_w > 0 and custom_h > 0:
            w = (custom_w // 64) * 64
            h = (custom_h // 64) * 64
            lo, hi = (640, 1536) if model_type == "sdxl" else (384, 1024)
            return max(lo, min(hi, w)), max(lo, min(hi, h))

        if model_type == "sdxl":
            base_map = {
                "1:1": (1024, 1024),
                "square": (1024, 1024),
                "9:16": (768, 1344),
                "vertical": (768, 1344),
                "16:9": (1344, 768),
                "horizontal": (1344, 768),
                "4:3": (1152, 896),
                "3:4": (896, 1152),
            }
            return base_map.get(ratio_str, (1024, 1024))
        else:
            base_map = {
                "1:1": (512, 512),
                "square": (512, 512),
                "9:16": (576, 1024),
                "vertical": (576, 1024),
                "16:9": (1024, 576),
                "horizontal": (1024, 576),
                "4:3": (768, 576),
                "3:4": (576, 768),
            }
            return base_map.get(ratio_str, (512, 512))

    def _resolve_target_dimensions(
        self,
        ratio_str: str,
        custom_w: Optional[int] = None,
        custom_h: Optional[int] = None,
        model_type: str = "sdxl"
    ) -> Tuple[int, int]:
        """Calculates crisp final output dimensions per aspect ratio & engine."""
        if custom_w and custom_h and custom_w > 0 and custom_h > 0:
            return custom_w, custom_h

        sq = (1024, 1024) if model_type == "sdxl" else (512, 512)
        target_map = {
            "1:1": sq,
            "square": sq,
            "9:16": (1080, 1920),
            "vertical": (1080, 1920),
            "16:9": (1920, 1080),
            "horizontal": (1920, 1080),
            "4:3": (1440, 1080),
            "3:4": (1080, 1440),
        }
        return target_map.get(ratio_str, sq)

    def _fit_aspect_to_diffusion_bucket(
        self,
        orig_w: int,
        orig_h: int,
        model_type: str = "sdxl"
    ) -> Tuple[int, int]:
        """Computes a 64px-aligned diffusion resolution matching the input aspect ratio (~1MP for SDXL)."""
        aspect = max(0.35, min(2.8, orig_w / max(1, orig_h)))
        target_area = (1024 * 1024) if model_type == "sdxl" else (512 * 512)
        lo, hi = (640, 1536) if model_type == "sdxl" else (384, 1024)

        w = round(math.sqrt(target_area * aspect) / 64.0) * 64
        w = max(lo, min(hi, w))
        h = round((w / aspect) / 64.0) * 64
        h = max(lo, min(hi, h))
        return w, h

    def _resolve_aspect_dimensions(
        self,
        ratio_str: str,
        custom_w: Optional[int] = None,
        custom_h: Optional[int] = None,
        model_type: str = "sdxl"
    ):
        """Maintains backward compatibility returning target dimensions."""
        return self._resolve_target_dimensions(ratio_str, custom_w, custom_h, model_type=model_type)

    def _format_prompt(
        self,
        prompt: str,
        style_key: str = "auto",
        negative_prompt: Optional[str] = None,
        model_id: str = ""
    ):
        """
        Translates, cleans, and applies model-aware style enhancements.
        Ensures the user subject always stays first in CLIP cross-attention
        and never injects unintended subject exclusions.
        """
        prompt_en = smart_translate_prompt(prompt).strip()
        detected_style = detect_style_from_prompt(prompt_en)
        effective_style_key = style_key

        if style_key in ("auto", "", None):
            effective_style_key = detected_style if detected_style != "general" else "auto"

        style = STYLE_PRESETS.get(effective_style_key, STYLE_PRESETS["auto"])
        mid_low = (model_id or "").lower()

        # Assemble positive prompt (user prompt first!)
        suffix = style.get("prompt_suffix", "")
        if "animagine" in mid_low and style_key != "none":
            if effective_style_key in ("auto", "anime", "original"):
                suffix = ", masterpiece, high score, great score, absurdres"
        elif "openjourney" in mid_low and "mdjrny-v4 style" not in prompt_en.lower():
            suffix = f"{suffix}, mdjrny-v4 style"

        full_positive = f"{prompt_en}{suffix}"
        full_positive = re.sub(r',\s*,+', ', ', full_positive).strip(', ')

        # Assemble negative prompt without injecting any human/person exclusions
        neg_parts = []
        if negative_prompt and negative_prompt.strip():
            neg_parts.append(smart_translate_prompt(negative_prompt.strip()))

        if style.get("negative"):
            neg_parts.append(style["negative"])

        if style_key != "none":
            if "animagine" in mid_low:
                neg_parts.append("lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, blurry")
            else:
                neg_parts.append(DEFAULT_NEGATIVE_PROMPT)

        all_negs = ", ".join(neg_parts)
        seen_words = set()
        clean_negs = []
        for word in [w.strip() for w in all_negs.split(',') if w.strip()]:
            w_lower = word.lower()
            if w_lower not in seen_words:
                seen_words.add(w_lower)
                clean_negs.append(word)

        full_negative = ", ".join(clean_negs)
        return full_positive, full_negative, prompt_en, effective_style_key

    def _normalize_params(
        self,
        model_cfg: Dict[str, Any],
        steps: Optional[int],
        cfg_scale: Optional[float],
    ) -> Tuple[int, float]:
        """Normalizes steps and CFG cleanly per model (allowing 4-12 steps for Lightning/Turbo)."""
        min_s = 4 if model_cfg.get("is_distilled") else 10
        max_s = 50
        raw_steps = steps if (steps is not None and steps > 0) else model_cfg.get("default_steps", 24)
        norm_steps = max(min_s, min(max_s, int(raw_steps)))

        raw_cfg = cfg_scale if (cfg_scale is not None and cfg_scale > 0) else model_cfg.get("default_cfg", 6.0)
        norm_cfg = max(1.0, min(20.0, float(raw_cfg)))

        # Protect distilled Turbo/Lightning models from burnt contrast when passed legacy high CFG
        if model_cfg.get("is_distilled") and norm_cfg > model_cfg.get("max_cfg", 4.5):
            norm_cfg = float(model_cfg.get("default_cfg", 2.0))
        if model_cfg.get("is_distilled") and norm_steps > model_cfg.get("max_steps", 15):
            norm_steps = int(model_cfg.get("default_steps", 6))

        return norm_steps, norm_cfg

    def generate_text_to_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        style_preset: str = "auto",
        aspect_ratio: str = "1:1",
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        seed: Optional[int] = -1,
        model_id: str = DEFAULT_MODEL_ID
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Generates a high-resolution image from text using SDXL (1024px native) or SD 1.5.
        Supports >77 token prompts via chunked dual/single CLIP encoding.
        """
        t0 = time.time()
        model_id = (model_id or DEFAULT_MODEL_ID).strip()
        model_cfg = get_model_config(model_id)

        with self._lock:
            self._ensure_pipeline(model_id)
            if self.t2i_pipe is None:
                raise RuntimeError(f"Text-to-Image pipeline failed to initialize for {model_id}")

            base_w, base_h = self._resolve_base_dimensions(aspect_ratio, width, height, model_type=model_cfg["type"])
            target_w, target_h = self._resolve_target_dimensions(aspect_ratio, width, height, model_type=model_cfg["type"])

            full_prompt, full_negative, prompt_en, effective_style = self._format_prompt(
                prompt=prompt,
                style_key=style_preset,
                negative_prompt=negative_prompt,
                model_id=model_id
            )

            norm_steps, norm_cfg = self._normalize_params(model_cfg, steps, cfg_scale)
            actual_seed = random.randint(0, 2147483647) if (seed is None or seed < 0) else seed
            generator = torch.Generator(device=self.device).manual_seed(actual_seed)

            safe_print(
                f"[ImageGenerator] T2I [{model_id}]: '{prompt[:35]}...' -> EN: '{prompt_en[:45]}...' "
                f"(Base: {base_w}x{base_h} -> Target: {target_w}x{target_h}, steps={norm_steps}, cfg={norm_cfg}, style={effective_style}, seed={actual_seed})"
            )

            self._ensure_offload_mode("t2i")

            if self.current_model_type == "sdxl":
                p_emb, n_emb, p_pool, n_pool = self._encode_sdxl_prompt(self.t2i_pipe, full_prompt, full_negative)
                with torch.inference_mode():
                    result = self.t2i_pipe(
                        prompt=None,
                        negative_prompt=None,
                        prompt_embeds=p_emb,
                        negative_prompt_embeds=n_emb,
                        pooled_prompt_embeds=p_pool,
                        negative_pooled_prompt_embeds=n_pool,
                        width=base_w,
                        height=base_h,
                        num_inference_steps=norm_steps,
                        guidance_scale=norm_cfg,
                        generator=generator
                    )
            else:
                p_emb, n_emb = self._encode_sd15_prompt(self.t2i_pipe, full_prompt, full_negative)
                with torch.inference_mode():
                    result = self.t2i_pipe(
                        prompt=None,
                        negative_prompt=None,
                        prompt_embeds=p_emb,
                        negative_prompt_embeds=n_emb,
                        width=base_w,
                        height=base_h,
                        num_inference_steps=norm_steps,
                        guidance_scale=norm_cfg,
                        generator=generator
                    )

        raw_image = cast(Image.Image, result.images[0])
        if raw_image.size != (target_w, target_h):
            image = raw_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        else:
            image = raw_image

        duration = round(time.time() - t0, 2)
        meta: Dict[str, Any] = {
            "mode": "t2i",
            "prompt": prompt,
            "translated_prompt": prompt_en,
            "final_positive": full_prompt,
            "negative_prompt_used": full_negative,
            "style_preset": style_preset,
            "applied_style": effective_style,
            "width": target_w,
            "height": target_h,
            "base_width": base_w,
            "base_height": base_h,
            "steps": norm_steps,
            "cfg_scale": norm_cfg,
            "seed": actual_seed,
            "model_id": model_id,
            "model_type": model_cfg["type"],
            "scheduler_name": model_cfg.get("scheduler_name", "DPM++ 2M Karras"),
            "duration_sec": duration,
            "device": self.device.upper()
        }
        return image, meta

    def generate_image_to_image(
        self,
        input_image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        style_preset: str = "auto",
        aspect_ratio: str = "auto",
        strength: Optional[float] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        seed: Optional[int] = -1,
        model_id: str = DEFAULT_MODEL_ID
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Generates a transformed image from an existing input image with zero distortion.
        Supports 'auto' aspect ratio matching exact source proportions and guarantees
        at least 5 real denoising steps on distilled Turbo/Lightning models.
        """
        t0 = time.time()
        model_id = (model_id or DEFAULT_MODEL_ID).strip()
        model_cfg = get_model_config(model_id)

        with self._lock:
            self._ensure_pipeline(model_id)
            if self.i2i_pipe is None:
                raise RuntimeError(f"Image-to-Image pipeline failed to initialize for {model_id}")

            prompt_en_probe = smart_translate_prompt(prompt)
            if style_preset in ("auto", "original", "", None):
                detected_style = detect_style_from_prompt(prompt_en_probe)
                effective_preset = "original" if detected_style == "general" else detected_style
            else:
                effective_preset = style_preset

            full_prompt, full_negative, prompt_en, effective_style = self._format_prompt(
                prompt=prompt,
                style_key=effective_preset,
                negative_prompt=negative_prompt,
                model_id=model_id
            )

            orig_w, orig_h = input_image.size
            if aspect_ratio in ("auto", "original", "", None):
                base_w, base_h = self._fit_aspect_to_diffusion_bucket(orig_w, orig_h, model_type=model_cfg["type"])
                # For auto mode, keep crisp native SDXL ~1MP resolution proportional to source
                target_w, target_h = base_w, base_h
            else:
                base_w, base_h = self._resolve_base_dimensions(aspect_ratio, model_type=model_cfg["type"])
                target_w, target_h = self._resolve_target_dimensions(aspect_ratio, model_type=model_cfg["type"])

            # Use proportional center-crop fit so non-square -> square edits never squash subjects
            src_rgb = input_image.convert("RGB")
            src_ratio = orig_w / max(1, orig_h)
            dst_ratio = base_w / max(1, base_h)
            if abs(src_ratio - dst_ratio) > 0.04:
                prepared_input = ImageOps.fit(src_rgb, (base_w, base_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            else:
                prepared_input = src_rgb.resize((base_w, base_h), Image.Resampling.LANCZOS)

            norm_steps, norm_cfg = self._normalize_params(model_cfg, steps, cfg_scale)
            raw_strength = strength if (strength is not None and strength > 0) else model_cfg.get("default_strength", 0.55)
            norm_strength = max(0.15, min(0.95, float(raw_strength)))

            # Guarantee sufficient active denoising iterations (e.g. Lightning/Turbo I2I with strength 0.5 & steps 6)
            exec_steps = norm_steps
            min_active_steps = 5
            if int(exec_steps * norm_strength) < min_active_steps:
                exec_steps = min(50, max(norm_steps, math.ceil(min_active_steps / max(norm_strength, 0.15))))

            actual_seed = random.randint(0, 2147483647) if (seed is None or seed < 0) else seed
            generator = torch.Generator(device=self.device).manual_seed(actual_seed)

            safe_print(
                f"[ImageGenerator] I2I [{model_id}]: '{prompt[:35]}...' -> EN: '{prompt_en[:45]}...' "
                f"(Src: {orig_w}x{orig_h} -> Base: {base_w}x{base_h} -> Target: {target_w}x{target_h}, "
                f"strength={norm_strength}, steps={exec_steps}, cfg={norm_cfg}, style={effective_style}, seed={actual_seed})"
            )

            self._ensure_offload_mode("i2i")

            if self.current_model_type == "sdxl":
                p_emb, n_emb, p_pool, n_pool = self._encode_sdxl_prompt(self.i2i_pipe, full_prompt, full_negative)
                with torch.inference_mode():
                    result = self.i2i_pipe(
                        prompt=None,
                        negative_prompt=None,
                        prompt_embeds=p_emb,
                        negative_prompt_embeds=n_emb,
                        pooled_prompt_embeds=p_pool,
                        negative_pooled_prompt_embeds=n_pool,
                        image=prepared_input,
                        strength=norm_strength,
                        num_inference_steps=exec_steps,
                        guidance_scale=norm_cfg,
                        generator=generator
                    )
            else:
                p_emb, n_emb = self._encode_sd15_prompt(self.i2i_pipe, full_prompt, full_negative)
                with torch.inference_mode():
                    result = self.i2i_pipe(
                        prompt=None,
                        negative_prompt=None,
                        prompt_embeds=p_emb,
                        negative_prompt_embeds=n_emb,
                        image=prepared_input,
                        strength=norm_strength,
                        num_inference_steps=exec_steps,
                        guidance_scale=norm_cfg,
                        generator=generator
                    )

        raw_image = cast(Image.Image, result.images[0])
        if raw_image.size != (target_w, target_h):
            image = raw_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        else:
            image = raw_image

        duration = round(time.time() - t0, 2)
        meta: Dict[str, Any] = {
            "mode": "i2i",
            "prompt": prompt,
            "translated_prompt": prompt_en,
            "final_positive": full_prompt,
            "negative_prompt_used": full_negative,
            "style_preset": style_preset,
            "applied_style": effective_style,
            "original_width": orig_w,
            "original_height": orig_h,
            "width": target_w,
            "height": target_h,
            "base_width": base_w,
            "base_height": base_h,
            "strength": norm_strength,
            "steps": norm_steps,
            "effective_steps": exec_steps,
            "cfg_scale": norm_cfg,
            "seed": actual_seed,
            "model_id": model_id,
            "model_type": model_cfg["type"],
            "scheduler_name": model_cfg.get("scheduler_name", "DPM++ 2M Karras"),
            "duration_sec": duration,
            "device": self.device.upper()
        }
        return image, meta

    def get_models_info(self):
        """Returns list of supported models with config parameters and verified cache status."""
        models = []
        for m in SUPPORTED_MODELS:
            is_cached = self._resolve_local_snapshot(m["id"]) is not None
            item: Dict[str, Any] = dict(m)
            item["is_cached"] = is_cached
            models.append(item)
        return models

    def get_style_presets(self):
        """Returns list of style presets."""
        return [
            {"key": k, "name": v["name"]}
            for k, v in STYLE_PRESETS.items()
        ]

    def get_random_prompt(self):
        """Returns a random prompt suggestion."""
        return random.choice(PROMPT_SUGGESTIONS)
