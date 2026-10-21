"""
AI Image Generator Engine (T2I & I2I) for Clarity AI Studio
Powered by Hugging Face Diffusers on NVIDIA CUDA (FP16)
Supports Text-to-Image, Image-to-Image, Style Presets, and Local Model Caching
"""

import os
import sys
import time
import random
from typing import Optional, Dict, Any, List

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import torch
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,  # type: ignore
    StableDiffusionImg2ImgPipeline,  # type: ignore
    DPMSolverMultistepScheduler  # type: ignore
)

def safe_print(msg: str):
    """Safely print message without throwing UnicodeEncodeError on Windows terminals."""
    try:
        print(msg)
    except Exception:
        try:
            print(msg.encode('ascii', 'replace').decode('ascii'))
        except Exception:
            pass


# Supported Models List
SUPPORTED_MODELS = [
    {
        "id": "runwayml/stable-diffusion-v1-5",
        "name": "⚡ Stable Diffusion v1.5 (Local RTX · Siêu Nhanh ~2.5s)",
        "badge": "Khuyên dùng - Sẵn sàng",
        "type": "sd15",
        "desc": "Mô hình đa năng đã nạp sẵn cục bộ trên máy, tốc độ sinh ảnh cực nhanh 2-3 giây trên GPU RTX 3060."
    },
    {
        "id": "prompthero/openjourney",
        "name": "🌌 Openjourney v4 (Phong Cách Nghệ Thuật Midjourney)",
        "badge": "Artistic - Đã nạp sẵn",
        "type": "sd15",
        "desc": "Tái hiện phong cách hội họa nghệ thuật độc đáo, trừu tượng và ma mị tương tự Midjourney v4."
    },
    {
        "id": "Lykon/dreamshaper-8",
        "name": "🎨 DreamShaper v8 (Tranh 3D, Nghệ Thuật & Concept)",
        "badge": "Phổ biến",
        "type": "sd15",
        "desc": "Tối ưu hóa cho tranh vẽ nghệ thuật, hoạt hình 3D, nhân vật viễn tưởng và chân dung chi tiết."
    },
    {
        "id": "SG161222/Realistic_Vision_V5.1_noVAE",
        "name": "📸 Realistic Vision v5.1 (Chân Dung & Nhiếp Ảnh Thực Tế)",
        "badge": "Siêu nét",
        "type": "sd15",
        "desc": "Mô hình chuyên dụng tái tạo chất lượng da người, ánh sáng studio và chi tiết phong cảnh như máy ảnh DSLR."
    }
]

# Smart Bilingual Translation & Semantic Understanding
def smart_translate_prompt(text: str) -> str:
    """Translates Vietnamese prompt to English if needed with multi-tier fallback to prevent rate limits."""
    text_clean = text.strip()
    if not text_clean:
        return ""
    vietnamese_chars = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
                           "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ")
    if not any(c in vietnamese_chars for c in text_clean):
        return text_clean

    import urllib.request, urllib.parse, json

    # Engine 1: Chrome Extension endpoint (ultra-fast, highly reliable, no 429)
    try:
        url = "https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl=auto&tl=en&q=" + urllib.parse.quote(text_clean)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], list) and len(data[0]) > 0:
                    t = data[0][0]
                    if t and t.strip():
                        return t.strip()
                elif isinstance(data[0], str):
                    return data[0].strip()
    except Exception as e:
        safe_print(f"[ImageGenerator] Translation Engine 1 fallback: {e}")

    # Engine 2: MyMemory API fallback
    try:
        url = "https://api.mymemory.translated.net/get?q=" + urllib.parse.quote(text_clean) + "&langpair=vi|en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3.5) as resp:
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
        import re
        t = text_clean.lower()
        replacements = []
        for pat, en in VIETNAMESE_CONCEPT_MAP.items():
            if re.search(pat, t):
                replacements.append(en)
        if replacements:
            return ", ".join(replacements)
    except Exception:
        pass

    return text_clean


def detect_style_from_prompt(prompt_en: str) -> str:
    """Detects artistic style from English prompt keywords to prevent style mismatch."""
    p = prompt_en.lower()
    if any(k in p for k in ["anime", "manga", "waifu", "shinkai", "chibi", "2d illustration", "cel shaded", "comic", "pixiv", "hoạt hình nhật"]):
        return "anime"
    if any(k in p for k in ["3d", "pixar", "disney", "cgi", "octane render", "clay", "unreal engine", "3d character"]):
        return "pixar3d"
    if any(k in p for k in ["cyberpunk", "neon", "sci-fi", "futuristic", "mecha", "blade runner", "night city"]):
        return "cyberpunk"
    if any(k in p for k in ["photo", "portrait", "dslr", "photograph", "real life", "skin texture", "85mm", "raw photo"]):
        return "photorealistic"
    if any(k in p for k in ["fantasy", "dragon", "magic", "mythical", "castle", "ethereal", "wizard", "elf"]):
        return "fantasy"
    if any(k in p for k in ["vintage", "retro", "1970s", "1980s", "film grain", "analog", "kodak"]):
        return "vintage"
    if any(k in p for k in ["cinematic", "film still", "movie scene", "dramatic lighting", "anamorphic"]):
        return "cinematic"
    return "general"


# Curated Style Presets (Subject-First, No Forced Scenery/Characters)
STYLE_PRESETS = {
    "auto": {
        "name": "✨ Tự động bám sát Prompt & Phong cách ảnh gốc (Auto)",
        "prompt_prefix": "",
        "prompt_suffix": ", masterpiece, best quality, ultra-detailed, sharp focus, 8k uhd",
        "negative": "worst quality, low quality, normal quality, blurry, lowres, bad anatomy, bad hands, deformed, watermark"
    },
    "none": {
        "name": "Tự do (Nguyên bản theo Prompt)",
        "prompt_prefix": "",
        "prompt_suffix": ", sharp focus, high quality",
        "negative": "worst quality, low quality, blurry, distorted"
    },
    "original": {
        "name": "🖼️ Giữ nguyên phong cách & màu sắc ảnh gốc (Style Lock I2I)",
        "prompt_prefix": "",
        "prompt_suffix": ", masterpiece, highly detailed, crisp clean lines, vibrant colors, preserve original artwork style",
        "negative": "worst quality, low quality, blurry, noise, artifacts, distorted features, bad anatomy, watermark"
    },
    "anime": {
        "name": "🌸 Anime Makoto Shinkai (Hoạt Hình Nhật Bản)",
        "prompt_prefix": "",
        "prompt_suffix": ", anime art style, vibrant colors, detailed anime digital illustration, trending on pixiv, masterpiece, 4k",
        "negative": "photorealistic, real life photo, western cartoon, 3d render, low quality, bad anatomy, deformed"
    },
    "photorealistic": {
        "name": "📸 Nhiếp Ảnh Đời Thực (Photorealistic DSLR)",
        "prompt_prefix": "",
        "prompt_suffix": ", professional photograph, natural lighting, sharp focus, 85mm portrait lens, raw photo, 8k uhd, crisp details",
        "negative": "painting, drawing, illustration, anime, cartoon, 3d render, airbrushed, plastic skin, deformed, blurry"
    },
    "cinematic": {
        "name": "🎬 Điện Ảnh 8K (Cinematic Movie)",
        "prompt_prefix": "",
        "prompt_suffix": ", cinematic atmosphere, dramatic volumetric lighting, sharp focus, 8k uhd, photorealistic, color graded masterpiece",
        "negative": "cartoon, 3d render, blurry, low resolution, oversaturated, amateur photography"
    },
    "pixar3d": {
        "name": "🧸 3D Hoạt Hình Pixar / Disney",
        "prompt_prefix": "",
        "prompt_suffix": ", cute 3D character render, modern Pixar Disney studio style, smooth clay textures, soft studio lighting, octane render, ray tracing",
        "negative": "realistic human photo, creepy, ugly, deformed, dark, grainy, 2d drawing, sketch"
    },
    "cyberpunk": {
        "name": "🌃 Cyberpunk Neon (Khoa Học Viễn Tưởng)",
        "prompt_prefix": "",
        "prompt_suffix": ", cyberpunk aesthetic, vibrant glowing neon lights, futuristic reflections, volumetric mist, dark night atmosphere, 8k octane render",
        "negative": "daylight, vintage, traditional, blurry, lowres, oversaturated, bad composition"
    },
    "fantasy": {
        "name": "🏰 Thần Thoại & Giả Tưởng (Epic Fantasy)",
        "prompt_prefix": "",
        "prompt_suffix": ", epic fantasy digital painting, mystical glowing ambiance, intricate ethereal details, concept art, artstation trending, masterpiece",
        "negative": "modern, futuristic, cars, blurry, poor quality, bad proportions, watermark"
    },
    "vintage": {
        "name": "🎞️ Phim Cổ Điển Retro (Vintage 1970s)",
        "prompt_prefix": "",
        "prompt_suffix": ", vintage 1970s Kodak film photograph, muted retro warm tones, subtle film grain, nostalgic mood, classic analog camera texture",
        "negative": "modern digital, ultra-sharp, cgi, futuristic, oversaturated, clean"
    }
}

DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, low quality, normal quality, lowres, blurry, bad anatomy, "
    "bad hands, missing fingers, extra digits, deformed, disfigured, poorly drawn face, "
    "bad eyes, mutation, extra limbs, watermark, signature, text, out of focus, cropped"
)

PROMPT_SUGGESTIONS = [
    {
        "prompt": "Chân dung phi hành gia mèo máy trong bộ giáp không gian phát sáng neon tím, bồng bềnh giữa dải ngân hà lấp lánh",
        "style": "cyberpunk",
        "ratio": "1:1"
    },
    {
        "prompt": "Cô gái anime ngồi đọc sách bên cửa sổ quán cà phê ngắm mưa rơi ngoài phố Tokyo, ánh đèn vàng ấm áp",
        "style": "anime",
        "ratio": "9:16"
    },
    {
        "prompt": "Siêu xe thể thao tương lai chạy trên đường cao tốc ven biển lúc hoàng hôn đỏ rực, mặt đường phản chiếu ánh nước",
        "style": "cinematic",
        "ratio": "16:9"
    },
    {
        "prompt": "Chú rồng con tí hon dễ thương màu ngọc bích ôm viên pha lê phát sáng trong hang động thạch nhũ kỳ ảo",
        "style": "pixar3d",
        "ratio": "1:1"
    },
    {
        "prompt": "Bức chân dung nghệ thuật cô gái với ánh mắt trong veo, mái tóc bồng bềnh cài hoa dại dưới nắng sớm",
        "style": "photorealistic",
        "ratio": "9:16"
    },
    {
        "prompt": "Lâu đài cổ tích nguy nga bay lơ lửng giữa những đám mây ngũ sắc với thác nước đổ xuống hư không",
        "style": "fantasy",
        "ratio": "16:9"
    }
]


class ImageGenerator:
    """
    Manages Diffusion models for Text-to-Image (T2I) and Image-to-Image (I2I).
    Uses shared weights between T2I and I2I pipelines to conserve GPU VRAM.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.current_model_id = None
        self.t2i_pipe: Any = None
        self.i2i_pipe: Any = None
        print(f"[ImageGenerator] Initialized on device: {self.device.upper()} (FP16: {self.torch_dtype == torch.float16})")

    def _ensure_pipeline(self, model_id: str):
        """Loads or switches model pipelines if needed."""
        if not model_id:
            model_id = "runwayml/stable-diffusion-v1-5"

        if self.current_model_id == model_id and self.t2i_pipe is not None:
            return

        print(f"[ImageGenerator] Loading model: {model_id}...")
        t0 = time.time()

        # Clean old pipeline memory
        if self.t2i_pipe is not None:
            del self.t2i_pipe
            del self.i2i_pipe
            self.t2i_pipe = None
            self.i2i_pipe = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

        # Load T2I Pipeline
        try:
            # Check local files first
            self.t2i_pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=True
            )
        except Exception:
            # Download if not local
            print(f"[ImageGenerator] Model not cached locally, downloading {model_id} from Hugging Face...")
            self.t2i_pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
                requires_safety_checker=False
            )

        # Optimize Scheduler
        self.t2i_pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.t2i_pipe.scheduler.config)

        # Move to device & optimize memory
        self.t2i_pipe.to(self.device)
        if self.device == "cuda":
            self.t2i_pipe.enable_attention_slicing()
            try:
                self.t2i_pipe.enable_vae_slicing()
            except Exception:
                pass

        # Share components with I2I pipeline (Zero extra VRAM!)
        i2i_cls: Any = StableDiffusionImg2ImgPipeline
        self.i2i_pipe = i2i_cls(
            vae=self.t2i_pipe.vae,
            text_encoder=self.t2i_pipe.text_encoder,
            tokenizer=self.t2i_pipe.tokenizer,
            unet=self.t2i_pipe.unet,
            scheduler=self.t2i_pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        self.i2i_pipe.to(self.device)
        if self.device == "cuda":
            self.i2i_pipe.enable_attention_slicing()
            try:
                self.i2i_pipe.enable_vae_slicing()
            except Exception:
                pass

        self.current_model_id = model_id
        print(f"[ImageGenerator] Pipeline {model_id} ready in {time.time() - t0:.2f}s!")

    def _resolve_aspect_dimensions(self, ratio_str: str, custom_w: Optional[int] = None, custom_h: Optional[int] = None):
        """Calculates pixel dimensions aligned to multiples of 64 optimized for SD1.5 without duplication."""
        if custom_w and custom_h and custom_w > 0 and custom_h > 0:
            w = (int(custom_w) // 64) * 64
            h = (int(custom_h) // 64) * 64
            return max(256, min(1024, w)), max(256, min(1024, h))

        # Optimized SD1.5 resolutions avoiding repetitive double heads
        ratio_map = {
            "1:1": (512, 512),
            "9:16": (512, 768),
            "16:9": (768, 512),
            "4:3": (640, 480),
            "3:4": (480, 640),
            "square": (512, 512),
            "vertical": (512, 768),
            "horizontal": (768, 512)
        }
        return ratio_map.get(ratio_str, (512, 512))

    def _format_prompt(self, prompt: str, style_key: str = "auto", negative_prompt: Optional[str] = None, model_id: str = ""):
        """
        Translates, cleans, and applies appropriate style enhancements without clashing.
        Ensures the user subject is strictly FIRST in CLIP cross-attention.
        """
        import re
        # 1. Smart translation
        prompt_en = smart_translate_prompt(prompt).strip()

        # 2. Determine effective style
        detected_style = detect_style_from_prompt(prompt_en)
        effective_style_key = style_key

        if style_key in ("auto", "", None):
            effective_style_key = detected_style if detected_style != "general" else "auto"

        style = STYLE_PRESETS.get(effective_style_key, STYLE_PRESETS["auto"])

        # 3. Assemble Positive Prompt (SUBJECT IS ALWAYS FIRST)
        suffix = style.get("prompt_suffix", "")
        
        # Check if model is openjourney and add trigger token if not present
        extra_model_token = ""
        if "openjourney" in str(model_id).lower() and "mdjrny-v4 style" not in prompt_en.lower():
            extra_model_token = ", mdjrny-v4 style"

        full_positive = f"{prompt_en}{suffix}{extra_model_token}"
        full_positive = re.sub(r',\s*,+', ', ', full_positive).strip(', ')

        # 4. Assemble Negative Prompt
        neg_parts = []
        if negative_prompt and negative_prompt.strip():
            translated_neg = smart_translate_prompt(negative_prompt.strip())
            neg_parts.append(translated_neg)

        if style.get("negative"):
            neg_parts.append(style["negative"])

        neg_parts.append(DEFAULT_NEGATIVE_PROMPT)

        # Detect human/character intent in prompt
        # If user did NOT ask for humans (e.g. dragon, animal, car, landscape, room, still life, food),
        # prevent model priors from hallucinating humans/crowds/faces!
        prompt_lower = f" {prompt_en.lower()} "
        human_indicators = [
            "person", "people", "man", "men", "woman", "women", "girl", "boy", "kid", "child",
            "children", "human", "portrait", "lady", "guy", "warrior", "samurai",
            "ninja", "soldier", "couple", "lovers", "bride", "groom", "witch", "wizard", "hero",
            "heroine", "queen", "king", "princess", "prince", "crowd", "model", "student", "teacher",
            "anime girl", "anime boy", "selfie", "human baby", "baby girl", "baby boy"
        ]
        has_human_intent = any(re.search(r'\b' + re.escape(h) + r'\b', prompt_lower) for h in human_indicators)

        if not has_human_intent:
            neg_parts.append("human, person, people, man, woman, girl, boy, couple, crowd, human face, human skin")

        # De-duplicate negative keywords
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

    def generate_text_to_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        style_preset: str = "auto",
        aspect_ratio: str = "1:1",
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: int = 25,
        cfg_scale: float = 8.0,
        seed: int = -1,
        model_id: str = "runwayml/stable-diffusion-v1-5"
    ):
        """Generates an image from text prompt with full semantic understanding."""
        t0 = time.time()
        self._ensure_pipeline(model_id)

        w, h = self._resolve_aspect_dimensions(aspect_ratio, width, height)
        full_prompt, full_negative, prompt_en, effective_style = self._format_prompt(
            prompt=prompt,
            style_key=style_preset,
            negative_prompt=negative_prompt,
            model_id=model_id
        )

        # Seed handling
        if seed is None or int(seed) < 0:
            actual_seed = random.randint(0, 2147483647)
        else:
            actual_seed = int(seed)

        generator = torch.Generator(device=self.device).manual_seed(actual_seed)

        steps = max(10, min(50, int(steps or 25)))
        cfg_scale = max(1.0, min(20.0, float(cfg_scale or 7.5)))

        safe_print(f"[ImageGenerator] T2I: '{prompt[:35]}...' -> EN: '{prompt_en[:35]}...' ({w}x{h}, {steps} steps, style: {effective_style}, seed {actual_seed})")

        with torch.inference_mode():
            result = self.t2i_pipe(
                prompt=full_prompt,
                negative_prompt=full_negative,
                width=w,
                height=h,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator
            )

        image = result.images[0]
        duration = round(time.time() - t0, 2)

        meta = {
            "mode": "t2i",
            "prompt": prompt,
            "translated_prompt": prompt_en,
            "final_positive": full_prompt,
            "style_preset": style_preset,
            "applied_style": effective_style,
            "width": w,
            "height": h,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": actual_seed,
            "model_id": model_id,
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
        strength: float = 0.45,
        steps: int = 25,
        cfg_scale: float = 7.5,
        seed: int = -1,
        model_id: str = "runwayml/stable-diffusion-v1-5"
    ):
        """
        Generates a transformed image from an existing input image.
        Uses strength 0.45 default to preserve the source image style, composition, and colors.
        """
        t0 = time.time()
        self._ensure_pipeline(model_id)

        # In I2I, if style is auto/original and prompt doesn't ask for a new style, preserve original artwork style
        if style_preset in ("auto", "original", "", None):
            detected_style = detect_style_from_prompt(smart_translate_prompt(prompt))
            if detected_style == "general":
                effective_preset = "original"
            else:
                effective_preset = detected_style
        else:
            effective_preset = style_preset

        full_prompt, full_negative, prompt_en, effective_style = self._format_prompt(
            prompt=prompt,
            style_key=effective_preset,
            negative_prompt=negative_prompt,
            model_id=model_id
        )

        # Maintain exact aspect ratio aligned to multiples of 64
        orig_w, orig_h = input_image.size
        # For optimal SD1.5 diffusion quality, ensure target dimensions are at least 512 on the longer side (and max 768)
        if max(orig_w, orig_h) < 512:
            scale = 512.0 / max(orig_w, orig_h)
        else:
            scale = min(768.0 / max(orig_w, orig_h), 1.0)

        target_w = max(384, int(round((orig_w * scale) / 64.0)) * 64)
        target_h = max(384, int(round((orig_h * scale) / 64.0)) * 64)
        target_w = min(1024, target_w)
        target_h = min(1024, target_h)

        resized_image = input_image.convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Seed handling
        if seed is None or int(seed) < 0:
            actual_seed = random.randint(0, 2147483647)
        else:
            actual_seed = int(seed)

        generator = torch.Generator(device=self.device).manual_seed(actual_seed)

        steps = max(10, min(50, int(steps or 25)))
        cfg_scale = max(1.0, min(20.0, float(cfg_scale or 7.5)))
        strength = max(0.1, min(0.95, float(strength or 0.45)))

        safe_print(f"[ImageGenerator] I2I: '{prompt[:35]}...' -> EN: '{prompt_en[:35]}...' ({target_w}x{target_h}, strength {strength}, style: {effective_style}, seed {actual_seed})")

        with torch.inference_mode():
            result = self.i2i_pipe(
                prompt=full_prompt,
                negative_prompt=full_negative,
                image=resized_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator
            )

        image = result.images[0]
        duration = round(time.time() - t0, 2)

        meta = {
            "mode": "i2i",
            "prompt": prompt,
            "translated_prompt": prompt_en,
            "final_positive": full_prompt,
            "style_preset": style_preset,
            "applied_style": effective_style,
            "original_width": orig_w,
            "original_height": orig_h,
            "width": target_w,
            "height": target_h,
            "strength": strength,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": actual_seed,
            "model_id": model_id,
            "duration_sec": duration,
            "device": self.device.upper()
        }
        return image, meta

    def get_models_info(self):
        """Returns list of supported models and their local cached status."""
        models = []
        for m in SUPPORTED_MODELS:
            is_cached = False
            # Check HuggingFace hub cache
            cache_folder = os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "huggingface",
                "hub",
                f"models--{m['id'].replace('/', '--')}"
            )
            if os.path.exists(cache_folder):
                is_cached = True

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
