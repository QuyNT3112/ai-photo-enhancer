"""
AI Image Generator Engine (T2I & I2I) for Clarity AI Studio
Powered by Hugging Face Diffusers on NVIDIA CUDA (FP16)
Supports Text-to-Image, Image-to-Image, Style Presets, and Local Model Caching
"""

import os
import time
import random
from typing import Optional, Dict, Any, List
import torch
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,  # type: ignore
    StableDiffusionImg2ImgPipeline,  # type: ignore
    DPMSolverMultistepScheduler  # type: ignore
)

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
    },
    {
        "id": "prompthero/openjourney",
        "name": "🌌 Openjourney v4 (Phong Cách Midjourney)",
        "badge": "Artistic",
        "type": "sd15",
        "desc": "Tái hiện phong cách hội họa nghệ thuật độc đáo, trừu tượng và ma mị tương tự Midjourney."
    }
]

# Curated Style Presets
STYLE_PRESETS = {
    "none": {
        "name": "Tự do (Không áp dụng phong cách)",
        "prompt_prefix": "",
        "prompt_suffix": ", high quality, detailed",
        "negative": "blurry, low quality, distorted"
    },
    "cinematic": {
        "name": "🎬 Điện Ảnh 8K (Cinematic Movie)",
        "prompt_prefix": "cinematic film still, 35mm photograph, dramatic lighting, ",
        "prompt_suffix": ", anamorphic lens flare, sharp focus, 8k uhd, photorealistic, masterpiece",
        "negative": "cartoon, 3d render, blurry, low resolution, oversaturated, amateur photography"
    },
    "photorealistic": {
        "name": "📸 Nhiếp Ảnh Đời Thực (Photorealistic DSLR)",
        "prompt_prefix": "professional ultra-detailed photograph, realistic skin texture, natural studio lighting, ",
        "prompt_suffix": ", 85mm portrait lens, f/1.8 aperture, raw photo, 8k resolution, crisp details",
        "negative": "painting, drawing, illustration, anime, cartoon, 3d, airbrushed, plastic skin, distorted anatomy"
    },
    "anime": {
        "name": "🌸 Anime Makoto Shinkai (Hoạt Hình Nhật Bản)",
        "prompt_prefix": "anime aesthetic masterpiece, makoto shinkai style, vivid colors, atmospheric scenery, ",
        "prompt_suffix": ", stunning sky, beautiful detailed digital illustration, trending on pixiv, 4k",
        "negative": "photorealistic, real life photo, western cartoon, low quality, bad anatomy, deformed"
    },
    "cyberpunk": {
        "name": "🌃 Cyberpunk Neon (Khoa Học Viễn Tưởng)",
        "prompt_prefix": "cyberpunk sci-fi aesthetic, vibrant glowing neon lights, futuristic city reflections, ",
        "prompt_suffix": ", volumetric mist, dark night atmosphere, high tech cyberpunk art, 8k octane render",
        "negative": "daylight, vintage, traditional, blurry, lowres, oversaturated, bad composition"
    },
    "pixar3d": {
        "name": "🧸 3D Hoạt Hình Pixar / Disney",
        "prompt_prefix": "cute 3d character render, pixar disney modern animation style, smooth clay textures, ",
        "prompt_suffix": ", expressive eyes, soft studio lighting, octane render, unreal engine 5, ray tracing",
        "negative": "realistic photo, creepy, ugly, deformed, dark, grainy, 2d drawing"
    },
    "fantasy": {
        "name": "🏰 Thần Thoại & Giả Tưởng (Epic Fantasy)",
        "prompt_prefix": "epic fantasy digital painting, mystical glowing ambiance, intricate ethereal details, ",
        "prompt_suffix": ", concept art by greg rutkowski, artstation trending, dramatic majestic composition",
        "negative": "modern, futuristic, cars, blurry, poor quality, bad proportions, watermark"
    },
    "vintage": {
        "name": "🎞️ Phim Cổ Điển Retro (Vintage 1970s)",
        "prompt_prefix": "vintage 1970s kodak film photograph, muted retro warm tones, subtle film grain, ",
        "prompt_suffix": ", nostalgic mood, classic composition, authentic analog camera texture",
        "negative": "modern digital, ultra-sharp, cgi, futuristic, oversaturated, clean"
    }
}

DEFAULT_NEGATIVE_PROMPT = (
    "ugly, deformed, disfigured, poor anatomy, bad hands, missing fingers, "
    "extra digits, blurry, low quality, pixelated, watermark, signature, grainy, "
    "low resolution, bad eyes, poorly drawn face, out of focus, cropped"
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

        self.current_model_id = model_id
        print(f"[ImageGenerator] Pipeline {model_id} ready in {time.time() - t0:.2f}s!")

    def _resolve_aspect_dimensions(self, ratio_str: str, custom_w: Optional[int] = None, custom_h: Optional[int] = None):
        """Calculates pixel dimensions aligned to multiples of 64."""
        if custom_w and custom_h and custom_w > 0 and custom_h > 0:
            w = (int(custom_w) // 64) * 64
            h = (int(custom_h) // 64) * 64
            return max(256, min(1024, w)), max(256, min(1024, h))

        ratio_map = {
            "1:1": (512, 512),
            "9:16": (512, 896),
            "16:9": (896, 512),
            "4:3": (640, 480),
            "3:4": (480, 640),
            "square": (512, 512),
            "vertical": (512, 896),
            "horizontal": (896, 512)
        }
        return ratio_map.get(ratio_str, (512, 512))

    def _format_prompt(self, prompt: str, style_key: str, negative_prompt: Optional[str] = None):
        """Applies style presets to positive and negative prompts."""
        style = STYLE_PRESETS.get(style_key, STYLE_PRESETS["none"])
        
        full_positive = f"{style['prompt_prefix']}{prompt.strip()}{style['prompt_suffix']}"
        
        neg_parts = []
        if negative_prompt and negative_prompt.strip():
            neg_parts.append(negative_prompt.strip())
        if style.get("negative"):
            neg_parts.append(style["negative"])
        neg_parts.append(DEFAULT_NEGATIVE_PROMPT)
        
        full_negative = ", ".join(neg_parts)
        return full_positive, full_negative

    def generate_text_to_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        style_preset: str = "none",
        aspect_ratio: str = "1:1",
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: int = 20,
        cfg_scale: float = 7.5,
        seed: int = -1,
        model_id: str = "runwayml/stable-diffusion-v1-5"
    ):
        """Generates an image from text prompt."""
        t0 = time.time()
        self._ensure_pipeline(model_id)

        w, h = self._resolve_aspect_dimensions(aspect_ratio, width, height)
        full_prompt, full_negative = self._format_prompt(prompt, style_preset, negative_prompt)

        # Seed handling
        if seed is None or int(seed) < 0:
            actual_seed = random.randint(0, 2147483647)
        else:
            actual_seed = int(seed)

        generator = torch.Generator(device=self.device).manual_seed(actual_seed)

        steps = max(10, min(50, int(steps)))
        cfg_scale = max(1.0, min(20.0, float(cfg_scale)))

        print(f"[ImageGenerator] T2I: '{prompt[:40]}...' ({w}x{h}, {steps} steps, seed {actual_seed})")

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
            "style_preset": style_preset,
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
        style_preset: str = "none",
        strength: float = 0.65,
        steps: int = 20,
        cfg_scale: float = 7.5,
        seed: int = -1,
        model_id: str = "runwayml/stable-diffusion-v1-5"
    ):
        """Generates a transformed image from an existing input image."""
        t0 = time.time()
        self._ensure_pipeline(model_id)

        full_prompt, full_negative = self._format_prompt(prompt, style_preset, negative_prompt)

        # Resize input image to multiples of 64 preserving max dimension ~768
        orig_w, orig_h = input_image.size
        scale = min(768 / orig_w, 768 / orig_h, 1.0)
        target_w = max(256, (int(orig_w * scale) // 64) * 64)
        target_h = max(256, (int(orig_h * scale) // 64) * 64)
        
        resized_image = input_image.convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Seed handling
        if seed is None or int(seed) < 0:
            actual_seed = random.randint(0, 2147483647)
        else:
            actual_seed = int(seed)

        generator = torch.Generator(device=self.device).manual_seed(actual_seed)

        steps = max(10, min(50, int(steps)))
        cfg_scale = max(1.0, min(20.0, float(cfg_scale)))
        strength = max(0.1, min(0.95, float(strength)))

        print(f"[ImageGenerator] I2I: '{prompt[:40]}...' ({target_w}x{target_h}, strength {strength}, seed {actual_seed})")

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
            "style_preset": style_preset,
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
