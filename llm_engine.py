"""
Local LLM & Vision Engine for Clarity AI Studio
Provides:
1. Natural Language Prompt Optimizer (Vietnamese/English -> Masterpiece SD Prompt)
2. Photography & Color Chatbot Consultant
3. Image-to-Prompt & Visual Feature / Color Palette Extractor
4. Multi-Provider Local LLM Hub:
   - Built-in Local Fast Hybrid Engine (100% Free, Zero-setup, Offline)
   - Ollama Local Server (http://127.0.0.1:11434 - Llama 3.2, Qwen 2.5, Llava, etc.)
   - LM Studio / Local OpenAI API (http://127.0.0.1:1234/v1)
   - Hugging Face Local Transformers (PyTorch CUDA/CPU)
"""

import os
import re
import json
import base64
import time
import math
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple

import numpy as np  # type: ignore
from PIL import Image  # type: ignore
import requests  # type: ignore

kmeans: Any = None
try:
    from scipy.cluster.vq import kmeans as _scipy_kmeans  # type: ignore
    kmeans = _scipy_kmeans
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

torch: Any = None
try:
    import torch as _torch  # type: ignore
    torch = _torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


# ============================================================================
# 1. COLOR THEORY & PALETTE UTILITIES
# ============================================================================

COLOR_NAMES_MAP = [
    ((255, 0, 0), "Đỏ Rực (Vibrant Red)", "red"),
    ((220, 20, 60), "Đỏ Thẫm (Crimson)", "red"),
    ((255, 105, 180), "Hồng Sen (Hot Pink)", "pink"),
    ((255, 182, 193), "Hồng Phấn (Pastel Pink)", "pink"),
    ((255, 140, 0), "Cam Đậm (Dark Orange)", "orange"),
    ((255, 165, 0), "Cam Cà Rốt (Vivid Orange)", "orange"),
    ((255, 215, 0), "Vàng Hoàng Kim (Golden Yellow)", "yellow"),
    ((255, 255, 0), "Vàng Tươi (Pure Yellow)", "yellow"),
    ((240, 230, 140), "Vàng Khaki / Cát (Warm Khaki)", "yellow"),
    ((0, 128, 0), "Xanh Lá Cây (Forest Green)", "green"),
    ((50, 205, 50), "Xanh Lục Bảo (Lime Green)", "green"),
    ((0, 255, 127), "Xanh Bạc Hà (Spring Green)", "green"),
    ((0, 255, 255), "Xanh Neon / Cyan (Cyberpunk Cyan)", "cyan"),
    ((64, 224, 208), "Xanh Ngọc Lam (Turquoise)", "cyan"),
    ((30, 144, 255), "Xanh Da Trời (Dodger Blue)", "blue"),
    ((0, 0, 255), "Xanh Lam Đậm (Deep Blue)", "blue"),
    ((25, 25, 112), "Xanh Đêm / Midnight (Midnight Navy)", "blue"),
    ((138, 43, 226), "Tím Neon (Vivid Violet)", "purple"),
    ((75, 0, 130), "Tím Thẫm / Indigo (Deep Indigo)", "purple"),
    ((218, 112, 214), "Tím Phong Lan (Orchid Lavender)", "purple"),
    ((139, 69, 19), "Nâu Đất (Saddle Brown)", "brown"),
    ((210, 180, 140), "Nâu Nhạt / Be (Warm Beige)", "brown"),
    ((255, 255, 255), "Trắng Tinh Khôi (Pure White)", "white"),
    ((245, 245, 245), "Trắng Khói (Smoke White)", "white"),
    ((192, 192, 192), "Bạc / Xám Nhạt (Light Silver)", "gray"),
    ((128, 128, 128), "Xám Trung Tính (Neutral Gray)", "gray"),
    ((47, 79, 79), "Xám Đen Đậm (Dark Slate)", "gray"),
    ((20, 20, 20), "Đen Carbon (Carbon Black)", "black"),
    ((0, 0, 0), "Đen Tuyệt Đối (Pure Black)", "black")
]


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}".upper()


def get_color_name(r: int, g: int, b: int) -> Tuple[str, str]:
    best_dist = float('inf')
    best_name = "Màu Không Xác Định"
    best_cat = "other"
    for (cr, cg, cb), name, cat in COLOR_NAMES_MAP:
        # Euclidean distance in RGB color space
        dist = math.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
        if dist < best_dist:
            best_dist = dist
            best_name = name
            best_cat = cat
    return best_name, best_cat


def extract_dominant_colors(pil_image: Image.Image, num_colors: int = 5) -> List[Dict[str, Any]]:
    """
    Extracts top dominant colors using fast K-Means clustering or histogram fallback.
    Returns list of {hex, rgb, percentage, name, category, is_light}.
    """
    # Resize small for lightning speed
    img = pil_image.convert('RGB')
    img.thumbnail((150, 150), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape
    pixels = arr.reshape(-1, 3)

    results = []

    if HAS_SCIPY and kmeans is not None and len(pixels) > 50:
        try:
            # Subsample for speed
            sample_step = max(1, len(pixels) // 2000)
            sample_pixels = pixels[::sample_step]
            centroids, _ = kmeans(sample_pixels, num_colors, iter=10)

            # Assign each pixel to nearest centroid to calculate percentages
            dists = np.linalg.norm(sample_pixels[:, None, :] - centroids[None, :, :], axis=2)
            assignments = np.argmin(dists, axis=1)
            counts = np.bincount(assignments, minlength=len(centroids))
            total_samples = float(len(sample_pixels))

            # Sort by frequency descending
            sorted_indices = np.argsort(-counts)

            for idx in sorted_indices:
                cnt = float(counts[idx])
                pct = round((cnt / total_samples) * 100.0, 1)
                if pct < 1.0 and len(results) >= 3:
                    continue
                c = centroids[idx]
                r, g, b = int(round(float(c[0]))), int(round(float(c[1]))), int(round(float(c[2])))
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                hex_code = rgb_to_hex(r, g, b)
                name, cat = get_color_name(r, g, b)
                # Perceived brightness formula (ITU-R BT.709)
                luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b)
                results.append({
                    'hex': hex_code,
                    'rgb': [r, g, b],
                    'percentage': pct,
                    'name': name,
                    'category': cat,
                    'is_light': luminance > 130
                })
        except Exception:
            results = []

    if not results:
        # Fallback simple grid sampling
        step = max(1, len(pixels) // num_colors)
        for i in range(num_colors):
            p = pixels[min(len(pixels) - 1, i * step)]
            r, g, b = int(p[0]), int(p[1]), int(p[2])
            name, cat = get_color_name(r, g, b)
            results.append({
                'hex': rgb_to_hex(r, g, b),
                'rgb': [r, g, b],
                'percentage': round(100.0 / num_colors, 1),
                'name': name,
                'category': cat,
                'is_light': (0.2126 * r + 0.7152 * g + 0.0722 * b) > 130
            })

    return results


def analyze_image_metrics(pil_image: Image.Image) -> Dict[str, Any]:
    """
    Computes key visual metrics: aspect ratio, brightness, contrast,
    color temperature, sharpness score, and probable visual art style.
    """
    w, h = pil_image.size
    aspect_ratio_val = w / h
    if 0.95 <= aspect_ratio_val <= 1.05:
        aspect_ratio_str = "1:1"
    elif aspect_ratio_val < 0.65:
        aspect_ratio_str = "9:16"
    elif aspect_ratio_val > 1.55:
        aspect_ratio_str = "16:9"
    elif aspect_ratio_val < 0.9:
        aspect_ratio_str = "4:5"
    else:
        aspect_ratio_str = "4:3"

    # Analyze color temperature & lighting
    small = pil_image.convert('RGB').resize((120, 120))
    arr = np.array(small, dtype=np.float32)
    r_mean = float(arr[:, :, 0].mean())
    g_mean = float(arr[:, :, 1].mean())
    b_mean = float(arr[:, :, 2].mean())

    luminance_mean = float(0.2126 * r_mean + 0.7152 * g_mean + 0.0722 * b_mean)
    contrast_std = float(arr.std())

    # Warm vs Cool temperature
    # High red/yellow vs high blue
    if r_mean > b_mean + 15:
        temperature = "Tông Ấm (Warm Tone / Vàng & Cam)"
        temp_keyword = "warm golden tones, warm color grading"
    elif b_mean > r_mean + 15:
        temperature = "Tông Lạnh (Cool Tone / Xanh Lam & Cyan)"
        temp_keyword = "cool cinematic blue tones, cold atmosphere"
    else:
        temperature = "Trung Tính & Cân Bằng (Neutral / Balanced)"
        temp_keyword = "balanced natural color tones"

    # Lighting mood
    if luminance_mean < 70:
        lighting_mood = "Ánh sáng Tối / Moody Low-Key (Huyền bí, đêm tối)"
        lighting_keyword = "dark moody atmosphere, low-key lighting, dramatic shadows"
    elif luminance_mean > 175:
        lighting_mood = "Ánh sáng Rực Rỡ / High-Key (Trong trẻo, tươi sáng)"
        lighting_keyword = "bright natural daylight, high-key lighting, airy clean mood"
    else:
        lighting_mood = "Ánh sáng Cân Bằng / Studio (Độ tương phản tự nhiên)"
        lighting_keyword = "soft studio diffused lighting, natural ambient light"

    # Dominant colors
    dominant_colors = extract_dominant_colors(pil_image, 5)

    return {
        'width': w,
        'height': h,
        'aspect_ratio': aspect_ratio_str,
        'luminance_mean': round(luminance_mean, 1),
        'contrast_score': round(contrast_std, 1),
        'temperature': temperature,
        'temperature_keyword': temp_keyword,
        'lighting_mood': lighting_mood,
        'lighting_keyword': lighting_keyword,
        'dominant_colors': dominant_colors
    }


# ============================================================================
# 2. LOCAL KNOWLEDGE & BILINGUAL SEMANTIC VOCABULARY
# ============================================================================

VIETNAMESE_CONCEPT_MAP = {
    # Subjects & Characters
    r"cô gái|thiếu nữ|nữ sinh|phụ nữ": "beautiful young Vietnamese woman, elegant feminine features, delicate facial details",
    r"chàng trai|nam thanh niên|đàn ông": "handsome young man, charismatic expression, sharp jawline",
    r"áo dài trắng": "flowing traditional pure white silk Vietnamese Ao Dai, delicate embroidery, graceful posture",
    r"áo dài": "traditional elegant Vietnamese silk Ao Dai with intricate floral patterns",
    r"nón lá": "traditional Vietnamese conical hat (Non La)",
    r"mèo|chú mèo|mèo con": "cute fluffy cat, expressive detailed eyes, soft fur texture",
    r"chó|chú chó|cún": "adorable friendly dog, expressive eyes, detailed fur",
    r"rồng|chú rồng": "majestic mythical oriental dragon, glowing crystalline scales, ethereal aura",
    r"phi hành gia": "futuristic astronaut in high-tech illuminated spacesuit",
    r"siêu xe|ô tô|xe hơi": "aerodynamic modern luxury supercar, sleek glossy carbon fiber reflections",
    r"chiến binh|hiệp sĩ": "heroic armored warrior, ornate ceremonial armor, commanding presence",

    # Locations & Environment
    r"hồ gươm|tháp rùa": "Hoan Kiem Lake Hanoi, historic Turtle Tower in the background, tranquil weeping willows",
    r"phố cổ|hội an": "Hoi An ancient lantern town, yellow heritage colonial walls, glowing colorful silk lanterns",
    r"sa pa|sapa|ruộng bậc thang": "Sapa mountain highland, lush stepped green terraced rice fields, ethereal morning mist",
    r"hà nội": "atmospheric Hanoi old quarter, historic architecture, vibrant street life",
    r"sài gòn|tp hcm": "bustling vibrant modern Saigon skyline, illuminated skyscrapers, dynamic urban reflections",
    r"bãi biển|biển": "tropical pristine sandy beach, turquoise crystal-clear ocean waves, sunny horizon",
    r"rừng|rừng cây|khu rừng": "lush mystical ancient forest, towering mossy trees, sunbeams filtering through canopy",
    r"thành phố tương lai|cyberpunk": "futuristic cyberpunk metropolis, towering neon holograms, reflective wet asphalt streets",
    r"quán cà phê|quán cafe": "cozy aesthetic coffee shop, warm ambient light, wooden rustic interior",

    # Weather & Lighting
    r"mưa|trời mưa|mưa phùn": "gentle cinematic rain, glistening wet surfaces, dynamic water droplets, reflective puddles",
    r"sương mù|mù sương": "atmospheric morning fog, ethereal misty haze, dreamy soft mood",
    r"hoàng hôn|chiều tà": "golden hour sunset, vibrant fiery orange and purple twilight gradient, warm golden light",
    r"bình minh|buổi sáng": "fresh morning golden hour sunrise, crisp morning glow, soft ethereal lens flare",
    r"đêm|buổi tối|ban đêm": "dramatic night atmosphere, starry night sky, illuminated ambient lights",
    r"neon|phát sáng": "vibrant glowing neon lights, cyberpunk luminescence, saturated cyan and magenta glow",

    # Styles & Genres
    r"anime|hoạt hình nhật": "Makoto Shinkai anime aesthetic, vibrant detailed sky, emotional cinematography, masterpiece illustration",
    r"chân dung|portrait": "professional 85mm portrait photograph, shallow depth of field, creamy smooth bokeh, realistic skin texture",
    r"điện ảnh|cinematic": "cinematic 35mm movie still, anamorphic widescreen, dramatic cinematic lighting, color graded 8k",
    r"3d|pixar|disney": "cute Pixar Disney 3D character animation render, smooth clay shader, expressive eyes, soft studio lighting",
    r"sơn dầu|tranh vẽ": "intricate classical oil painting, textured impasto brushstrokes, rich classical chiaroscuro",
    r"retro|vintage|cổ điển": "vintage 1970s Kodak film photograph, subtle authentic film grain, nostalgic warm muted palette"
}

ART_STYLE_TEMPLATES = {
    "photorealistic": {
        "prefix": "award-winning photograph, ultra-realistic DSLR shot, natural skin texture, ",
        "suffix": ", 85mm f/1.4 lens, sharp focus, raw photo, studio lighting, hyper-detailed, 8k resolution",
        "negative": "painting, drawing, illustration, cartoon, anime, 3d render, plastic skin, oversaturated, blurry, bad anatomy",
        "style_id": "photorealistic"
    },
    "cinematic": {
        "prefix": "cinematic movie still, dramatic atmosphere, anamorphic 35mm lens, ",
        "suffix": ", dramatic chiaroscuro lighting, volumetric rim light, color graded masterpiece, 8k uhd, photorealistic",
        "negative": "cartoon, 3d render, blurry, low resolution, flat lighting, amateur, overexposed, oversaturated",
        "style_id": "cinematic"
    },
    "anime": {
        "prefix": "stunning anime masterpiece, Makoto Shinkai and CoMix Wave studio art style, ",
        "suffix": ", vibrant atmospheric lighting, breathtaking sky, intricate digital art, trending on Pixiv, highly detailed",
        "negative": "photorealistic, real life photo, western cartoon, low quality, deformed, bad proportions, blurry",
        "style_id": "anime"
    },
    "cyberpunk": {
        "prefix": "cyberpunk neon aesthetic, futuristic sci-fi atmosphere, ",
        "suffix": ", neon reflections on wet asphalt, cyan and magenta glow, high-tech details, volumetric fog, octane render 8k",
        "negative": "vintage, daylight, rustic, traditional, blurry, poor contrast, low detail",
        "style_id": "cyberpunk"
    },
    "pixar3d": {
        "prefix": "delightful 3D character animation render, modern Pixar and Disney animation studio style, ",
        "suffix": ", expressive eyes, smooth stylized textures, soft rim lighting, raytracing, unreal engine 5, octane render",
        "negative": "realistic human photo, terrifying, ugly, deformed, dark, gritty, sketch",
        "style_id": "pixar3d"
    },
    "fantasy": {
        "prefix": "epic fantasy digital painting, mythical ethereal ambiance, ",
        "suffix": ", concept art by Greg Rutkowski, intricate magical details, atmospheric god rays, ArtStation trending, masterpiece",
        "negative": "modern, contemporary, cars, ugly, low resolution, bad anatomy, flat colors",
        "style_id": "fantasy"
    },
    "vintage": {
        "prefix": "vintage 1970s authentic Kodak Portra 400 film photograph, nostalgic warmth, ",
        "suffix": ", muted retro color grading, subtle film grain, natural analog lens imperfections, timeless composition",
        "negative": "modern digital, ultra-sharp cgi, futuristic, oversaturated, plastic, artificial",
        "style_id": "vintage"
    }
}


# ============================================================================
# 3. KNOWLEDGE BASE FOR PHOTO & COLOR CONSULTING
# ============================================================================

QA_KNOWLEDGE_BASE = [
    {
        "keywords": ["cyberpunk", "neon", "phối màu cyberpunk"],
        "title": "Phối Màu & Ánh Sáng Cyberpunk",
        "content": (
            "**Bảng màu phong cách Cyberpunk (Neon Sci-Fi):**\n"
            "- **Màu chủ đạo (Primary Duotone):** Xanh Neon / Cyan (`#00F0FF`) phối hợp với Hồng Tím / Hot Magenta (`#FF007F`). Đây là cặp màu tương phản kinh điển tạo nên hiệu ứng thị giác khoa học viễn tưởng mạnh mẽ.\n"
            "- **Màu nền (Background):** Xanh đen thẫm (`#0B0E14`) hoặc đen carbon của mặt đường nhựa ướt mưa.\n"
            "- **Màu điểm nhấn (Accents):** Vàng điện quang tử (`#FFE600`) hoặc Xanh lục huỳnh quang (`#00FF66`).\n"
            "- **Mẹo viết Prompt:** Thêm các từ khóa như `cyberpunk neon lighting, wet asphalt reflections, volumetric cyan and magenta glow, dark rain-slicked city streets, 8k octane render`."
        )
    },
    {
        "keywords": ["chân dung", "portrait", "bokeh", "85mm", "khẩu độ", "chụp người"],
        "title": "Bí Quyết Nhiếp Ảnh Chân Dung & Xóa Phông Bokeh",
        "content": (
            "**Kỹ thuật chụp chân dung nghệ thuật & thiết lập thông số:**\n"
            "- **Tiêu cự lý tưởng:** 85mm f/1.4 hoặc 50mm f/1.8. Tiêu cự 85mm giúp khuôn mặt giữ đúng tỷ lệ nhân trắc học không bị biến dạng góc rộng và tạo hiệu ứng xóa phông mượt mà (creamy bokeh).\n"
            "- **Bố cục:** Quy tắc 1/3 (đặt đôi mắt ở đường phân chia 1/3 trên cùng), bắt sáng điểm (catchlight) lung linh trong mắt nhân vật.\n"
            "- **Ánh sáng:** Ánh sáng Rembrandt (tạo hình tam giác sáng trên gò má bên tối) hoặc Ánh sáng Ven (Rim lighting) giúp tóc tách biệt khỏi hậu cảnh.\n"
            "- **Mẹo viết Prompt:** Thêm `professional 85mm portrait photograph, f/1.4 aperture, creamy bokeh background, catchlight in eyes, natural skin texture, studio softbox lighting`."
        )
    },
    {
        "keywords": ["anime", "makoto shinkai", "màu anime", "hoạt hình nhật"],
        "title": "Phong Cách Nghệ Thuật Makoto Shinkai & Anime Masterpiece",
        "content": (
            "**Đặc trưng phong cách Makoto Shinkai (Your Name / Weathering With You):**\n"
            "- **Bầu trời & Mây:** Bầu trời chuyển sắc mượt mà từ xanh ngọc bích sang lam biếc, mây tích (cumulonimbus) nhiều lớp trắng muốt viền sáng.\n"
            "- **Ánh sáng:** Những vệt nắng xiên (God rays / Komorebi) xuyên qua lá cây hoặc kẽ mây, vệt phản chiếu lấp lánh trên mặt nước và kim loại.\n"
            "- **Bảng màu:** Độ bão hòa cao nhưng hài hòa, nhấn mạnh xanh lam (`#1E90FF`), tím hoàng hôn (`#9370DB`) và vàng cam ấm áp (`#FF8C00`).\n"
            "- **Mẹo viết Prompt:** Thêm `Makoto Shinkai aesthetic, atmospheric anime masterpiece, dramatic sky with volumetric clouds, luminous lens flare, trending on Pixiv`."
        )
    },
    {
        "keywords": ["làm nét", "phục hồi", "ảnh mờ", "cũ", "clarity", "vỡ hạt"],
        "title": "Mẹo Phục Hồi Ảnh Cũ & Làm Nét Tối Đa Trên Clarity AI",
        "content": (
            "**Hướng dẫn tối ưu chất lượng phục hồi ảnh trên Clarity AI Studio:**\n"
            "1. **Ảnh Chân Dung Người:** Chọn mô hình `⚡ Siêu Chân Dung (Real-ESRGAN + GFPGAN)`. GFPGAN sẽ sử dụng mạng neural để tái tạo lại hoàn toàn từng sợi lông mi, con ngươi, khóe môi và kết cấu da tự nhiên.\n"
            "2. **Ảnh Phong Cảnh, Xe Cộ, Văn Bản:** Chọn `Real-ESRGAN Đa Dụng (x4plus)` để giữ nguyên hình khối vật thể, khử nhiễu hạt ISO mà không biến dạng chữ.\n"
            "3. **Ảnh Hoạt Hình / Anime / Tranh Vẽ 2D / Sticker:** Chọn `Siêu Nét Vector & Nét Vẽ 2D (Anime Cel Art)` để tái tạo đường nét đen đậm sắc sảo như nét vẽ vector mực đen, giữ mảng màu phẳng và da mịn màng tuyệt đối không nhiễu hạt, hỗ trợ giữ nguyên 1X gốc.\n"
            "4. **Đổi tỷ lệ:** Dùng tính năng `Đổi Tỷ Lệ Ảnh` với chế độ `AI Sinh Không Gian (LaMa)` để mở rộng khung hình sang 16:9 hoặc 9:16 mà không bị cắt xén ảnh gốc!"
        )
    },
    {
        "keywords": ["màu pastel", "bảng màu mùa thu", "phối màu ấm", "autumn"],
        "title": "Bảng Màu Pastel & Mùa Thu Ấm Áp",
        "content": (
            "**Bảng màu Mùa Thu Ấm Áp (Warm Autumn Palette):**\n"
            "- **Cam Cháy / Terracotta:** `#E2725B` - Gợi cảm giác lá phong rụng và gạch nung cổ kính.\n"
            "- **Vàng Mù Tạt / Mustard Ochre:** `#D4A017` - Nắng vàng chiều thu ấm áp.\n"
            "- **Nâu Cacao / Warm Chestnut:** `#7B3F00` - Thân cây gỗ mộc mạc và mặt đất.\n"
            "- **Xanh Rêu Ô Liu / Olive Moss:** `#556B2F` - Màu sắc cân bằng tự nhiên.\n"
            "- **Mẹo phối màu:** Đây là cách phối màu tương đồng (Analogous Palette) nằm cạnh nhau trên vòng tròn màu, mang lại cảm giác bình yên, hoài niệm và ấm cúng."
        )
    }
]


# ============================================================================
# 4. MAIN LLM & VISION ENGINE CLASS
# ============================================================================

class LLMEngine:
    def __init__(self, device: Optional[str] = None):
        if device:
            self.device = device
        elif HAS_TORCH and torch is not None and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Provider configurations
        self.active_provider = "built_in"  # 'built_in', 'ollama', 'lm_studio', 'local_hf'
        self.ollama_host = "http://127.0.0.1:11434"
        self.ollama_model = "llama3.2"
        self.lm_studio_host = "http://127.0.0.1:1234/v1"
        self.lm_studio_model = "local-model"

        # Cached in-memory models (lazy loaded)
        self.hf_model = None
        self.hf_tokenizer = None
        self.blip_processor = None
        self.blip_model = None

        print(f"[LLMEngine] Initialized engine with default provider '{self.active_provider}' on device: {self.device}")

    # ------------------------------------------------------------------------
    # Provider Diagnostics & Health Checks
    # ------------------------------------------------------------------------

    def get_providers_status(self) -> Dict[str, Any]:
        """Scans and reports the real-time status of all local AI providers."""
        ollama_status = {"available": False, "models": [], "host": self.ollama_host}
        try:
            r = requests.get(f"{self.ollama_host}/api/tags", timeout=1.2)
            if r.status_code == 200:
                data = r.json()
                models = [m.get("name") for m in data.get("models", [])]
                ollama_status = {
                    "available": True,
                    "models": models,
                    "host": self.ollama_host,
                    "count": len(models)
                }
        except Exception:
            pass

        lm_studio_status = {"available": False, "models": [], "host": self.lm_studio_host}
        try:
            r = requests.get(f"{self.lm_studio_host}/models", timeout=1.2)
            if r.status_code == 200:
                data = r.json()
                models = [m.get("id") for m in data.get("data", [])]
                lm_studio_status = {
                    "available": True,
                    "models": models,
                    "host": self.lm_studio_host,
                    "count": len(models)
                }
        except Exception:
            pass

        return {
            "active_provider": self.active_provider,
            "device": self.device,
            "providers": {
                "built_in": {
                    "name": "⚡ Clarity Local Engine (Tích hợp sẵn - Siêu Tốc)",
                    "available": True,
                    "type": "native_hybrid",
                    "badge": "Sẵn sàng 100% Free",
                    "desc": "Chạy trực tiếp trong Python với từ điển chuyên gia và phân tích Computer Vision, phản hồi tức thì 0.05s."
                },
                "local_qwen": {
                    "name": "🤖 Qwen 2.5 Instruct Local GPU (RTX 3060)",
                    "available": True,
                    "type": "neural_llm",
                    "badge": "CUDA RTX 3060",
                    "desc": "Mô hình ngôn ngữ lớn Qwen 2.5 Instruct chạy trực tiếp trên GPU máy tính, hiểu tiếng Việt và tối ưu prompt sáng tạo xuất sắc."
                },
                "ollama": {
                    "name": "🦙 Ollama Local Server",
                    "available": ollama_status["available"],
                    "models": ollama_status["models"],
                    "current_model": self.ollama_model,
                    "host": self.ollama_host,
                    "badge": f"{len(ollama_status['models'])} model" if ollama_status["available"] else "Chưa bật",
                    "desc": "Kết nối máy chủ Ollama cục bộ (Llama 3.2, Qwen 2.5, Llava Vision, Mistral...)"
                },
                "lm_studio": {
                    "name": "🖥️ LM Studio / Local OpenAI API",
                    "available": lm_studio_status["available"],
                    "models": lm_studio_status["models"],
                    "current_model": self.lm_studio_model,
                    "host": self.lm_studio_host,
                    "badge": f"{len(lm_studio_status['models'])} model" if lm_studio_status["available"] else "Chưa bật",
                    "desc": "Cổng kết nối chuẩn OpenAI tương thích LM Studio, Text-Gen-WebUI, vLLM chạy tại localhost:1234."
                }
            }
        }

    def set_provider_config(self, provider: str, config: Optional[Dict[str, Any]] = None):
        """Update provider and associated parameters."""
        valid_providers = ['built_in', 'local_qwen', 'ollama', 'lm_studio']
        if provider in valid_providers:
            self.active_provider = provider
        if config:
            if 'ollama_host' in config:
                self.ollama_host = config['ollama_host']
            if 'ollama_model' in config:
                self.ollama_model = config['ollama_model']
            if 'lm_studio_host' in config:
                self.lm_studio_host = config['lm_studio_host']
            if 'lm_studio_model' in config:
                self.lm_studio_model = config['lm_studio_model']
        return {"status": "success", "active_provider": self.active_provider}

    # ------------------------------------------------------------------------
    # 5. Core Feature 1: Natural Language Prompt Optimizer
    # ------------------------------------------------------------------------

    def optimize_prompt(
        self,
        user_input: str,
        style_preset: Optional[str] = None,
        aspect_ratio: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Takes natural language (Vietnamese or English) and expands it
        into a professional, production-ready Stable Diffusion prompt.
        """
        user_input_clean = user_input.strip()
        if not user_input_clean:
            return {
                "status": "error",
                "message": "Vui lòng nhập ý tưởng hoặc câu mô tả ảnh."
            }

        start_time = time.time()

        # Step 1: Detect style or respect user selection
        detected_style = style_preset or "cinematic"
        input_lower = user_input_clean.lower()

        if not style_preset or style_preset == "none":
            if any(k in input_lower for k in ["anime", "manga", "shinkai", "hoạt hình nhật", "2d"]):
                detected_style = "anime"
            elif any(k in input_lower for k in ["cyberpunk", "neon", "tương lai", "sci-fi"]):
                detected_style = "cyberpunk"
            elif any(k in input_lower for k in ["chân dung", "portrait", "chụp ảnh", "máy ảnh", "người thật"]):
                detected_style = "photorealistic"
            elif any(k in input_lower for k in ["3d", "pixar", "disney", "hoạt hình 3d"]):
                detected_style = "pixar3d"
            elif any(k in input_lower for k in ["thần thoại", "fantasy", "phép thuật", "rồng"]):
                detected_style = "fantasy"
            elif any(k in input_lower for k in ["vintage", "retro", "phim cổ điển", "cổ"]):
                detected_style = "vintage"
            else:
                detected_style = "cinematic"
        # Check if Local Qwen or external Ollama or LM Studio is active
        llm_expanded = None
        if self.active_provider == "local_qwen":
            llm_expanded = self._call_local_qwen_prompt_optimizer(user_input_clean, detected_style)
        elif self.active_provider == "ollama":
            llm_expanded = self._call_ollama_prompt_optimizer(user_input_clean, detected_style)
        elif self.active_provider == "lm_studio":
            llm_expanded = self._call_openai_prompt_optimizer(user_input_clean, detected_style)

        if llm_expanded:
            duration = round(time.time() - start_time, 2)
            llm_expanded["duration_sec"] = duration
            llm_expanded["provider_used"] = self.active_provider
            return {"status": "success", "data": llm_expanded}

        # Native High-Speed Semantic Expansion
        expanded = self._semantic_expand_prompt(user_input_clean, detected_style, aspect_ratio)
        duration = round(time.time() - start_time, 2)
        expanded["duration_sec"] = duration
        expanded["provider_used"] = "Clarity Local Hybrid"

        return {"status": "success", "data": expanded}

    def _semantic_expand_prompt(self, text: str, style_name: str, ratio: Optional[str]) -> Dict[str, Any]:
        """High-speed offline semantic expander with complete context preservation."""
        from generator import smart_translate_prompt, detect_style_from_prompt, STYLE_PRESETS, DEFAULT_NEGATIVE_PROMPT

        style_info = STYLE_PRESETS.get(style_name, STYLE_PRESETS.get("cinematic", STYLE_PRESETS["auto"]))
        text_lower = text.lower()

        # 1. Translate full sentence first so zero words are discarded
        full_translated_sentence = smart_translate_prompt(text).strip()

        # 2. Enrich with matched domain concepts if any
        applied_rules = []
        enriched_tags = []
        for pattern, english_repr in VIETNAMESE_CONCEPT_MAP.items():
            if re.search(pattern, text_lower):
                applied_rules.append(pattern.split('|')[0])
                if english_repr.lower() not in full_translated_sentence.lower():
                    enriched_tags.append(english_repr)

        core_concept = full_translated_sentence
        if enriched_tags:
            core_concept = f"{full_translated_sentence}, {', '.join(enriched_tags[:2])}"

        # Assemble full prompt with aesthetic layers (Subject Always First)
        suffix = style_info.get("prompt_suffix", "")
        full_positive = f"{core_concept}{suffix}"

        # Clean multiple commas or redundant spaces
        full_positive = re.sub(r',\s*,+', ', ', full_positive).strip(', ')
        negative_prompt = style_info.get("negative", DEFAULT_NEGATIVE_PROMPT)

        # Determine recommended aspect ratio if not passed
        recommended_ratio = ratio or "1:1"
        if any(k in text_lower for k in ["chân dung", "portrait", "áo dài", "đứng"]):
            recommended_ratio = "9:16"
        elif any(k in text_lower for k in ["phong cảnh", "siêu xe", "thành phố", "điện ảnh", "bãi biển"]):
            recommended_ratio = "16:9"

        return {
            "original_input": text,
            "optimized_prompt": full_positive,
            "negative_prompt": negative_prompt,
            "style_preset": style_name,
            "aspect_ratio": recommended_ratio,
            "breakdown": {
                "core_subject": full_translated_sentence if full_translated_sentence else text,
                "style_name": style_name.capitalize(),
                "matched_concepts": applied_rules[:4],
                "enhancement_tags": "Masterpiece, 8K UHD, Volumetric Lighting, Studio Quality"
            },
            "explanation_vi": (
                f"Đã phân tích ý tưởng của bạn và nâng cấp thành Prompt chuẩn Stable Diffusion: "
                f"Tối ưu phong cách {style_name.capitalize()}, bổ sung chi tiết ánh sáng, độ phân giải 8K và cấu trúc ống kính."
            )
        }

    # ------------------------------------------------------------------------
    # 6. Core Feature 2: Image-to-Prompt & Visual Deconstruction
    # ------------------------------------------------------------------------

    def image_to_prompt(self, image_path_or_pil, prompt_hint: str = "") -> Dict[str, Any]:
        """
        Reverse-engineers an image: extracts color palette, lighting, composition,
        and generates a high-quality prompt replicating its essence.
        """
        start_time = time.time()

        if isinstance(image_path_or_pil, str):
            if not os.path.exists(image_path_or_pil):
                return {"status": "error", "message": "Không tìm thấy tệp ảnh"}
            img = Image.open(image_path_or_pil).convert('RGB')
        elif isinstance(image_path_or_pil, Image.Image):
            img = image_path_or_pil.convert('RGB')
        else:
            return {"status": "error", "message": "Định dạng ảnh không hợp lệ"}

        # Step 1: Deep Computer Vision & Color Analysis
        metrics = analyze_image_metrics(img)

        # Step 2: Try Vision Model from Ollama if available and active
        vision_caption = None
        if self.active_provider == "ollama":
            vision_caption = self._call_ollama_vision(img, prompt_hint)
        elif self.active_provider == "local_blip" and HAS_TORCH:
            try:
                vision_caption = self._call_local_blip(img)
            except Exception:
                pass

        # Step 4: Classify style from visual metrics
        aspect_ratio = metrics['aspect_ratio']
        temp_keyword = metrics['temperature_keyword']
        lighting_keyword = metrics['lighting_keyword']
        colors = metrics['dominant_colors']

        # Determine dominant color names for prompt
        top_color_names = [c['name'].split('(')[-1].replace(')', '').strip() for c in colors[:3]]
        palette_tag = ", ".join(top_color_names)

        # Build synthesized prompt
        if vision_caption:
            base_subject = vision_caption
        else:
            # Fallback heuristic subject description
            base_subject = "a compelling detailed visual composition, high aesthetic appeal"

        inferred_style = "cinematic"
        if "anime" in base_subject.lower() or "illustration" in base_subject.lower():
            inferred_style = "anime"
        elif "cyberpunk" in base_subject.lower() or "neon" in base_subject.lower():
            inferred_style = "cyberpunk"
        elif "portrait" in base_subject.lower() or "face" in base_subject.lower():
            inferred_style = "photorealistic"

        style_cfg = ART_STYLE_TEMPLATES.get(inferred_style, ART_STYLE_TEMPLATES["cinematic"])

        # Construct final rich prompt
        final_prompt = (
            f"{style_cfg['prefix']}{base_subject}, "
            f"{palette_tag} color palette, {temp_keyword}, {lighting_keyword}"
            f"{style_cfg['suffix']}"
        )
        final_prompt = re.sub(r',\s*,+', ', ', final_prompt).strip(', ')

        duration = round(time.time() - start_time, 2)

        return {
            "status": "success",
            "data": {
                "generated_prompt": final_prompt,
                "negative_prompt": style_cfg["negative"],
                "style_preset": inferred_style,
                "aspect_ratio": aspect_ratio,
                "visual_metrics": {
                    "width": metrics["width"],
                    "height": metrics["height"],
                    "aspect_ratio": aspect_ratio,
                    "temperature": metrics["temperature"],
                    "lighting_mood": metrics["lighting_mood"],
                    "dominant_colors": colors
                },
                "vision_caption": vision_caption or "Phân tích đặc trưng quang học & bảng màu AI",
                "duration_sec": duration,
                "provider_used": self.active_provider if vision_caption else "Clarity Computer Vision Engine"
            }
        }

    # ------------------------------------------------------------------------
    # 7. Core Feature 3: Photography & Color Chatbot Consultant
    # ------------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        image_path: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Conversational assistant answering photography, color theory, prompt craft,
        and project questions. Automatically handles uploaded images.
        """
        msg_clean = user_message.strip()
        has_image = bool(image_path and os.path.exists(image_path))

        # If user uploaded an image and message is empty or asking to convert to prompt
        if has_image and (not msg_clean or any(k in msg_clean.lower() for k in ["biến ảnh thành prompt", "prompt", "tạo prompt", "mô tả ảnh"])):
            res = self.image_to_prompt(image_path, prompt_hint=msg_clean)
            if res.get("status") == "success":
                data = res["data"]
                colors_html = "".join([
                    f"<span class='color-swatch-chip' style='background:{c['hex']}; color:{'#000' if c['is_light'] else '#fff'}'>"
                    f"{c['hex']} ({c['percentage']}%)</span> "
                    for c in data['visual_metrics']['dominant_colors']
                ])
                bot_reply = (
                    f"### 🎨 Đã Phân Tích & Chuyển Đổi Ảnh Thành Prompt Thành Công!\n\n"
                    f"**📸 Chi tiết thị giác nhận diện:**\n"
                    f"- **Bố cục & Tỷ lệ:** `{data['aspect_ratio']}` ({data['visual_metrics']['width']} x {data['visual_metrics']['height']} px)\n"
                    f"- **Nhiệt độ màu:** {data['visual_metrics']['temperature']}\n"
                    f"- **Ánh sáng:** {data['visual_metrics']['lighting_mood']}\n"
                    f"- **Bảng màu chủ đạo:**\n{colors_html}\n\n"
                    f"**✨ Prompt Tối Ưu Cho Stable Diffusion / Midjourney:**\n"
                    f"```text\n{data['generated_prompt']}\n```\n\n"
                    f"**🛡️ Negative Prompt Khuyên Dùng:**\n"
                    f"```text\n{data['negative_prompt']}\n```\n\n"
                    f"> *Mẹo:* Bạn có thể bấm nút **'Sử dụng trong Tạo Ảnh AI'** bên dưới để chuyển thẳng prompt này sang studio tạo ảnh!"
                )
                return {
                    "status": "success",
                    "reply": bot_reply,
                    "prompt_data": data,
                    "has_image_result": True
                }

        # Check reverse image search request
        if has_image and any(k in msg_clean.lower() for k in ["tìm ảnh tương tự", "ảnh tương tự", "tìm giống", "tương tự", "similar", "reverse"]):
            from visual_search import visual_search_engine
            results = visual_search_engine.search_by_image(image_path, top_k=8, min_similarity=0.20)
            count = len(results)
            if count > 0:
                bot_reply = (
                    f"### 🔍 Đã Tìm Thấy {count} Hình Ảnh Tương Tự Bằng Mô Hình CLIP AI!\n\n"
                    f"Dựa trên các đặc trưng thị giác (màu sắc, bố cục, chủ thể) của ảnh bạn cung cấp, "
                    f"dưới đây là các ảnh có độ tương đồng cao nhất trong thư viện:"
                )
            else:
                bot_reply = "⚠️ Chưa tìm thấy ảnh nào có độ tương đồng đủ cao với ảnh bạn cung cấp trong thư viện."
            return {
                "status": "success",
                "reply": bot_reply,
                "search_results": results,
                "is_search_result": True
            }

        # Check text-to-image search request in chat
        search_triggers = ["tìm ảnh ", "tìm hình ", "tìm kiếm ảnh ", "tìm kiếm hình ", "search image ", "search ", "tìm các ảnh", "tìm những ảnh", "tìm bức ảnh"]
        is_search_query = any(msg_clean.lower().startswith(p) for p in search_triggers) or any(k in msg_clean.lower() for k in ["tìm ảnh về", "tìm hình về", "tìm kiếm bức ảnh"])
        if is_search_query:
            search_q = msg_clean
            for p in ["tìm kiếm hình ảnh về", "tìm kiếm ảnh về", "tìm hình ảnh về", "tìm kiếm hình ảnh", "tìm kiếm ảnh", "tìm các ảnh", "tìm những ảnh", "tìm bức ảnh", "tìm ảnh về", "tìm hình về", "tìm ảnh ", "tìm hình ", "search image ", "search ", "tìm "]:
                if search_q.lower().startswith(p):
                    search_q = search_q[len(p):].strip()
                    break
            if search_q:
                from visual_search import visual_search_engine
                results = visual_search_engine.search_by_text(search_q, top_k=8, min_similarity=0.15)
                count = len(results)
                if count > 0:
                    bot_reply = (
                        f"### 🔎 Đã Tìm Thấy {count} Hình Ảnh Phù Hợp Với Từ Khóa: *\"{search_q}\"*!\n\n"
                        f"Mô hình **OpenAI CLIP ViT-B/32** đã đối chiếu ngữ nghĩa và trích xuất các hình ảnh sát nhất trong kho lưu trữ:"
                    )
                else:
                    bot_reply = f"⚠️ Không tìm thấy ảnh nào khớp với từ khóa *\"{search_q}\"*. Bạn có thể thử từ khóa tương tự hoặc tải thêm ảnh vào hệ thống!"
                return {
                    "status": "success",
                    "reply": bot_reply,
                    "search_results": results,
                    "is_search_result": True,
                    "query": search_q
                }
        if self.active_provider == "local_qwen":
            reply = self._call_local_qwen(
                msg_clean,
                system_prompt="Bạn là chuyên gia AI cố vấn Nhiếp ảnh, Màu sắc và Prompt Engineering của Clarity AI Studio. Hãy trả lời chi tiết, chuyên nghiệp và thân thiện bằng tiếng Việt với định dạng Markdown rõ ràng."
            )
            if reply:
                return {"status": "success", "reply": reply}
        elif self.active_provider == "ollama":
            reply = self._call_ollama_chat(msg_clean, image_path, chat_history)
            if reply:
                return {"status": "success", "reply": reply}
        elif self.active_provider == "lm_studio":
            reply = self._call_openai_chat(msg_clean, chat_history)
            if reply:
                return {"status": "success", "reply": reply}

        # Local Knowledge Base Matching
        msg_lower = msg_clean.lower()
        matched_content = []
        for item in QA_KNOWLEDGE_BASE:
            if any(k in msg_lower for k in item["keywords"]):
                matched_content.append(item["content"])

        if matched_content:
            reply = "\n\n---\n\n".join(matched_content)
            return {"status": "success", "reply": reply}

        # General helpful fallback answering about photography, colors, or prompt craft
        fallback_reply = (
            f"Chào bạn! Tôi là **Trợ lý AI Nhiếp Ảnh & Màu Sắc** của Clarity AI Studio. 🌟\n\n"
            f"Tôi có thể hỗ trợ bạn:\n"
            f"1. **Tối ưu Prompt từ ngôn ngữ tự nhiên:** Chỉ cần mô tả ý tưởng (ví dụ: *'chú rồng ngọc bích phát sáng trong hang động'*), tôi sẽ viết prompt chuẩn 8K kèm negative prompt.\n"
            f"2. **Biến ảnh thành Prompt (Image-to-Prompt):** Nhấn biểu tượng đính kèm ảnh hoặc kéo thả ảnh vào đây, tôi sẽ phân tích bảng màu HEX, ánh sáng và bóc tách prompt để tái tạo phong cách ảnh.\n"
            f"3. **Tư vấn màu sắc & Nhiếp ảnh:** Phối màu Cyberpunk, Vintage, bảng màu Pastel, kỹ thuật xóa phông chân dung 85mm f/1.4, bố cục điện ảnh.\n"
            f"4. **Hướng dẫn dự án Clarity AI:** Cách siêu làm nét ảnh chân dung (Real-ESRGAN + GFPGAN), đổi tỷ lệ ảnh với AI Fill, hoặc nén ảnh về mức MB mục tiêu.\n\n"
            f"Bạn muốn khám phá nội dung nào trước?"
        )
        return {"status": "success", "reply": fallback_reply}

    # ------------------------------------------------------------------------
    # 8. Local Neural LLM (Qwen 2.5 Instruct on PyTorch CUDA)
    # ------------------------------------------------------------------------

    def _call_local_qwen(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Runs Qwen 2.5 Instruct directly on local GPU via PyTorch transformers."""
        if not HAS_TORCH or torch is None:
            return None
        try:
            if self.hf_model is None or self.hf_tokenizer is None:
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
                model_id = "Qwen/Qwen2.5-0.5B-Instruct"
                print(f"[LLMEngine] Loading local Qwen model ({model_id}) on {self.device}...")
                self.hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="cuda" if self.device == "cuda" else None
                )
                if self.hf_model is not None:
                    self.hf_model.eval()

            if self.hf_tokenizer is None or self.hf_model is None:
                return None

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            text = self.hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.hf_tokenizer([text], return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.hf_model.generate(**inputs, max_new_tokens=300)
                generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                response = self.hf_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return response.strip()
        except Exception as e:
            print(f"[LLMEngine] Local Qwen error: {e}")
            return None

    def _call_local_qwen_prompt_optimizer(self, user_text: str, style: str) -> Optional[Dict[str, Any]]:
        """Uses local Qwen 2.5 to optimize Stable Diffusion prompt."""
        system_prompt = (
            "You are an expert AI prompt engineer for Stable Diffusion and Midjourney. "
            "Convert the user's concept into a highly detailed English prompt. "
            "Return valid JSON only with keys: 'optimized_prompt', 'negative_prompt', 'style_preset', 'explanation_vi'."
        )
        user_msg = f"Concept: {user_text}\nDesired Style: {style}"
        raw_output = self._call_local_qwen(user_msg, system_prompt=system_prompt)
        if raw_output:
            try:
                m = re.search(r'\{.*\}', raw_output, re.DOTALL)
                if m:
                    parsed = json.loads(m.group(0))
                    parsed["original_input"] = user_text
                    parsed["aspect_ratio"] = "1:1"
                    return parsed
            except Exception:
                pass
            # If not JSON, use the raw output as the prompt
            style_info = ART_STYLE_TEMPLATES.get(style, ART_STYLE_TEMPLATES["cinematic"])
            return {
                "original_input": user_text,
                "optimized_prompt": raw_output.replace('\n', ', '),
                "negative_prompt": style_info["negative"],
                "style_preset": style,
                "aspect_ratio": "1:1",
                "explanation_vi": "Đã tối ưu hóa thông qua mô hình Neural LLM Qwen 2.5 Instruct (Local GPU)."
            }
        return None

    # ------------------------------------------------------------------------
    # 9. External API Connectors (Ollama, LM Studio, BLIP)
    # ------------------------------------------------------------------------

    def _call_ollama_prompt_optimizer(self, user_text: str, style: str) -> Optional[Dict[str, Any]]:
        """Invokes Ollama to generate a high quality prompt."""
        system_prompt = (
            "You are an expert AI prompt engineer for Stable Diffusion and Midjourney. "
            "Convert the user's natural language concept into a professional, highly-detailed English prompt. "
            "Return valid JSON only, with keys: 'optimized_prompt', 'negative_prompt', 'style_preset', 'explanation_vi'."
        )
        user_msg = f"Concept: {user_text}\nDesired Style: {style}"
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            "stream": False,
            "format": "json"
        }
        try:
            r = requests.post(f"{self.ollama_host}/api/chat", json=payload, timeout=8.0)
            if r.status_code == 200:
                content = r.json().get("message", {}).get("content", "")
                parsed = json.loads(content)
                parsed["original_input"] = user_text
                parsed["aspect_ratio"] = "1:1"
                return parsed
        except Exception as e:
            print(f"[LLMEngine] Ollama prompt optimizer error: {e}")
        return None

    def _call_openai_prompt_optimizer(self, user_text: str, style: str) -> Optional[Dict[str, Any]]:
        """Invokes LM Studio or local OpenAI compatible server."""
        system_prompt = (
            "You are an expert AI prompt engineer for Stable Diffusion and Midjourney. "
            "Convert the user's natural language concept into a professional, highly-detailed English prompt. "
            "Return valid JSON only, with keys: 'optimized_prompt', 'negative_prompt', 'style_preset', 'explanation_vi'."
        )
        payload = {
            "model": self.lm_studio_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Concept: {user_text}\nStyle: {style}"}
            ],
            "temperature": 0.7
        }
        try:
            r = requests.post(f"{self.lm_studio_host}/chat/completions", json=payload, timeout=8.0)
            if r.status_code == 200:
                content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                # Find JSON block
                m = re.search(r'\{.*\}', content, re.DOTALL)
                if m:
                    parsed = json.loads(m.group(0))
                    parsed["original_input"] = user_text
                    parsed["aspect_ratio"] = "1:1"
                    return parsed
        except Exception as e:
            print(f"[LLMEngine] LM Studio prompt optimizer error: {e}")
        return None

    def _call_ollama_vision(self, pil_image: Image.Image, prompt_hint: str) -> Optional[str]:
        """Calls vision-capable models in Ollama (e.g. llava, llama3.2-vision)."""
        buffered = BytesIO()
        pil_image.resize((512, 512)).save(buffered, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = (
            "Describe the visual subject, setting, colors, and lighting of this image in detail "
            "suitable for a text-to-image prompt. Be concise and descriptive."
        )
        if prompt_hint:
            prompt += f" Focus especially on: {prompt_hint}"

        payload = {
            "model": self.ollama_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64]
                }
            ],
            "stream": False
        }
        try:
            r = requests.post(f"{self.ollama_host}/api/chat", json=payload, timeout=12.0)
            if r.status_code == 200:
                return r.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"[LLMEngine] Ollama vision error: {e}")
        return None

    def _call_ollama_chat(
        self,
        message: str,
        image_path: Optional[str],
        history: Optional[List[Dict[str, str]]]
    ) -> Optional[str]:
        """Conversational chat via Ollama."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Clarity AI Photography & Color Consultant assistant. "
                    "You give friendly, insightful, and expert advice in Vietnamese about photography styles, "
                    "lighting, color harmony, and AI prompt engineering for Stable Diffusion."
                )
            }
        ]
        if history:
            for h in history[-4:]:
                messages.append(h)

        user_content: Dict[str, Any] = {"role": "user", "content": message}
        if image_path and os.path.exists(image_path):
            try:
                with open(image_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                    user_content["images"] = [img_b64]
            except Exception:
                pass

        messages.append(user_content)

        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False
        }
        try:
            r = requests.post(f"{self.ollama_host}/api/chat", json=payload, timeout=12.0)
            if r.status_code == 200:
                return r.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"[LLMEngine] Ollama chat error: {e}")
        return None

    def _call_openai_chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]]
    ) -> Optional[str]:
        """Conversational chat via LM Studio / OpenAI API."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Clarity AI Photography & Color Consultant assistant. "
                    "Answer thoughtfully in Vietnamese regarding art styles, colors, lighting, and AI generation."
                )
            }
        ]
        if history:
            for h in history[-4:]:
                messages.append(h)
        messages.append({"role": "user", "content": message})

        payload = {
            "model": self.lm_studio_model,
            "messages": messages,
            "temperature": 0.7
        }
        try:
            r = requests.post(f"{self.lm_studio_host}/chat/completions", json=payload, timeout=10.0)
            if r.status_code == 200:
                return r.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"[LLMEngine] LM Studio chat error: {e}")
        return None

    def _call_local_blip(self, pil_image: Image.Image) -> Optional[str]:
        """Lazy loads BLIP captioner on-demand if weights exist and safe."""
        if not HAS_TORCH or torch is None:
            return None

        try:
            if self.blip_model is None or self.blip_processor is None:
                from transformers import BlipProcessor, BlipForConditionalGeneration  # type: ignore
                model_id = "Salesforce/blip-image-captioning-base"
                self.blip_processor = BlipProcessor.from_pretrained(model_id, use_fast=False)
                blip_inst: Any = BlipForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self.blip_model = blip_inst.to(self.device)
                if self.blip_model is not None:
                    self.blip_model.eval()

            if self.blip_processor is None or self.blip_model is None:
                return None

            inputs = self.blip_processor(pil_image, return_tensors="pt")  # type: ignore
            inputs = inputs.to(self.device)
            if self.device == "cuda":
                inputs = {k: v.to(torch.float16) if hasattr(v, 'dtype') and v.dtype == torch.float32 else v for k, v in inputs.items()}

            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_new_tokens=50)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)  # type: ignore
                return caption.strip()
        except Exception:
            return None
