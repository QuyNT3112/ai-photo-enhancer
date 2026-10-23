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
# 3. VERIFIED KNOWLEDGE BASE FOR PHOTO & COLOR CONSULTING
# ============================================================================

KNOWLEDGE_DEFINITIONS = [
    {
        "id": "pastel",
        "keywords": [
            "màu pastel", "mau pastel", "pastel là gì", "định nghĩa màu pastel",
            "dinh nghia mau pastel", "màu phấn", "tone pastel", "pastel", "gam màu pastel",
            "bảng màu pastel", "phong cách pastel"
        ],
        "title": "Định Nghĩa Chính Xác Về Màu Pastel",
        "content": (
            "### 🎨 Định Nghĩa Chính Xác Về Màu Pastel (Pastel Colors)\n\n"
            "**1. Màu Pastel là gì?**\n"
            "- **Màu Pastel** (còn gọi là *\"phấn màu\"* hay *\"màu nhạt dịu\"*) là nhóm màu sắc có **độ bão hòa rất thấp (Low Saturation: 15% – 35%)** và **độ sáng rất cao (High Luminance / Value: 85% – 98%)**.\n"
            "- Về nguyên lý hội họa và quang học, màu pastel được tạo thành bằng cách **pha trộn một lượng lớn màu trắng vào màu nguyên bản (Pure Hue)**. Quá trình này trong lý thuyết màu sắc được gọi là tạo sắc thái nhạt (*Tinting*).\n\n"
            "**2. Đặc tính thị giác & Khác biệt:**\n"
            "- Trái ngược hoàn toàn với màu rực rỡ (*Vibrant / Highly Saturated*) hay màu neon chói lọi, màu pastel đem lại cảm giác **thanh khiết, êm dịu, ngọt ngào, tinh tế và thư thái** cho mắt nhìn.\n"
            "- Không gây chói lóa hay mỏi mắt, thường gắn liền với sự nhẹ nhàng, lãng mạn và tuổi thanh xuân.\n\n"
            "**3. Các tông màu Pastel kinh điển & Mã HEX chuẩn:**\n"
            "- 🌸 **Baby Pink (Hồng phấn nhạt):** `#FFD1DC` / `#FFB6C1`\n"
            "- 🌿 **Mint Green (Xanh bạc hà dịu):** `#C1E1C1` / `#B2F2BB`\n"
            "- ☁️ **Baby Blue / Powder Blue (Xanh lơ mây):** `#AEC6CF` / `#B3E5FC`\n"
            "- 🪻 **Lavender (Tím hoa cà phấn):** `#E6E6FA` / `#D8BFD8`\n"
            "- 🧈 **Butter / Cream (Vàng kem bơ nhạt):** `#FFFACD` / `#FEF3C7`\n"
            "- 🍑 **Peach Pastel (Cam đào phấn):** `#FFDAB9` / `#FFE5B4`\n\n"
            "**4. Ứng dụng trong Nhiếp ảnh & Nghệ thuật Thị giác:**\n"
            "- **Thể loại ảnh:** Rất thịnh hành trong chụp chân dung nữ sinh phong cách Hàn Quốc/Nhật Bản, ảnh cưới lãng mạn (*lookbook dreamy wedding*), ảnh trẻ em, ảnh sản phẩm mỹ phẩm (*flatlay*), lookbook thời trang xuân hè.\n"
            "- **Kỹ thuật khi chụp:** Chụp dưới ánh sáng khuếch tán mềm (*diffused soft light*, trời có mây mỏng hoặc ánh sáng qua rèm trắng), bối cảnh và trang phục màu be/trắng, đo sáng hơi dư nhẹ (+0.3 EV đến +0.7 EV) để ảnh sáng trong mà không cháy viền.\n"
            "- **Mẹo viết Prompt AI (Stable Diffusion / Midjourney):**\n"
            "  `soft pastel color palette, delicate baby pink and mint tones, gentle diffused morning light, dreamy airy atmosphere, high-key photography, aesthetic lookbook, 8k resolution`."
        )
    },
    {
        "id": "autumn_palette",
        "keywords": [
            "bảng màu mùa thu", "màu mùa thu", "phối màu mùa thu", "autumn palette",
            "tone mùa thu", "mùa thu ấm áp", "phong cảnh mùa thu"
        ],
        "title": "Bảng Màu Mùa Thu Ấm Áp (Warm Autumn Palette)",
        "content": (
            "### 🍂 Bảng Màu Mùa Thu Ấm Áp (Warm Autumn Palette)\n\n"
            "Khác với màu pastel, bảng màu mùa thu sử dụng các **sắc tố đất (Earth Tones)** ấm áp, sâu lắng và giàu hoài niệm:\n\n"
            "1. **Cam Cháy / Terracotta:** `#E2725B` — Sắc màu lá phong đỏ rực và gạch nung cổ kính.\n"
            "2. **Vàng Mù Tạt / Mustard Ochre:** `#D4A017` — Ánh nắng vàng óng chiều thu và lá ngân hạnh.\n"
            "3. **Nâu Hạt Dẻ / Warm Chestnut:** `#7B3F00` — Thân cây gỗ cổ thụ và đất mẹ mộc mạc.\n"
            "4. **Xanh Rêu Ô Liu / Olive Moss:** `#556B2F` — Màu thảm cỏ rêu phong cân bằng tự nhiên.\n\n"
            "💡 **Nguyên lý phối màu:** Đây là cách phối màu tương đồng (*Analogous Harmony*), các màu nằm liền kề nhau trên bánh xe màu sắc, tạo cảm giác vô cùng ấm cúng, hoài niệm và bình yên."
        )
    },
    {
        "id": "bokeh",
        "keywords": [
            "bokeh là gì", "định nghĩa bokeh", "xóa phông là gì", "hiệu ứng bokeh",
            "creamy bokeh", "bokeh"
        ],
        "title": "Định Nghĩa Về Hiệu Ứng Bokeh Trong Nhiếp Ảnh",
        "content": (
            "### 🔮 Định Nghĩa Hiệu Ứng Bokeh Trong Nhiếp Ảnh\n\n"
            "**1. Bokeh là gì?**\n"
            "- Từ **Bokeh** bắt nguồn từ tiếng Nhật *暈け (boke)*, có nghĩa là *\"nhòe, mờ\"*. Trong nhiếp ảnh, Bokeh là thuật ngữ chỉ **chất lượng thẩm mỹ thị giác của những vùng nằm ngoài khoảng lấy nét (out-of-focus background)**.\n"
            "- Không chỉ đơn thuần là làm mờ hậu cảnh, Bokeh thể hiện cách mà các điểm sáng ở hậu cảnh tan chảy thành những đốm sáng tròn, lục giác hay xoáy mềm mại.\n\n"
            "**2. Các kiểu Bokeh phổ biến:**\n"
            "- **Creamy Bokeh (Mịn như kem):** Hậu cảnh tan chảy hoàn toàn mềm mịn, không có viền cứng, thường thấy trên các ống kính cao cấp như 85mm f/1.4 hoặc 70-200mm f/2.8.\n"
            "- **Circular / Bubble Bokeh (Bong bóng tròn):** Các nguồn sáng ban đêm (đèn đường, đèn neon) biến thành các đốm tròn lung linh rực rỡ.\n"
            "- **Swirly Bokeh (Xoáy tròn):** Hiệu ứng xoáy độc đáo của các ống kính cổ điển (như Helios 44-2 58mm f/2).\n\n"
            "**3. Cách tạo Bokeh đẹp nhất khi chụp:**\n"
            "- Dùng **khẩu độ lớn** (`f/1.4`, `f/1.8`, `f/2.8`).\n"
            "- Dùng **tiêu cự tele** (`85mm`, `105mm`, `135mm`).\n"
            "- Đặt **mẫu đứng càng gần máy ảnh** và **hậu cảnh càng xa mẫu** càng tốt.\n\n"
            "**4. Mẹo viết Prompt AI:**\n"
            "`creamy smooth circular bokeh background, 85mm f/1.4 lens, sparkling golden night lights, shallow depth of field, sharp subject focus`."
        )
    },
    {
        "id": "aperture",
        "keywords": [
            "khẩu độ là gì", "định nghĩa khẩu độ", "khẩu độ máy ảnh", "f-stop là gì",
            "khẩu độ", "f-stop", "lá khẩu"
        ],
        "title": "Định Nghĩa Khẩu Độ (Aperture / F-stop)",
        "content": (
            "### 📸 Định Nghĩa Khẩu Độ (Aperture / F-stop) Trong Nhiếp Ảnh\n\n"
            "**1. Khẩu độ là gì?**\n"
            "- Khẩu độ là **độ mở của các lá khẩu (diaphragm blades)** nằm bên trong ống kính máy ảnh. Nó kiểm soát **lượng ánh sáng** đi qua ống kính đến cảm biến và quyết định **độ sâu trường ảnh (Depth of Field)**.\n\n"
            "**2. Nguyên tắc số F (F-stop):**\n"
            "- **Số F càng nhỏ (f/1.2, f/1.4, f/1.8, f/2.8):** Khẩu độ mở càng lớn → Thu nhận nhiều ánh sáng hơn → Vùng nét mỏng, xóa phông mù mịt (thích hợp cho chân dung, chụp thiếu sáng).\n"
            "- **Số F càng lớn (f/8, f/11, f/16):** Khẩu độ khép càng nhỏ → Thu nhận ít ánh sáng hơn → Vùng nét rất dày, rõ từ tiền cảnh đến hậu cảnh (thích hợp cho phong cảnh, kiến trúc).\n\n"
            "**3. Điểm ngọt của ống kính (Sweet Spot):**\n"
            "- Hầu hết các ống kính đạt độ sắc nét quang học tối đa ở mức khẩu độ `f/5.6 – f/8`. Tránh khép quá sâu như `f/16` hoặc `f/22` vì hiện tượng nhiễu xạ (*diffraction*) sẽ làm giảm độ nét."
        )
    },
    {
        "id": "shutter_speed",
        "keywords": [
            "tốc độ màn trập là gì", "định nghĩa tốc độ màn trập", "shutter speed là gì",
            "tốc độ chụp là gì", "tốc độ màn trập", "màn trập"
        ],
        "title": "Định Nghĩa Tốc Độ Màn Trập (Shutter Speed)",
        "content": (
            "### ⏱️ Định Nghĩa Tốc Độ Màn Trập (Shutter Speed)\n\n"
            "**1. Tốc độ màn trập là gì?**\n"
            "- Tốc độ màn trập là **khoảng thời gian mà màn trập máy ảnh mở ra** để ánh sáng chiếu vào cảm biến hình ảnh (tính bằng phần giây hoặc giây).\n\n"
            "**2. Tác động thị giác của tốc độ màn trập:**\n"
            "- **Tốc độ cực nhanh (1/1000s – 1/8000s):** Bắt dính (đóng băng) khoảnh khắc chuyển động siêu tốc như giọt nước văng, chim đang đập cánh, vận động viên thể thao.\n"
            "- **Tốc độ an toàn cầm tay (1/125s – 1/250s):** Chụp người bình thường không bị rung tay (nguyên tắc: tốc độ ≥ 1/tiêu cự lens).\n"
            "- **Tốc độ chậm / Phơi sáng dài (1/2s – 30s):** Biến dòng nước chảy thành dải lụa mềm, làm nhòe vệt đèn xe cộ ban đêm (*light trails*), hoặc phơi sáng chụp dải ngân hà Milky Way (cần chân máy tripod cố định)."
        )
    },
    {
        "id": "iso",
        "keywords": [
            "định nghĩa iso", "iso là gì", "độ nhạy sáng iso", "chỉ số iso",
            "iso trong máy ảnh", "hạt nhiễu iso", "nhiễu hạt iso"
        ],
        "title": "Định Nghĩa ISO (Độ Nhạy Sáng Cảm Biến)",
        "content": (
            "### 💡 Định Nghĩa ISO (Độ Nhạy Sáng Cảm Biến)\n\n"
            "**1. ISO là gì?**\n"
            "- Trong máy ảnh số, **ISO** đại diện cho **mức độ khuếch đại tín hiệu điện tử của cảm biến hình ảnh** đối với ánh sáng thu nhận được.\n\n"
            "**2. Nguyên tắc sử dụng ISO:**\n"
            "- **ISO thấp (ISO 100 – 400):** Tín hiệu thuần khiết nhất, ảnh đạt độ mịn màng, dải tương phản động (dynamic range) cao nhất và màu sắc trung thực nhất. Sử dụng khi chụp ban ngày đủ sáng hoặc trong studio.\n"
            "- **ISO trung bình (ISO 800 – 1600):** Dùng khi chụp trong nhà, bóng râm hoặc hoàng hôn.\n"
            "- **ISO cao (ISO 3200 – 12800+):** Dùng trong điều kiện thiếu sáng ngặt nghèo ban đêm. *Đánh đổi:* Sẽ xuất hiện hạt nhiễu kỹ thuật số (*digital noise / grain*), màu sắc giảm độ tương phản.\n\n"
            "💡 **Mẹo:** Nếu ảnh chụp thiếu sáng bị vỡ hạt, bạn có thể đưa vào tab **Làm Nét Ảnh** của Clarity Studio với chế độ `SCUNet Khử Nhiễu 1X` để triệt tiêu hoàn toàn hạt nhiễu mà vẫn giữ trọn vẹn chi tiết!"
        )
    },
    {
        "id": "exposure_triangle",
        "keywords": [
            "tam giác đo sáng là gì", "tam giác phơi sáng", "exposure triangle là gì",
            "tam giác đo sáng", "tam giác phơi sáng là gì"
        ],
        "title": "Định Nghĩa Tam Giác Đo Sáng (Exposure Triangle)",
        "content": (
            "### 🔺 Định Nghĩa Tam Giác Đo Sáng (Exposure Triangle)\n\n"
            "Tam giác đo sáng là nền tảng cốt lõi của toàn bộ nghệ thuật nhiếp ảnh, bao gồm **3 chân kiềng tương tác qua lại lẫn nhau** để tạo ra độ phơi sáng chuẩn (0 EV):\n\n"
            "1. **Khẩu độ (Aperture):** Quyết định lượng ánh sáng đi vào và độ sâu trường ảnh (xóa phông hay nét sâu).\n"
            "2. **Tốc độ màn trập (Shutter Speed):** Quyết định thời gian nhận sáng và trạng thái chuyển động (bắt dính hay nhòe mượt).\n"
            "3. **Độ nhạy sáng (ISO):** Quyết định độ khuếch đại sáng điện tử và độ sạch của hạt ảnh (mịn hay nhiễu noise).\n\n"
            "⚖️ **Quy tắc bù trừ cân bằng:** Khi bạn thay đổi 1 yếu tố (ví dụ tăng tốc độ lên 1 stop để bắt dính mẫu), bạn phải mở khẩu thêm 1 stop hoặc tăng ISO lên 1 stop tương ứng để bức ảnh không bị thiếu sáng!"
        )
    },
    {
        "id": "dof",
        "keywords": [
            "độ sâu trường ảnh là gì", "dof là gì", "depth of field là gì",
            "độ sâu trường ảnh", "độ sâu trường nét"
        ],
        "title": "Định Nghĩa Độ Sâu Trường Ảnh (Depth of Field - DoF)",
        "content": (
            "### 🔍 Định Nghĩa Độ Sâu Trường Ảnh (Depth of Field - DoF)\n\n"
            "**1. DoF là gì?**\n"
            "- **Độ sâu trường ảnh (DoF)** là khoảng cách giữa điểm gần nhất và điểm xa nhất trong khung hình mà các chi tiết vẫn hiển thị rõ ràng và sắc nét có thể chấp nhận được.\n\n"
            "**2. Hai trạng thái DoF:**\n"
            "- **DoF mỏng / Nông (Shallow DoF):** Vùng nét rất hẹp, chỉ có chủ thể rõ nét còn phông nền trước và sau đều nhòe mờ (đặc trưng của ảnh chân dung, macro, chụp sản phẩm).\n"
            "- **DoF dày / Sâu (Deep DoF):** Toàn bộ khung cảnh từ nhánh cỏ dưới chân đến rặng núi phía xa đều nét căng (đặc trưng của ảnh phong cảnh, kiến trúc, kỷ yếu đông người).\n\n"
            "**3. Ba yếu tố quyết định DoF:**\n"
            "- Khẩu độ càng lớn (f/1.4) → DoF càng mỏng.\n"
            "- Tiêu cự càng dài (200mm) → DoF càng mỏng.\n"
            "- Khoảng cách từ máy đến chủ thể càng gần → DoF càng mỏng."
        )
    },
    {
        "id": "rule_of_thirds",
        "keywords": [
            "quy tắc 1/3 là gì", "quy tắc một phần ba", "rule of thirds là gì",
            "quy tắc 1/3", "bố cục 1/3"
        ],
        "title": "Định Nghĩa Quy Tắc 1/3 (Rule of Thirds)",
        "content": (
            "### 📐 Định Nghĩa Quy Tắc 1/3 (Rule of Thirds)\n\n"
            "**1. Quy tắc 1/3 là gì?**\n"
            "- Quy tắc 1/3 là nguyên lý bố cục kinh điển trong hội họa và nhiếp ảnh. Khung hình được chia thành một mạng lưới **9 ô chữ nhật đều nhau** bởi **2 đường thẳng đứng và 2 đường nằm ngang**.\n"
            "- Bốn giao điểm của các đường này được gọi là **Điểm Vàng / Điểm Quyền Lực (Power Points)**.\n\n"
            "**2. Cách áp dụng thực tế:**\n"
            "- **Chân dung:** Đặt đôi mắt hoặc khuôn mặt nhân vật vào đúng 1 trong 4 giao điểm trên, bức ảnh sẽ có điểm nhìn cuốn hút và tự nhiên hơn đặt chính giữa.\n"
            "- **Phong cảnh:** Đặt đường chân trời ở đường kẻ 1/3 dưới (nếu bầu trời mây đẹp) hoặc 1/3 trên (nếu mặt đất/mặt biển có tiền cảnh đẹp), tránh chia đôi khung hình ở giữa 50/50."
        )
    },
    {
        "id": "white_balance",
        "keywords": [
            "cân bằng trắng là gì", "white balance là gì", "nhiệt độ màu là gì",
            "độ kelvin là gì", "cân bằng trắng", "nhiệt độ màu"
        ],
        "title": "Định Nghĩa Cân Bằng Trắng (White Balance) & Nhiệt Độ Màu (Kelvin)",
        "content": (
            "### 🌡️ Định Nghĩa Cân Bằng Trắng & Nhiệt Độ Màu (Kelvin)\n\n"
            "**1. Cân bằng trắng (White Balance - WB) là gì?**\n"
            "- Là quá trình cân chỉnh màu sắc trong máy ảnh để **vật thể có màu trắng ngoài đời thực sẽ hiển thị đúng màu trắng tinh khiết trên ảnh**, không bị ám vàng hoặc ám xanh bởi các nguồn sáng môi trường.\n\n"
            "**2. Thang nhiệt độ màu Kelvin (K):**\n"
            "- **2000K – 3000K:** Ánh sáng ngọn nến, đèn dây tóc (ấm vàng cam đậm).\n"
            "- **5000K – 5500K:** Ánh sáng mặt trời ban ngày tiêu chuẩn (trung tính tự nhiên).\n"
            "- **6500K – 7500K:** Trời nhiều mây, bóng râm (lạnh xanh lam dịu).\n"
            "- **9000K – 10000K:** Trời xanh ngắt vùng cực hoặc tuyết trắng (rất lạnh xanh).\n\n"
            "💡 **Sáng tạo với WB:** Bạn có thể cố tình chỉnh WB cao hơn để ảnh ấm áp lãng mạn như chiều hoàng hôn, hoặc chỉnh WB thấp hơn để ảnh mang vẻ cô đơn, lạnh lùng điện ảnh."
        )
    },
    {
        "id": "raw_vs_jpeg",
        "keywords": [
            "ảnh raw là gì", "file raw là gì", "raw là gì", "raw khác jpeg",
            "định dạng raw", "tại sao nên chụp raw"
        ],
        "title": "Định Nghĩa Ảnh RAW & So Sánh Với JPEG",
        "content": (
            "### 📁 Định Nghĩa Ảnh RAW & So Sánh Với JPEG\n\n"
            "**1. Ảnh RAW là gì?**\n"
            "- **File RAW** (nghĩa đen là *\"thô\"*) là tệp dữ liệu nguyên bản thu nhận trực tiếp từ các cảm biến điểm ảnh, chưa hề qua bất kỳ thuật toán nén, khử nhiễu hay chỉnh màu nào của máy ảnh.\n"
            "- Nó đóng vai trò như **cuộn phim âm bản kỹ thuật số**.\n\n"
            "**2. So sánh RAW vs JPEG:**\n"
            "- **Độ sâu màu (Color Bit Depth):** RAW ghi lại 12-bit hoặc 14-bit màu (hàng tỷ sắc độ), trong khi JPEG chỉ lưu 8-bit (16.7 triệu màu).\n"
            "- **Cứu sáng & Chỉnh sửa:** Với RAW, bạn có thể dễ dàng kéo lại vùng trời bị cháy sáng (*highlights*) hoặc kéo sáng vùng tối (*shadows*) mà không bị vỡ hạt hay bệt màu.\n"
            "- **Dung lượng:** RAW nặng gấp 4–6 lần JPEG và cần phần mềm chuyên dụng (Lightroom, Photoshop) để xuất ảnh."
        )
    },
    {
        "id": "dynamic_range",
        "keywords": [
            "dynamic range là gì", "dải tương phản động là gì", "hdr là gì",
            "dynamic range", "dải dynamic range"
        ],
        "title": "Định Nghĩa Dải Tương Phản Động (Dynamic Range & HDR)",
        "content": (
            "### 🌄 Định Nghĩa Dải Tương Phản Động (Dynamic Range & HDR)\n\n"
            "**1. Dynamic Range là gì?**\n"
            "- **Dynamic Range (Dải tương phản động)** là tỷ lệ đo lường khoảng cách giữa vùng sáng nhất (*Highlights*) và vùng tối nhất (*Shadows*) mà cảm biến máy ảnh có thể đồng thời ghi nhận chi tiết rõ ràng mà không bị cháy trắng tinh hay bết đen xì.\n\n"
            "**2. HDR (High Dynamic Range):**\n"
            "- Khi cảnh chụp vượt quá giới hạn của cảm biến (như chụp ngược sáng hoàng hôn), kỹ thuật HDR kết hợp nhiều bức ảnh chụp ở các mức phơi sáng khác nhau (thiếu sáng, chuẩn sáng, dư sáng) thành một bức ảnh hoàn chỉnh lưu giữ chi tiết cả trời lẫn đất."
        )
    },
    {
        "id": "complementary_colors",
        "keywords": [
            "màu tương phản là gì", "phối màu tương phản là gì", "complementary colors là gì",
            "màu tương phản", "cặp màu tương phản"
        ],
        "title": "Định Nghĩa Màu Tương Phản (Complementary Colors)",
        "content": (
            "### 🎨 Định Nghĩa Màu Tương Phản (Complementary Colors)\n\n"
            "**1. Màu tương phản là gì?**\n"
            "- Là các cặp màu **nằm ở vị trí đối xứng nhau chính xác 180 độ trên vòng tròn bánh xe màu sắc (Color Wheel)**.\n"
            "- Khi đặt cạnh nhau, chúng tôn vinh và làm nổi bật lẫn nhau đến mức tối đa nhờ sự tương phản nhiệt độ và thị giác.\n\n"
            "**2. Ba cặp màu tương phản kinh điển:**\n"
            "- 🟡 **Vàng & Tím (Yellow & Purple):** Sang trọng, quý phái, bắt mắt.\n"
            "- 🔴 **Đỏ & Xanh Lá (Red & Green):** Lễ hội, nổi bật, sức sống tự nhiên.\n"
            "- 🟠 **Cam & Xanh Lam (Orange & Teal):** Cặp màu vàng son của điện ảnh Hollywood.\n\n"
            "💡 **Mẹo áp dụng:** Tránh dùng hai màu với tỷ lệ 50-50 vì sẽ gây tranh chấp thị giác. Hãy dùng quy tắc 70-30 hoặc 80-20 (ví dụ nền xanh lam 80%, nhân vật mặc váy cam ấm 20%)."
        )
    },
    {
        "id": "analogous_colors",
        "keywords": [
            "màu tương đồng là gì", "phối màu tương đồng là gì", "analogous colors là gì",
            "màu tương đồng", "bảng màu liền kề"
        ],
        "title": "Định Nghĩa Phối Màu Tương Đồng (Analogous Colors)",
        "content": (
            "### 🍃 Định Nghĩa Phối Màu Tương Đồng (Analogous Colors)\n\n"
            "**1. Màu tương đồng là gì?**\n"
            "- Là nhóm gồm **3 đến 5 màu sắc nằm liền kề sát cạnh nhau** trên vòng tròn bánh xe màu sắc (ví dụ: Vàng – Vàng Cam – Cam).\n\n"
            "**2. Cảm xúc thẩm mỹ:**\n"
            "- Mang lại cảm giác **vô cùng hài hòa, êm dịu, dễ chịu và thanh bình**, không tạo ra sự xung đột thị giác gắt gao như màu tương phản.\n"
            "- Thường thấy trong bảng màu thiên nhiên cây cỏ (*Xanh lá mạ – Xanh lục – Xanh ngọc*) hoặc mùa thu lá rụng (*Vàng – Cam – Đỏ trầm*)."
        )
    },
    {
        "id": "teal_orange",
        "keywords": [
            "tone teal orange là gì", "teal and orange là gì", "màu teal orange",
            "teal orange", "tone điện ảnh teal orange"
        ],
        "title": "Định Nghĩa Tông Màu Teal & Orange Điện Ảnh",
        "content": (
            "### 🎬 Định Nghĩa Tông Màu Teal & Orange Điện Ảnh\n\n"
            "**1. Teal & Orange là gì?**\n"
            "- Đây là bảng phân màu (*Color Grading*) thống trị ngành công nghiệp điện ảnh Hollywood trong suốt 2 thập kỷ qua (từ Transformers, Mad Max, Blade Runner đến các MV ca nhạc hiện đại).\n\n"
            "**2. Nguyên lý khoa học:**\n"
            "- Màu da con người (bất kể chủng tộc) luôn nằm trong dải sắc thái **Cam ấm (Orange/Peach)**.\n"
            "- Màu tương phản trực tiếp đối xứng với Cam trên bánh xe màu là **Xanh mòng két (Teal / Lam lục)**.\n"
            "- Bằng cách đẩy vùng tối (*Shadows*) sang ánh xanh Teal và giữ vùng sáng da người (*Highlights*) ở sắc Cam ấm, nhân vật chính sẽ luôn bật sáng và nổi khối 3D tách biệt khỏi cảnh nền."
        )
    },
    {
        "id": "cyberpunk_neon",
        "keywords": [
            "cyberpunk", "màu neon", "phối màu cyberpunk", "neon sci-fi", "phong cách cyberpunk"
        ],
        "title": "Phối Màu & Ánh Sáng Cyberpunk Neon Sci-Fi",
        "content": (
            "### 🏙️ Định Nghĩa Phong Cách Phối Màu Cyberpunk Neon\n\n"
            "**1. Đặc trưng thị giác Cyberpunk:**\n"
            "- **Màu chủ đạo:** Sự đối lập cực hạn giữa **Xanh Neon / Cyan (`#00F0FF`)** và **Hồng Tím Huỳnh Quang / Hot Magenta (`#FF007F`)**.\n"
            "- **Nền tương phản:** Xanh đen thẫm (`#0B0E14`) của đêm tối và mặt đường nhựa ướt mưa.\n"
            "- **Màu điểm nhấn:** Vàng điện quang (`#FFE600`) hoặc Xanh lục acid (`#00FF66`).\n\n"
            "**2. Mẹo viết Prompt AI:**\n"
            "`cyberpunk neon lighting, vibrant glowing cyan and magenta reflections, wet asphalt street at night, futuristic volumetric mist, 8k octane render`."
        )
    },
    {
        "id": "prompt_engineering",
        "keywords": [
            "prompt là gì", "lời nhắc là gì", "negative prompt là gì", "cách viết prompt",
            "prompt trong ai", "prompt stable diffusion"
        ],
        "title": "Định Nghĩa Prompt & Negative Prompt Trong AI Sinh Ảnh",
        "content": (
            "### ✍️ Định Nghĩa Prompt & Negative Prompt Trong AI Sinh Ảnh\n\n"
            "**1. Prompt (Lời nhắc tích cực) là gì?**\n"
            "- Prompt là câu lệnh văn bản mô tả cụ thể chủ thể, hành động, bối cảnh, ánh sáng, góc máy và phong cách nghệ thuật mà bạn muốn mô hình Diffusion tạo ra.\n"
            "- **Cấu trúc vàng của 1 Prompt đỉnh cao:**\n"
            "  `[Chủ thể chính] + [Chi tiết trang phục/hành động] + [Bối cảnh môi trường] + [Ánh sáng & Thời tiết] + [Phong cách & Ống kính máy ảnh] + [Chất lượng chi tiết (8k, masterpiece)]`.\n\n"
            "**2. Negative Prompt (Lời nhắc tiêu cực) là gì?**\n"
            "- Là danh sách các yếu tố bạn **cấm AI đưa vào bức ảnh** (ví dụ: mờ nhòe, thừa ngón tay, mắt méo, chữ, watermark, tỷ lệ cơ thể biến dạng).\n"
            "- Giúp mô hình lọc bỏ các lỗi phát sinh thường gặp trong không gian tiềm ẩn (*latent space*)."
        )
    },
    {
        "id": "cfg_scale",
        "keywords": [
            "cfg scale là gì", "cfg là gì", "classifier free guidance", "chỉ số cfg",
            "thông số cfg scale"
        ],
        "title": "Định Nghĩa Chỉ Số CFG Scale Trong AI Tạo Ảnh",
        "content": (
            "### 🎚️ Định Nghĩa Chỉ Số CFG Scale (Classifier-Free Guidance)\n\n"
            "**1. CFG Scale là gì?**\n"
            "- Là thông số điều khiển **mức độ trung thành và tuân thủ chặt chẽ của mô hình AI đối với câu lệnh Prompt** của bạn.\n\n"
            "**2. Thang giá trị khuyên dùng:**\n"
            "- **CFG 1.0 – 4.0:** AI sáng tạo tự do, ít nghe theo prompt, ảnh có thể bay bổng nhưng lệch ý.\n"
            "- **CFG 7.0 – 8.5 (Khoảng lý tưởng nhất):** Bức ảnh tuân thủ sát sao ý tưởng mô tả mà màu sắc vẫn mềm mại, tự nhiên và cân bằng chi tiết.\n"
            "- **CFG > 12.0:** AI bị ép buộc quá mức, dễ xuất hiện hiện tượng cháy sáng, viền đen sắc gắt, bão hòa màu quá đà và biến dạng hình thái (*burnt artifacts*)."
        )
    },
    {
        "id": "denoising_strength",
        "keywords": [
            "denoising strength là gì", "strength trong img2img", "độ biến đổi ảnh",
            "strength trong i2i"
        ],
        "title": "Định Nghĩa Denoising Strength Trong Image-to-Image",
        "content": (
            "### 🔄 Định Nghĩa Denoising Strength Trong Chế Độ Image-to-Image (I2I)\n\n"
            "**1. Denoising Strength là gì?**\n"
            "- Là chỉ số kiểm soát **lượng nhiễu được thêm vào ảnh gốc trước khi AI bắt đầu quá trình khử nhiễu tái tạo lại** (từ 0.0 đến 1.0).\n\n"
            "**2. Lựa chọn mức Strength phù hợp:**\n"
            "- **0.25 – 0.35 (Bảo toàn cao):** Giữ gần như 100% đường nét, phong cách và bố cục ảnh gốc, chỉ tinh chỉnh màu sắc và làm mịn.\n"
            "- **0.40 – 0.50 (Cân bằng - Khuyên dùng):** Giữ nguyên khối hình ảnh nguồn nhưng cho phép AI bổ sung chi tiết mới mượt mà theo Prompt.\n"
            "- **0.70 – 0.85 (Biến đổi mạnh):** Biến ảnh chụp thật thành tranh vẽ Anime / 3D Pixar hoàn toàn mới dựa trên khung bố cục gốc."
        )
    },
    {
        "id": "clarity_tools",
        "keywords": [
            "gfpgan", "realesrgan", "scunet", "lama outpainting", "birefnet",
            "tính năng clarity", "công nghệ clarity"
        ],
        "title": "Định Nghĩa Các Công Nghệ AI Cốt Lõi Trên Clarity AI Studio",
        "content": (
            "### 🧠 Hệ Thống Trí Tuệ Nhân Tạo Trên Clarity AI Studio\n\n"
            "Clarity AI Studio được trang bị những mô hình thị giác máy tính tân tiến nhất:\n\n"
            "1. **Real-ESRGAN:** Phục hồi siêu phân giải 4X, tái tạo chi tiết vi mô cho ảnh phong cảnh, xe cộ và đồ họa.\n"
            "2. **GFPGAN (Generative Facial Prior):** Mạng nơ-ron chuyên biệt phục hồi khuôn mặt, tái tạo từng sợi lông mi, con ngươi và cấu trúc da thật tự nhiên.\n"
            "3. **SCUNet:** Mô hình khử nhiễu và làm mịn các bức ảnh cũ bị vỡ hạt ISO, ẩm mốc.\n"
            "4. **LaMa Outpainting:** Mô hình khuếch tán mở rộng biên ảnh sang 16:9 hoặc 9:16 mà không bị cắt cụt ảnh gốc.\n"
            "5. **BiRefNet:** Mô hình phân đoạn ngữ nghĩa tách nền siêu chính xác đến từng sợi tóc tơ và viền áo mỏng.\n"
            "6. **Salesforce BLIP & CLIP:** Hệ thống hiểu ảnh đa phương thức (Vision-Language) giúp bóc tách ý niệm và tìm kiếm ảnh bằng ngữ nghĩa."
        )
    }
]

QA_KNOWLEDGE_BASE = KNOWLEDGE_DEFINITIONS


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

        # Step 2: Vision Model - Always prioritize local BLIP (fast on CUDA & locally cached)
        vision_caption = None
        if HAS_TORCH:
            try:
                vision_caption = self._call_local_blip(img)
            except Exception as e:
                print(f"[LLMEngine] BLIP error in image_to_prompt: {e}")

        if not vision_caption and self.active_provider == "ollama":
            vision_caption = self._call_ollama_vision(img, prompt_hint)

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
            # If user provided guidance, append context
            if prompt_hint and len(prompt_hint) > 3 and not any(k in prompt_hint.lower() for k in ["promt", "prompt", "tạo ảnh", "ảnh này", "cho tôi", "mô tả"]):
                from generator import smart_translate_prompt
                translated_hint = smart_translate_prompt(prompt_hint)
                base_subject = f"{base_subject}, {translated_hint}"
        else:
            # Fallback heuristic subject description
            base_subject = "a compelling detailed visual composition, high aesthetic appeal"

        subj_low = base_subject.lower()
        if any(w in subj_low for w in ["snowman", "pumpkin", "lantern", "cartoon", "cute", "3d", "toy", "figurine", "character", "pixar", "disney", "ghost"]):
            inferred_style = "pixar3d"
        elif any(w in subj_low for w in ["anime", "manga", "illustration", "waifu", "shinkai"]):
            inferred_style = "anime"
        elif any(w in subj_low for w in ["cyberpunk", "neon", "sci-fi", "futuristic"]):
            inferred_style = "cyberpunk"
        elif any(w in subj_low for w in ["fantasy", "dragon", "magic", "mystical", "castle"]):
            inferred_style = "fantasy"
        elif any(w in subj_low for w in ["portrait", "face", "girl", "woman", "man", "person"]):
            inferred_style = "photorealistic"
        else:
            inferred_style = "cinematic"

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
                "provider_used": "Salesforce BLIP Vision Engine" if vision_caption else "Clarity Computer Vision Engine"
            }
        }

    # ------------------------------------------------------------------------
    # 7. Core Feature 3: Photography & Color Chatbot Consultant
    # ------------------------------------------------------------------------

    def _lookup_knowledge(self, user_msg: str) -> Optional[str]:
        """Matches user query against verified photography & color theory knowledge base with high precision."""
        import re
        msg_low = user_msg.lower().strip()
        for item in KNOWLEDGE_DEFINITIONS:
            for kw in item["keywords"]:
                if len(kw) <= 4:
                    if re.search(r'\b' + re.escape(kw) + r'\b', msg_low):
                        return item["content"]
                else:
                    if kw in msg_low:
                        return item["content"]
        return None

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

        # Check reverse image search request
        if has_image and any(k in msg_clean.lower() for k in ["tìm ảnh tương tự", "ảnh tương tự", "tìm giống", "tương tự", "similar", "reverse"]):
            from visual_search import visual_search_engine
            scope = "web" if any(w in msg_clean.lower() for w in ["trên web", "trên mạng", "internet", "toàn cầu"]) else "all"
            results = visual_search_engine.search_by_image(image_path, top_k=8, min_similarity=0.20, scope=scope)
            count = len(results)
            scope_label = "trên các trang Web Internet & Thư viện máy" if scope == "all" else "trên các trang Web Internet"
            if count > 0:
                bot_reply = (
                    f"### 🔍 Đã Tìm Thấy {count} Hình Ảnh Tương Tự Bằng Mô Hình CLIP AI ({scope_label})!\n\n"
                    f"Dựa trên các đặc trưng thị giác (màu sắc, bố cục, chủ thể) của ảnh bạn cung cấp, "
                    f"dưới đây là các ảnh có độ tương đồng cao nhất:"
                )
            else:
                bot_reply = "⚠️ Chưa tìm thấy ảnh nào có độ tương đồng đủ cao với ảnh bạn cung cấp."
            return {
                "status": "success",
                "reply": bot_reply,
                "search_results": results,
                "is_search_result": True
            }

        # If user uploaded an image: Default is Image-to-Prompt Reverse Engineering (BLIP AI)
        if has_image:
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
                    f"**📸 Chi tiết thị giác nhận diện (BLIP AI Vision + Computer Vision):**\n"
                    f"- **Chủ thể nhận diện:** *\"{data.get('vision_caption', '')}\"*\n"
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

        # If user asks for prompt for an image but hasn't uploaded one
        if not has_image and any(k in msg_clean.lower() for k in ["promt để tạo ảnh này", "prompt để tạo ảnh này", "tạo ảnh này", "vẽ giống ảnh này", "ảnh này prompt gì"]):
            return {
                "status": "success",
                "reply": "💡 **Bạn hãy tải lên hoặc kéo thả bức ảnh bạn muốn lấy prompt vào ô chat** (bấm nút 📎 Đính Kèm Ảnh bên dưới), hệ thống AI Vision (BLIP) sẽ nhận diện chính xác chủ thể, màu sắc và tạo prompt tối ưu ngay lập tức cho bạn nhé!"
            }

        # Check verified knowledge definitions first (100% accurate, authoritative explanations)
        knowledge_reply = self._lookup_knowledge(msg_clean)
        if knowledge_reply:
            return {
                "status": "success",
                "reply": knowledge_reply
            }

        # Check text-to-image search request in chat
        search_triggers = ["tìm ảnh", "tìm hình", "tìm kiếm ảnh", "tìm kiếm hình", "search image", "search ", "tìm các ảnh", "tìm những ảnh", "tìm bức ảnh"]
        is_search_query = any(msg_clean.lower().startswith(p) for p in search_triggers) or any(k in msg_clean.lower() for k in ["tìm ảnh về", "tìm hình về", "tìm kiếm bức ảnh", "tìm kiếm ảnh"])
        if is_search_query:
            search_q = msg_clean
            for p in ["tìm kiếm hình ảnh về", "tìm kiếm ảnh về", "tìm hình ảnh về", "tìm kiếm hình ảnh", "tìm kiếm ảnh", "tìm các ảnh", "tìm những ảnh", "tìm bức ảnh", "tìm ảnh về", "tìm hình về", "tìm ảnh ", "tìm hình ", "search image ", "search ", "tìm "]:
                if search_q.lower().startswith(p):
                    search_q = search_q[len(p):].strip()
                    break

            scope = "all"
            if any(w in search_q.lower() for w in ["trên web", "trên mạng", "trên internet", "từ web"]):
                scope = "web"
                for w in ["trên web", "trên mạng", "trên internet", "từ web"]:
                    search_q = re.sub(re.escape(w), "", search_q, flags=re.IGNORECASE).strip()

            if search_q:
                from visual_search import visual_search_engine
                from entity_knowledge import detect_franchise_entity
                entity_info = detect_franchise_entity(search_q)

                results = visual_search_engine.search_by_text(search_q, top_k=8, min_similarity=0.15, scope=scope)
                count = len(results)
                scope_label = "Web Internet & Thư Viện" if scope == "all" else "Web Toàn Cầu"
                if count > 0:
                    if entity_info:
                        char_md = "\n".join([
                            f"- {ch['icon']} **{ch['name']}**: *{ch['badge']}*"
                            for ch in entity_info.get("characters", [])[1:8]
                        ])
                        bot_reply = (
                            f"### {entity_info['icon']} Đã Nhận Diện Vũ Trụ: **{entity_info['title']}**!\n\n"
                            f"> **Tác giả / Nguồn gốc:** {entity_info['creator']} · {entity_info['category']}\n\n"
                            f"{entity_info['description']}\n\n"
                            f"**👥 Các nhân vật & yếu tố tiêu biểu đã nhận diện:**\n{char_md}\n\n"
                            f"**🔍 Dưới đây là {count} hình ảnh các nhân vật & tác phẩm liên quan ({scope_label}):**"
                        )
                    else:
                        bot_reply = (
                            f"### 🔎 Đã Tìm Thấy {count} Hình Ảnh Phù Hợp ({scope_label}) Với Từ Khóa: *\"{search_q}\"*!\n\n"
                            f"Mô hình **OpenAI CLIP ViT-B/32** đã đối chiếu ngữ nghĩa và trích xuất các hình ảnh sát nhất:"
                        )
                else:
                    bot_reply = f"⚠️ Không tìm thấy ảnh nào khớp với từ khóa *\"{search_q}\"*. Bạn có thể thử từ khóa tương tự hoặc chuyển sang chế độ tìm kiếm Web!"
                return {
                    "status": "success",
                    "reply": bot_reply,
                    "search_results": results,
                    "is_search_result": True,
                    "query": search_q,
                    "entity_info": entity_info
                }
        # Direct Creative Prompt Generation in chat
        prompt_triggers = [
            "viết prompt", "tạo prompt", "gợi ý prompt", "prompt cho", "prompt ảnh",
            "sinh prompt", "câu lệnh tạo ảnh", "viết câu lệnh", "prompt vẽ", "làm prompt",
            "hãy viết prompt", "giúp tôi viết prompt", "viết cho tôi prompt"
        ]
        is_prompt_req = any(k in msg_clean.lower() for k in prompt_triggers) or (
            msg_clean.lower().startswith(("tôi muốn tạo ảnh", "tạo ảnh phong cách", "hãy vẽ")) and len(msg_clean) > 12
        )
        if is_prompt_req:
            return self._format_prompt_chat_reply(msg_clean)

        # Provider routing with intelligent fallback
        if self.active_provider == "local_qwen":
            reply = self._call_local_qwen(msg_clean, history=chat_history)
            if reply and len(reply.strip()) > 8:
                return {"status": "success", "reply": reply}
            # Fallback to conversational engine if local neural returned empty/failed
            reply = self._generate_conversational_reply(msg_clean, chat_history)
            return {"status": "success", "reply": reply}

        elif self.active_provider == "ollama":
            reply = self._call_ollama_chat(msg_clean, image_path, chat_history)
            if reply:
                return {"status": "success", "reply": reply}
            reply = self._generate_conversational_reply(msg_clean, chat_history)
            return {"status": "success", "reply": reply}

        elif self.active_provider == "lm_studio":
            reply = self._call_openai_chat(msg_clean, chat_history)
            if reply:
                return {"status": "success", "reply": reply}
            reply = self._generate_conversational_reply(msg_clean, chat_history)
            return {"status": "success", "reply": reply}

        # Default built_in: High-speed, natural, context-aware conversational engine
        reply = self._generate_conversational_reply(msg_clean, chat_history)
        return {"status": "success", "reply": reply}

    def _format_prompt_chat_reply(self, user_msg: str) -> Dict[str, Any]:
        """Creates an enhanced SD/Midjourney prompt directly inside chat with action button."""
        clean_concept = user_msg
        for trigger in [
            "viết giúp tôi prompt về", "viết giúp tôi prompt", "hãy viết prompt cho",
            "hãy viết prompt về", "hãy viết prompt", "viết prompt cho", "viết prompt về",
            "viết prompt", "tạo giúp tôi prompt", "tạo prompt cho", "tạo prompt về",
            "tạo prompt", "gợi ý prompt cho", "gợi ý prompt về", "gợi ý prompt",
            "sinh prompt cho", "sinh prompt", "câu lệnh tạo ảnh cho", "câu lệnh tạo ảnh",
            "tôi muốn tạo ảnh", "tạo ảnh phong cách", "hãy vẽ"
        ]:
            if clean_concept.lower().startswith(trigger):
                clean_concept = clean_concept[len(trigger):].strip()
                break

        if not clean_concept:
            clean_concept = user_msg

        res = self.optimize_prompt(clean_concept)
        if res.get("status") == "success":
            data = res["data"]
            reply = (
                f"### ✨ Đã Tạo Prompt Nghệ Thuật Cho Ý Tưởng Của Bạn!\n\n"
                f"Dựa trên ý tưởng: *\"{clean_concept}\"*, mình đã tối ưu hóa bố cục, ánh sáng điện ảnh và chi tiết 8K sắc nét:\n\n"
                f"**🎨 Prompt Tối Ưu (Stable Diffusion / Midjourney):**\n"
                f"```text\n{data['optimized_prompt']}\n```\n\n"
                f"**🛡️ Negative Prompt Khuyên Dùng:**\n"
                f"```text\n{data['negative_prompt']}\n```\n\n"
                f"**💡 Gợi ý thiết lập:**\n"
                f"- **Tỷ lệ khung hình:** `{data.get('aspect_ratio', '1:1')}`\n"
                f"- **Phong cách:** {data.get('style_preset', 'Cinematic').capitalize()}\n\n"
                f"> *Mẹo:* Bạn có thể bấm nút **'Sử dụng trong Tạo Ảnh AI'** bên dưới hoặc sao chép prompt trên dán sang tab **Tạo Ảnh AI (Generator)** để sinh ảnh ngay nhé!"
            )
            return {
                "status": "success",
                "reply": reply,
                "prompt_data": {
                    "generated_prompt": data['optimized_prompt'],
                    "negative_prompt": data['negative_prompt'],
                    "aspect_ratio": data.get('aspect_ratio', '1:1'),
                    "style_preset": data.get('style_preset', 'cinematic')
                }
            }
        return {
            "status": "success",
            "reply": f"Rất tiếc là mình chưa trích xuất được prompt cho ý tưởng *\"{clean_concept}\"*. Bạn có thể miêu tả chi tiết thêm một chút về chủ thể, trang phục hoặc bối cảnh không?"
        }

    def _analyze_chat_context(
        self,
        message: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, Optional[str]]:
        msg_low = message.lower().strip()

        # Track previous topic from chat history
        prev_topic = None
        if chat_history:
            for item in reversed(chat_history[-6:]):
                txt = (item.get("content") or "").lower()
                if any(k in txt for k in ["chân dung", "portrait", "xóa phông", "bokeh", "chụp người", "chụp mẫu", "mặt người", "85mm"]):
                    prev_topic = "portrait"
                    break
                elif any(k in txt for k in ["hoàng hôn", "sunset", "golden hour", "bình minh", "ngược sáng", "silhouette"]):
                    prev_topic = "sunset"
                    break
                elif any(k in txt for k in ["chụp đêm", "ban đêm", "phơi sáng", "night", "vệt đèn", "tối"]):
                    prev_topic = "night"
                    break
                elif any(k in txt for k in ["phong cảnh", "landscape", "thiên nhiên", "núi", "biển", "kiến trúc"]):
                    prev_topic = "landscape"
                    break
                elif any(k in txt for k in ["đường phố", "street", "đời thường", "khoảnh khắc"]):
                    prev_topic = "street"
                    break
                elif any(k in txt for k in ["điện thoại", "smartphone", "iphone", "android"]):
                    prev_topic = "smartphone"
                    break
                elif any(k in txt for k in ["phối màu", "bảng màu", "cyberpunk", "vintage", "pastel", "mùa thu", "màu sắc"]):
                    prev_topic = "color"
                    break
                elif any(k in txt for k in ["làm nét", "phục hồi", "gfpgan", "scunet", "mờ", "vỡ hạt", "tách nền", "đổi tỷ lệ", "nén ảnh"]):
                    prev_topic = "studio_tools"
                    break
                elif any(k in txt for k in ["prompt", "tạo ảnh", "câu lệnh"]):
                    prev_topic = "prompt"
                    break

        # Check follow-up inquiries
        is_follow_up = any(msg_low.startswith(p) for p in [
            "thế còn", "vậy còn", "vậy ", "thế ", "thế nếu", "sao lại", "còn ",
            "nói rõ hơn", "giải thích thêm", "cho ví dụ", "còn gì nữa", "sao thế",
            "thế thì", "vậy thì", "còn về"
        ])

        # Greetings & conversational pleasantries first
        if re.search(r'^(chào|xin chào|hello|hi|hey|alo|chào bạn|chào bot|bạn ơi|chào buổi sáng|chào buổi tối|good morning|hi bot|hế lô)\b', msg_low):
            return "greeting", prev_topic

        if any(k in msg_low for k in ["bạn là ai", "bạn tên gì", "giới thiệu bản thân", "bạn làm được gì", "bạn có thể làm gì", "chức năng của bạn"]):
            return "identity", prev_topic

        if any(k in msg_low for k in ["cảm ơn", "thanks", "thank you", "tuyệt vời", "hay quá", "bạn giỏi", "chuẩn luôn", "ok bạn", "tốt lắm"]):
            return "thanks", prev_topic

        if any(k in msg_low for k in ["tạm biệt", "bye", "hẹn gặp lại"]):
            return "farewell", prev_topic

        # Studio Tools & Software Features (Higher priority when app functions are mentioned)
        if any(k in msg_low for k in [
            "làm nét", "phục hồi", "gfpgan", "scunet", "ảnh mờ", "vỡ hạt", "ảnh cũ",
            "đổi tỷ lệ", "mở rộng ảnh", "mở rộng khung", "lama", "tách nền", "xóa nền", "birefnet",
            "nén ảnh", "giảm dung lượng", "chuyển định dạng", "clarity", "tính năng", "công cụ"
        ]):
            return "studio_tools", prev_topic

        # Primary Photography Genres
        if any(k in msg_low for k in ["chân dung", "portrait", "xóa phông", "bokeh", "chụp người", "chụp mẫu"]):
            return "portrait", prev_topic

        if any(k in msg_low for k in ["hoàng hôn", "sunset", "golden hour", "bình minh", "sunrise", "ngược sáng", "silhouette"]):
            return "sunset", prev_topic

        if any(k in msg_low for k in ["chụp đêm", "ban đêm", "phơi sáng", "long exposure", "vệt đèn", "chụp tối"]):
            return "night", prev_topic

        if any(k in msg_low for k in ["phong cảnh", "landscape", "thiên nhiên", "chụp núi", "chụp biển", "kiến trúc", "tòa nhà"]):
            return "landscape", prev_topic

        if any(k in msg_low for k in ["đường phố", "street", "đời thường", "khoảnh khắc", "bắt dính"]):
            return "street", prev_topic

        if any(k in msg_low for k in ["điện thoại", "smartphone", "iphone", "android", "camera điện thoại"]):
            return "smartphone", prev_topic

        if any(k in msg_low for k in ["bố cục", "quy tắc 1/3", "đường dẫn", "khung trong khung", "tỷ lệ vàng", "composition"]):
            return "composition", prev_topic

        if any(k in msg_low for k in ["phối màu", "bảng màu", "cyberpunk", "vintage", "retro", "pastel", "mùa thu", "autumn", "màu sắc", "teal orange", "mã màu"]):
            return "color", prev_topic

        # Targeted Parameter Inquiries / Follow-ups
        if any(k in msg_low for k in ["khẩu độ", "khau do", "aperture", "f/", "f1.", "f2.", "f4", "f8", "độ mở"]):
            return "follow_up_aperture", prev_topic

        if any(k in msg_low for k in ["ống kính", "ong kinh", "lens", "tiêu cự", "tieu cu", "chọn lens", "mua lens", "85mm", "50mm", "35mm"]):
            return "follow_up_lens", prev_topic

        if any(k in msg_low for k in ["iso", "độ nhạy sáng", "nhiễu hạt"]):
            return "follow_up_iso", prev_topic

        if any(k in msg_low for k in ["tốc độ", "toc do", "shutter", "màn trập", "tốc độ chụp"]):
            return "follow_up_shutter", prev_topic

        if any(k in msg_low for k in ["nói rõ hơn", "giải thích thêm", "cho ví dụ", "chi tiết hơn", "cụ thể hơn"]) and prev_topic:
            return "follow_up_details", prev_topic

        return "general", prev_topic

    def _generate_conversational_reply(
        self,
        message: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        intent, prev_topic = self._analyze_chat_context(message, chat_history)

        # 1. GREETING
        if intent == "greeting":
            greetings = [
                "Chào bạn! Rất vui được đồng hành cùng bạn tại Clarity AI Studio. ✨ Hôm nay bạn đang có ý tưởng chụp ảnh nào mới, cần tư vấn thông số máy, phối màu hay cần hỗ trợ làm nét/chỉnh sửa bức ảnh nào không?",
                "Xin chào bạn! Mình là Trợ lý AI của Clarity Studio đây. Mình luôn sẵn sàng đồng hành cùng bạn. Hôm nay mình có thể giúp gì cho bạn: tư vấn kỹ thuật nhiếp ảnh, mẹo chụp đẹp hay tạo prompt nghệ thuật?",
                "Chào bạn nhé! Chúc bạn một ngày tràn đầy cảm hứng sáng tạo. Bạn đang chuẩn bị chụp ảnh hay muốn phục hồi bức ảnh nào thế, chia sẻ cho mình biết nhé!",
                "Chào bạn! Rất vui được gặp lại bạn. Hôm nay bạn muốn khám phá mẹo nhiếp ảnh, tìm bảng màu ưng ý hay tối ưu hóa bức ảnh nào cùng Clarity AI?"
            ]
            if chat_history and len(chat_history) >= 2:
                return "Chào bạn nhé! Mình vẫn ở đây và sẵn sàng cùng bạn sáng tạo. Bạn có điều gì muốn trao đổi tiếp về bức ảnh hay chủ đề vừa rồi không?"
            import random
            return random.choice(greetings)

        # 2. IDENTITY & CAPABILITIES
        if intent == "identity":
            return (
                "Chào bạn! Mình là **Trợ lý AI Nhiếp Ảnh, Màu Sắc & Sáng Tạo** của Clarity AI Studio. 🌟\n\n"
                "Mình được thiết kế như một người bạn đồng hành chuyên nghiệp, sẵn sàng hỗ trợ bạn:\n"
                "- 📸 **Tư vấn thông số & Kỹ thuật chụp:** Chọn ống kính (35mm/50mm/85mm), khẩu độ, ISO, tốc độ, bố cục 1/3, cách chụp hoàng hôn, ngược sáng, phơi sáng đêm hay mẹo chụp đẹp bằng điện thoại.\n"
                "- 🎨 **Phối màu & Lý thuyết thẩm mỹ:** Bảng màu Cyberpunk neon, Vintage film hoài niệm, Pastel Hàn Quốc, Mùa thu ấm áp, quy tắc 60-30-10.\n"
                "- ✍️ **Sáng tác Prompt AI chuẩn 8K:** Chỉ cần bạn chia sẻ ý tưởng bằng tiếng Việt, mình sẽ viết prompt tiếng Anh chuẩn Masterpiece kèm Negative Prompt cho Stable Diffusion / Midjourney.\n"
                "- 🛠️ **Làm chủ Clarity Studio:** Hướng dẫn làm nét chân dung (GFPGAN), khử mờ SCUNet, mở rộng ảnh không cắt xén bằng AI Fill (LaMa), tách nền tóc tơ (BiRefNet) và nén ảnh tối ưu dung lượng.\n"
                "- 🖼️ **Phân tích hình ảnh:** Bạn có thể kéo thả hoặc đính kèm ảnh vào khung chat, mình sẽ phân tích bảng màu HEX, ánh sáng và bóc tách prompt để tái tạo phong cách!\n\n"
                "Hôm nay bạn muốn bắt đầu với nội dung nào trước?"
            )

        # 3. THANKS
        if intent == "thanks":
            return "Không có gì đâu nè! Rất vui vì đã hỗ trợ được cho bạn. ✨ Chúc bạn có những bức ảnh thật ưng ý và tràn đầy cảm hứng nghệ thuật. Cần hỏi thêm điều gì cứ thoải mái nhắn mình nhé!"

        # 4. FAREWELL
        if intent == "farewell":
            return "Tạm biệt bạn nhé! Chúc bạn một ngày tuyệt vời và có thật nhiều tác phẩm ảnh ưng ý. Hẹn gặp lại bạn bất cứ lúc nào!"

        # 5. PORTRAIT PHOTOGRAPHY
        if intent == "portrait":
            return (
                "### 📸 Kỹ Thuật Chụp Chân Dung Nghệ Thuật & Xóa Phông Bokeh\n\n"
                "Để chụp ảnh chân dung ngoài trời đẹp, trong trẻo và nổi bật chủ thể:\n\n"
                "1. **Ống kính (Lens) khuyên dùng:**\n"
                "- **85mm f/1.4 hoặc f/1.8:** 'Tiêu chuẩn vàng' của nhiếp ảnh chân dung. Tiêu cự 85mm giữ tỷ lệ khuôn mặt nhân trắc học tự nhiên (không bị méo viền), nén hậu cảnh lại gần và tạo hiệu ứng xóa phông cực kỳ mượt mà (*creamy bokeh*).\n"
                "- **50mm f/1.8:** Ống kính tiêu cự mắt người đa dụng, chụp góc rộng hơn một chút, rất tốt khi chụp nửa người trong quán cà phê hoặc không gian hẹp.\n\n"
                "2. **Cài đặt thông số máy ảnh:**\n"
                "- **Khẩu độ (Aperture):** Mở lớn `f/1.4 – f/2.8` để xóa mờ hậu cảnh và tách chủ thể sắc nét.\n"
                "- **Tốc độ màn trập (Shutter Speed):** Tối thiểu `1/160s – 1/250s` để đóng băng cử động nhẹ của mẫu và chống rung tay.\n"
                "- **ISO:** Giữ ở mức thấp nhất có thể `ISO 100 – 200` để da mẫu mịn màng, không bị vỡ hạt nhiễu.\n\n"
                "3. **Lấy nét & Ánh sáng:**\n"
                "- Luôn bật **Eye AF** (lấy nét tự động vào con ngươi mắt gần ống kính nhất).\n"
                "- Chọn góc ánh sáng hắt chéo 45 độ (*Rembrandt lighting*) để sống mũi cao và khuôn mặt thon gọn có chiều sâu. Bắt điểm sáng lung linh (*catchlight*) trong mắt mẫu."
            )

        # 6. SUNSET & GOLDEN HOUR
        if intent == "sunset":
            return (
                "### 🌅 Bí Quyết Chụp Ảnh Hoàng Hôn & Giờ Vàng (Golden Hour)\n\n"
                "Ánh sáng hoàng hôn mang sắc vàng ấm áp kỳ diệu nhưng thay đổi rất nhanh trong vòng 30 phút:\n\n"
                "1. **Thời điểm vàng:**\n"
                "- Bắt đầu chụp từ khoảng 45 phút trước khi mặt trời lặn hẳn. Lúc này mặt trời xuống thấp, ánh sáng mềm, đổ bóng xiên dài tôn dáng mẫu.\n\n"
                "2. **Cài đặt thông số máy ảnh:**\n"
                "- **Cân bằng trắng (White Balance):** Chuyển sang `Cloudy (Nhiều mây)` hoặc chỉnh độ K khoảng `6000K – 6500K` để ảnh giữ trọn sắc vàng cam rực rỡ, không bị máy ảnh tự khử vàng thành xám lạnh.\n"
                "- **Khẩu độ & Tốc độ:** Khẩu `f/2.8 – f/4.0`, ISO `100`, tốc độ `1/250s – 1/800s`.\n\n"
                "3. **Kỹ thuật chụp ngược sáng hoàng hôn:**\n"
                "- **Lấy rõ mặt mẫu:** Bắt buộc dùng tấm hắt sáng (*Reflector*) màu bạc/vàng hoặc đèn fill nhẹ để bù sáng vào mắt và khuôn mặt, tránh mặt bị đen thẫm.\n"
                "- **Chụp bóng đen nghệ thuật (*Silhouette*):** Đo sáng thẳng vào vùng trời hoàng hôn rực rỡ nhất, khóa sáng (AE Lock), để mẫu tạo dáng nghiêng có đường cong rõ ràng (váy xòe, tóc bay) in hình bóng đen sắc nét trên nền trời."
            )

        # 7. NIGHT & LONG EXPOSURE
        if intent == "night":
            return (
                "### 🌃 Kỹ Thuật Chụp Đêm & Phơi Sáng Dài (Long Exposure)\n\n"
                "Khi chụp ảnh trong điều kiện thiếu sáng hoặc cảnh đêm thành phố:\n\n"
                "1. **Chụp phong cảnh đêm & Vệt đèn xe (*Light Trails*):**\n"
                "- **Bắt buộc dùng chân máy (Tripod):** Tắt chống rung trên ống kính (IS/VR) khi đã cố định máy trên tripod.\n"
                "- **Khẩu độ:** Khép khẩu `f/8 – f/11` để tạo tia sao (*starburst*) lấp lánh từ các bóng đèn đường.\n"
                "- **ISO:** Giữ `ISO 100 – 200` để bầu trời đêm đen sâu, trong trẻo, không bị nhiễu hạt.\n"
                "- **Thời gian phơi sáng:** `10s – 25s`. Bật chế độ hẹn giờ chụp 2 giây (*Self-timer*) để máy không bị rung khi ngón tay bạn bấm nút chụp.\n\n"
                "2. **Chụp chân dung phố đêm (*Night Street Portrait*):**\n"
                "- Mở khẩu tối đa `f/1.4 – f/2.0` để gom sáng và biến ánh đèn xe cộ, biển hiệu thành những đốm bokeh tròn lung linh.\n"
                "- Tận dụng nguồn sáng có sẵn từ tủ kính cửa hàng, biển đèn LED neon rọi vào mặt mẫu. ISO `800 – 1600`, tốc độ `1/100s`."
            )

        # 8. LANDSCAPE & NATURE
        if intent == "landscape":
            return (
                "### 🏞️ Bí Quyết Nhiếp Ảnh Phong Cảnh & Thiên Nhiên Sắc Nét\n\n"
                "Để có những bức ảnh phong cảnh bao la, chiều sâu hùng vĩ:\n\n"
                "1. **Ống kính & Tiêu cự:**\n"
                "- Ống kính góc siêu rộng `16mm – 24mm` (Full-frame) để thu trọn không gian đất trời.\n"
                "- Hoặc ống kính tele `70-200mm` để nén các dãy núi trập trùng thành từng lớp mây mờ ảo.\n\n"
                "2. **Khẩu độ tối ưu (Sweet Spot):**\n"
                "- Cài đặt khẩu độ `f/8 – f/11`. Đây là dải khẩu độ sắc nét nhất của lens, giữ độ nét sâu từ tiền cảnh đến tận chân trời. Tránh khép quá `f/16` vì hiện tượng nhiễu xạ (*diffraction*) sẽ làm mờ ảnh.\n\n"
                "3. **Bố cục 3 lớp không gian (Foreground - Midground - Background):**\n"
                "- *Tiền cảnh:* Đặt một tảng đá rêu, bụi hoa dại hoặc con sóng vỡ ở 1/3 góc dưới để tạo điểm neo thị giác.\n"
                "- *Trung cảnh:* Mặt hồ, thung lũng, dòng sông uốn lượn.\n"
                "- *Hậu cảnh:* Bầu trời mây tích và rặng núi xa xôi.\n\n"
                "4. **Kính lọc hỗ trợ:** Dùng kính lọc CPL (Circular Polarizer) để loại bỏ ánh chói trên mặt nước/lá cây và giúp nền trời xanh ngắt."
            )

        # 9. STREET & ACTION
        if intent == "street":
            return (
                "### 🚶 Nhiếp Ảnh Đường Phố (Street Photography) & Bắt Khoảnh Khắc\n\n"
                "Nhiếp ảnh đường phố tôn vinh tính chân thực và khoảnh khắc quyết định (*The Decisive Moment*):\n\n"
                "1. **Ống kính nhỏ gọn:**\n"
                "- Dùng tiêu cự `35mm` hoặc `28mm` (như ống kính pancake). Kích thước nhỏ gọn giúp người đi đường không cảm thấy bị đe dọa hay gượng gạo.\n\n"
                "2. **Thiết lập thông số bắt dính chuyển động:**\n"
                "- **Chế độ:** Ưu tiên khẩu độ (Av/A) khép `f/4 – f/5.6` để vùng nét đủ dày, không bị trượt nét khi người di chuyển.\n"
                "- **Tốc độ:** Tối thiểu `1/250s – 1/500s` để 'đóng băng' người đi bộ hoặc xe đạp.\n"
                "- **ISO:** Để Auto ISO (giới hạn tối đa 3200) để máy linh hoạt theo ánh sáng đường phố.\n\n"
                "3. **Mẹo tiếp cận:** Chọn một góc ánh sáng đẹp (vệt nắng xiên qua hẻm phố) rồi đứng kiên nhẫn chờ chủ thể thú vị bước vào khung hình trước khi bấm máy!"
            )

        # 10. SMARTPHONE PHOTOGRAPHY
        if intent == "smartphone":
            return (
                "### 📱 5 Mẹo Chụp Ảnh Bằng Điện Thoại Đẹp Như Máy Ảnh Chuyên Nghiệp\n\n"
                "Bạn hoàn toàn có thể tạo nên những bức ảnh xuất sắc chỉ với chiếc smartphone của mình:\n\n"
                "1. **Lau sạch camera trước khi chụp:**\n"
                "- Dầu ngón tay và bụi bẩn bám trên camera là lý do chính khiến ảnh điện thoại bị mờ đục và lóa đèn ban đêm. Lau nhẹ bằng vạt áo sạch ảnh sẽ nét trong ngay lập tức!\n\n"
                "2. **Dùng camera 2X hoặc 3X cho ảnh chân dung:**\n"
                "- Camera chính 1X là góc rộng (tương đương 24-26mm), chụp gần sẽ khiến khuôn mặt bị bè to và mũi to ra. Chuyển sang ống kính tele 2X hoặc 3X (hoặc chế độ Chân Dung / Portrait Mode) để khuôn mặt thon gọn, thanh thoát và tự nhiên.\n\n"
                "3. **Chạm khóa nét và hạ thanh đo sáng:**\n"
                "- Chạm giữ vào mặt mẫu trên màn hình để bật `AE/AF Lock`, sau đó vuốt biểu tượng mặt trời xuống khoảng 10-15%. Việc này giúp bầu trời không bị cháy trắng và màu da trông mịn, sang trọng hơn.\n\n"
                "4. **Bật lưới bố cục (Grid 3x3):**\n"
                "- Vào Cài đặt máy ảnh điện thoại bật chế độ Lưới (*Grid*). Đặt đôi mắt mẫu ở 1/3 phía trên và giữ thẳng đường chân trời.\n\n"
                "5. **Tận dụng ánh sáng cửa sổ góc 45 độ:**\n"
                "- Không nên đứng dưới ánh đèn trần chiếu thẳng đỉnh đầu (sẽ tạo bóng thâm mắt). Hãy đứng nghiêng 45 độ so với cửa sổ có rèm trắng mỏng để đón ánh sáng khuếch tán dịu nhẹ."
            )

        # 11. COMPOSITION
        if intent == "composition":
            return (
                "### 📐 Nghệ Thuật Bố Cục Nhiếp Ảnh Thực Chiến\n\n"
                "Bố cục là ngôn ngữ dẫn dắt cảm xúc người xem vào bức ảnh:\n\n"
                "1. **Quy tắc 1/3 (Rule of Thirds):**\n"
                "- Chia khung hình thành 9 ô với 4 điểm giao nhau quyền lực. Đặt chủ thể hoặc đôi mắt nhân vật vào 1 trong 4 điểm giao này thay vì luôn đặt chính giữa, bức ảnh sẽ năng động và thoáng đãng hơn rất nhiều.\n\n"
                "2. **Đường dẫn thị giác (Leading Lines):**\n"
                "- Sử dụng con đường, lan can cầu thang, hàng cây thẳng tắp, hoặc bờ tường để dẫn mắt người xem tự nhiên hướng về nhân vật chính.\n\n"
                "3. **Khung trong khung (Frame within a Frame):**\n"
                "- Chụp xuyên qua khung cửa sổ, vòm cổng rêu phong, kẽ lá cây hoặc gương chiếu hậu để tạo cảm giác chiều sâu không gian nhiều tầng lớp.\n\n"
                "4. **Không gian thở (Negative Space):**\n"
                "- Dành khoảng trống rộng rãi theo hướng nhìn của nhân vật hoặc hướng chuyển động của chủ thể, mang lại cảm giác tự do, bình yên và giàu chất thơ."
            )

        # 12. COLOR HARMONIES
        if intent == "color":
            return (
                "### 🎨 Bảng Màu & Phối Màu Nhiếp Ảnh Kinh Điển\n\n"
                "Dưới đây là các bảng phối màu được các nhiếp ảnh gia và đạo diễn điện ảnh ưa chuộng nhất:\n\n"
                "1. **Cyberpunk Neon Sci-Fi:**\n"
                "- **Xanh Neon (Cyan):** `#00F0FF` phối hợp **Hồng Tím (Magenta):** `#FF007F`.\n"
                "- **Nền:** Xanh đen thẫm `#0B0E14` (đường nhựa ướt mưa).\n"
                "- **Điểm nhấn:** Vàng điện quang `#FFE600`. Tương phản cực đại mang lại cảm giác tương lai và công nghệ.\n\n"
                "2. **Tone Pastel Hàn Quốc (Trong trẻo, thanh xuân):**\n"
                "- **Hồng phấn (Baby Pink):** `#FFB6C1`\n"
                "- **Xanh bạc hà (Mint):** `#98FF98`\n"
                "- **Lam dịu (Soft Sky):** `#87CEEB`\n"
                "- **Tím hoa cà (Lavender):** `#E6E6FA`. Mang lại cảm giác nhẹ nhàng, ngây thơ và mộng mơ.\n\n"
                "3. **Vintage Film Kodak Hoài Niệm (Tone ấm 1970s):**\n"
                "- **Vàng nắng ấm:** `#E6A817` · **Cam đất:** `#C86432` · **Xanh rêu ô liu:** `#556B2F`.\n"
                "- Đặc trưng bởi dải màu ấm áp, độ tương phản dịu và hạt film grain tự nhiên.\n\n"
                "4. **Teal & Orange (Hollywood Cinematic Blockbuster):**\n"
                "- Tông màu da người (Orange ấm) tương phản trực tiếp với nền trời/biển (Teal xanh ngọc), khiến chủ thể luôn tách bật sống động khỏi cảnh nền.\n\n"
                "💡 **Quy tắc vàng 60 - 30 - 10:** 60% diện tích dành cho màu chủ đạo (background), 30% cho màu bổ trợ cân bằng, 10% dành cho màu điểm nhấn nổi bật nhất."
            )

        # 13. CLARITY AI STUDIO TOOLS
        if intent == "studio_tools":
            return (
                "### 🛠️ Hướng Dẫn Sử Dụng Các Tính Năng AI Trên Clarity Studio\n\n"
                "Clarity AI Studio tích hợp đầy đủ các công cụ xử lý ảnh chuyên sâu bằng trí tuệ nhân tạo:\n\n"
                "1. **Làm Nét Ảnh & Phục Hồi Chân Dung:**\n"
                "- Vào tab **Làm Nét Ảnh (Enhance)**.\n"
                "- *Ảnh người:* Chọn chế độ `⚡ Siêu Chân Dung (Real-ESRGAN + GFPGAN)`. GFPGAN sử dụng mạng nơ-ron nhận diện khuôn mặt để tái tạo từng sợi lông mi, con ngươi và chi tiết da tự nhiên mà không bị bết dính!\n"
                "- *Ảnh cũ, mờ, vỡ hạt:* Chọn `Real-ESRGAN Đa Dụng (x4plus)` hoặc `SCUNet Khử Nhiễu 1X`.\n"
                "- *Tranh vẽ 2D / Anime / Sticker:* Chọn `Siêu Nét Vector & Nét Vẽ 2D (AnimeSharp)` để nét mực đen sắc lẹm và da phẳng mịn.\n\n"
                "2. **Đổi Tỷ Lệ & Mở Rộng Ảnh Không Cắt Xén (Outpainting):**\n"
                "- Vào tab **Đổi Tỷ Lệ (Aspect)**.\n"
                "- Chọn tỷ lệ mong muốn (16:9 cho máy tính, 9:16 cho Story/Reels) và chọn chế độ `AI Sinh Không Gian (LaMa)` để mở rộng biên ảnh tự nhiên mà không bị méo hình.\n\n"
                "3. **Tách Nền Chuyên Nghiệp:**\n"
                "- Vào tab **Tách Nền (Background Remover)** với mô hình `BiRefNet` cắt từng sợi tóc tơ và viền áo cực kỳ mượt mà.\n\n"
                "4. **Nén Ảnh Thông Minh:**\n"
                "- Vào tab **Nén Ảnh (Compressor)**, chọn dung lượng MB mục tiêu để giảm kích thước tệp mà mắt thường không thấy suy giảm chất lượng.\n\n"
                "5. **Tìm Kiếm Ảnh Tương Tự (CLIP AI):**\n"
                "- Vào tab **Tìm Kiếm AI (Visual Search)** để tìm ảnh bằng từ khóa ngữ nghĩa hoặc tải ảnh lên để tìm các ảnh tương đồng bố cục, màu sắc."
            )

        # 14. FOLLOW-UP: APERTURE
        if intent == "follow_up_aperture":
            if prev_topic == "portrait":
                return (
                    "### 🔍 Khẩu Độ Khuyên Dùng Cho Ảnh Chân Dung\n\n"
                    "Đối với ảnh chân dung, bạn nên lựa chọn khẩu độ theo các trường hợp sau:\n\n"
                    "- **f/1.4 – f/1.8:** Mức khẩu độ lý tưởng nhất để xóa mờ hậu cảnh mịn màng (*creamy bokeh*), tách biệt mẫu và tạo các đốm sáng lung linh. *Lưu ý:* Vùng nét rất mỏng nên hãy lấy nét thật chuẩn vào mắt gần ống kính nhất.\n"
                    "- **f/2.2 – f/2.8:** Khuyên dùng khi chụp nửa người hoặc muốn toàn bộ khuôn mặt (từ mắt đến chóp mũi và tóc) nét sắc cạnh mà phông nền vẫn mờ ảo đẹp mắt.\n"
                    "- **f/4.0 trở lên:** Khi chụp đôi (2 người trở lên) để đảm bảo cả hai người đều nằm trọn trong vùng rõ nét."
                )
            elif prev_topic == "landscape":
                return (
                    "### 🔍 Khẩu Độ Khuyên Dùng Cho Ảnh Phong Cảnh\n\n"
                    "Đối với ảnh phong cảnh, bạn nên cài đặt khẩu độ ở mức **Sweet Spot f/8 – f/11**:\n\n"
                    "- Đây là dải khẩu độ sắc nét nhất của hầu hết các ống kính, giúp rõ nét sâu từ từng nhánh cỏ tiền cảnh đến rặng núi xa xôi.\n"
                    "- Tránh khép khẩu quá nhỏ như `f/16` hoặc `f/22` vì hiện tượng nhiễu xạ quang học (*diffraction*) sẽ làm mờ ảnh và giảm độ tương phản."
                )
            elif prev_topic == "night":
                return (
                    "### 🔍 Khẩu Độ Khuyên Dùng Khi Chụp Đêm\n\n"
                    "- **Chụp phong cảnh phơi sáng / Vệt đèn xe:** Khép khẩu `f/8 – f/11` kết hợp chân máy tripod để các bóng đèn đường tạo thành tia sao (*starburst*) lấp lánh và nét sâu toàn khung hình.\n"
                    "- **Chụp chân dung phố đêm:** Mở khẩu lớn `f/1.4 – f/2.0` để thu nhận nhiều ánh sáng nhất và biến ánh đèn neon thành đốm bokeh tròn mềm mại."
                )
            else:
                return (
                    "### 🔍 Hướng Dẫn Cài Đặt Khẩu Độ (Aperture) Cho Từng Thể Loại\n\n"
                    "- **Xóa phông chân dung:** `f/1.4 – f/2.8` (mở lớn để phông nền mờ ảo, làm nổi bật mẫu).\n"
                    "- **Chụp đời thường, đường phố:** `f/4.0 – f/5.6` (độ nét vừa đủ dày, bắt trọn khoảnh khắc).\n"
                    "- **Phong cảnh, kiến trúc:** `f/8 – f/11` (vùng ngọt ngào sắc nét toàn khung hình từ gần đến xa).\n"
                    "- **Phơi sáng tia sao ban đêm:** `f/8 – f/11` kết hợp chân máy."
                )

        # 15. FOLLOW-UP: LENS
        if intent == "follow_up_lens":
            if prev_topic == "portrait":
                return (
                    "### 🎯 Ống Kính (Lens) Tốt Nhất Cho Chân Dung\n\n"
                    "1. **85mm f/1.4 hoặc f/1.8:** Đứng số 1 về chụp chân dung ngoại cảnh. Nén hậu cảnh tuyệt đẹp, bokeh tròn mềm và khuôn mặt không bị méo.\n"
                    "2. **50mm f/1.8 (Ống kính quốc dân):** Tiêu cự đa dụng, chụp chân dung nửa người rất xuất sắc cả trong nhà lẫn ngoài trời.\n"
                    "3. **35mm f/1.4:** Dành cho chân dung môi trường (*environmental portrait*), lấy được không gian quán xá, phong cảnh xung quanh mẫu."
                )
            elif prev_topic == "landscape":
                return (
                    "### 🎯 Ống Kính (Lens) Tốt Nhất Cho Phong Cảnh\n\n"
                    "1. **16-35mm f/2.8 hoặc f/4:** Cho góc nhìn bao la, hùng vĩ.\n"
                    "2. **24-70mm f/2.8:** Đa dụng chụp từ góc rộng vừa phải đến chi tiết thiên nhiên.\n"
                    "3. **70-200mm:** Chụp nén các rặng núi, lớp sương mù và bình minh ở xa cực kỳ ấn tượng."
                )
            else:
                return (
                    "### 🎯 Tư Vấn Chọn Tiêu Cự Ống Kính (Lens)\n\n"
                    "- **Chân dung:** 85mm f/1.8 hoặc 50mm f/1.8 (tiêu chuẩn vàng xóa phông).\n"
                    "- **Đường phố, đời thường:** 35mm f/2 hoặc 28mm nhỏ gọn.\n"
                    "- **Phong cảnh:** 16-35mm góc siêu rộng.\n"
                    "- **Sự kiện, thể thao, chim muông:** 70-200mm tele."
                )

        # 16. FOLLOW-UP: MOBILE
        if intent == "follow_up_mobile":
            return (
                "### 📱 Mẹo Chụp Trên Điện Thoại Cho Tình Huống Này\n\n"
                "Nếu bạn chụp bằng smartphone:\n"
                "1. **Chụp chân dung:** Hãy chuyển sang camera tele **2X hoặc 3X** (hoặc chế độ Chân dung Portrait) thay vì chụp bằng camera 1X góc rộng để mặt mẫu không bị biến dạng to phình.\n"
                "2. **Kiểm soát đo sáng:** Nhấn giữ trên màn hình để khóa `AE/AF Lock`, sau đó vuốt nhẹ biểu tượng mặt trời xuống 1 chút để da không bị cháy sáng và màu ảnh trong hơn.\n"
                "3. **Ánh sáng:** Đứng nghiêng 45 độ so với cửa sổ hoặc nguồn sáng dịu, lau sạch camera trước khi bấm máy!"
            )

        # 17. FOLLOW-UP: DETAILS
        if intent == "follow_up_details":
            return (
                "### 💡 Chi Tiết Thêm & Mẹo Thực Hành\n\n"
                "Để đạt hiệu quả cao nhất trong thực tế:\n"
                "- **Chuẩn bị trước:** Luôn kiểm tra thẻ nhớ, pin và lau sạch mặt trước ống kính.\n"
                "- **Định dạng file:** Chụp file **RAW** (hoặc RAW+JPEG) để dễ dàng cứu sáng vùng tối (shadows) và kéo lại màu sắc khi chỉnh sửa.\n"
                "- **Tương tác với mẫu:** Trò chuyện tự nhiên, để mẫu cử động nhẹ nhàng thay vì đứng đơ tạo dáng tĩnh.\n"
                "- **Hậu kỳ với Clarity AI:** Bạn có thể đưa ảnh vào tab **Làm Nét Ảnh** để phục hồi chi tiết da mắt bằng GFPGAN hoặc mở rộng khung hình bằng LaMa!"
            )

        # 18. GENERAL OPEN-ENDED INQUIRY
        # Try local neural model if ready on CUDA
        if HAS_TORCH and self.device == "cuda":
            try:
                neural_reply = self._call_local_qwen(message, history=chat_history)
                if neural_reply and len(neural_reply.strip()) > 15:
                    return neural_reply.strip()
            except Exception:
                pass

        return (
            f"Về câu hỏi *\"{message}\"* của bạn:\n\n"
            f"Trong nhiếp ảnh và nghệ thuật thị giác, việc cân bằng giữa **ánh sáng, bố cục và màu sắc** luôn là chìa khóa tạo nên một tác phẩm ấn tượng:\n"
            f"- **Ánh sáng:** Luôn ưu tiên hướng sáng tự nhiên mềm mại (như ánh sáng cửa sổ hoặc giờ vàng hoàng hôn) để tạo khối và chiều sâu.\n"
            f"- **Bố cục:** Áp dụng quy tắc 1/3 hoặc đường dẫn thị giác để mắt người xem tự nhiên hướng về chủ thể chính.\n"
            f"- **Hậu kỳ:** Bạn có thể tận dụng các công cụ của **Clarity AI Studio** như làm nét chân dung (GFPGAN), khử nhiễu ảnh cũ (SCUNet), hoặc mở rộng khung hình (LaMa Outpainting) để tác phẩm hoàn hảo nhất.\n\n"
            f"Nếu bạn muốn tìm hiểu sâu hơn về thông số máy ảnh, cách chọn lens hay phối màu cho thể loại này, hãy nhắn cụ thể cho mình biết nhé!"
        )

    # ------------------------------------------------------------------------
    # 8. Local Neural LLM (Qwen 2.5 Instruct on PyTorch CUDA)
    # ------------------------------------------------------------------------

    def _call_local_qwen(
        self,
        prompt: str,
        system_prompt: str = "",
        history: Optional[List[Dict[str, str]]] = None
    ) -> Optional[str]:
        """Runs Qwen 2.5 Instruct directly on local GPU via PyTorch transformers with conversation memory."""
        if not HAS_TORCH or torch is None:
            return None
        try:
            if self.hf_model is None or self.hf_tokenizer is None:
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
                model_id = "Qwen/Qwen2.5-0.5B-Instruct"
                print(f"[LLMEngine] Loading local Qwen model ({model_id}) on {self.device}...")
                try:
                    self.hf_tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
                    self.hf_model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="cuda" if self.device == "cuda" else None,
                        local_files_only=True
                    )
                except Exception:
                    self.hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
                    self.hf_model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="cuda" if self.device == "cuda" else None
                    )
                if self.hf_model is not None:
                    self.hf_model.eval()

            if self.hf_tokenizer is None or self.hf_model is None:
                return None

            if not system_prompt:
                system_prompt = (
                    "Bạn là Trợ lý AI Cố vấn Chuyên sâu về Nhiếp ảnh, Màu sắc, Ánh sáng và Công nghệ Clarity AI Studio. "
                    "Hãy đối thoại tự nhiên, mượt mà, thân thiện, thông minh và súc tích bằng tiếng Việt chuẩn. "
                    "Quy tắc quan trọng:\n"
                    "1. Trả lời trực tiếp, đúng trọng tâm câu hỏi của người dùng và nhớ ngữ cảnh các câu trước đó.\n"
                    "2. Giọng điệu ấm áp, tự nhiên như một chuyên gia nhiếp ảnh nhiệt tình, tuyệt đối không trả lời máy móc hay rập khuôn.\n"
                    "3. Trình bày bằng định dạng Markdown rõ ràng, dễ nhìn (in đậm từ khóa, gạch đầu dòng ngắn gọn).\n"
                    "4. Tuyệt đối không dùng chữ tiếng Trung hay từ ngữ kỹ thuật vô nghĩa."
                )

            messages = [{"role": "system", "content": system_prompt}]

            # Append conversation history for true multi-turn context
            if history:
                for h in history[-6:]:
                    r = h.get("role", "user")
                    c = (h.get("content") or "").strip()
                    if c:
                        c_clean = c[:300] + "..." if len(c) > 300 else c
                        messages.append({"role": r, "content": c_clean})

            messages.append({"role": "user", "content": prompt})

            text = self.hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.hf_tokenizer([text], return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.hf_model.generate(
                    **inputs,
                    max_new_tokens=450,
                    do_sample=True,
                    temperature=0.65,
                    top_p=0.9,
                    repetition_penalty=1.15
                )
                generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                response = self.hf_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                response = response.strip()
                # Clean stray CJK characters that 0.5B occasionally introduces
                response = re.sub(r'[\u4e00-\u9fff]+', '', response).strip()
                return response if len(response) > 8 else None
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
        """Lazy loads BLIP captioner on-demand on CUDA/CPU for fast accurate image captioning."""
        if not HAS_TORCH or torch is None:
            return None

        try:
            # Patch safe load check for PyTorch 2.5 compatibility
            try:
                import transformers.modeling_utils
                transformers.modeling_utils.check_torch_load_is_safe = lambda: None
            except Exception:
                pass

            if self.blip_model is None or self.blip_processor is None:
                from transformers import BlipProcessor, BlipForConditionalGeneration  # type: ignore
                model_id = "Salesforce/blip-image-captioning-base"
                self.blip_processor = BlipProcessor.from_pretrained(model_id, use_fast=False, local_files_only=True)
                blip_inst: Any = BlipForConditionalGeneration.from_pretrained(
                    model_id,
                    dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    local_files_only=True
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
                out = self.blip_model.generate(**inputs, max_new_tokens=60)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)  # type: ignore
                return caption.strip()
        except Exception as e:
            print(f"[LLMEngine] BLIP caption error: {e}")
            return None
