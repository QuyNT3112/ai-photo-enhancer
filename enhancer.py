"""
AI Image Enhancer Engine
Utilizes Real-ESRGAN and GFPGAN with CUDA acceleration.
Tuned for NVIDIA RTX 3060 GPU with 12GB VRAM.
"""

import os
import sys
import time
import torch  # type: ignore
import cv2  # type: ignore
import numpy as np  # type: ignore
from typing import Any
from PIL import Image  # type: ignore

# Import Real-ESRGAN
from realesrgan import RealESRGANer  # type: ignore
from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
from realesrgan.archs.srvgg_arch import SRVGGNetCompact  # type: ignore

# Import GFPGAN
GFPGANer: Any = None
try:
    from gfpgan import GFPGANer as _GFPGANer  # type: ignore
    GFPGANer = _GFPGANer
    HAS_GFPGAN = True
except Exception:
    HAS_GFPGAN = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Model file paths
MODEL_PATHS = {
    'general': os.path.join(MODELS_DIR, 'RealESRGAN_x4plus.pth'),
    'anime': os.path.join(MODELS_DIR, 'RealESRGAN_x4plus_anime_6B.pth'),
    'anime_v3': os.path.join(MODELS_DIR, 'realesr-animevideov3.pth'),
    'face': os.path.join(MODELS_DIR, 'GFPGANv1.4.pth'),
    'scunet': os.path.join(MODELS_DIR, 'scunet_color_real_gan.pth')
}


class ImageEnhancer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.half = (self.device == 'cuda')  # fp16 for speed and memory efficiency on RTX 3060
        self.cached_models = {}
        print(f"[Enhancer] Initialized on device: {self.device.upper()} (FP16: {self.half})")

    def get_system_info(self):
        info = {
            'device': self.device,
            'gpu_name': torch.cuda.get_device_name(0) if self.device == 'cuda' else 'CPU',
            'vram_total_gb': round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if self.device == 'cuda' else 0,
            'vram_allocated_gb': round(torch.cuda.memory_allocated(0) / (1024**3), 2) if self.device == 'cuda' else 0,
            'vram_reserved_gb': round(torch.cuda.memory_reserved(0) / (1024**3), 2) if self.device == 'cuda' else 0,
            'models_available': {
                'realesrgan_general': os.path.exists(MODEL_PATHS['general']),
                'realesrgan_anime': os.path.exists(MODEL_PATHS['anime']),
                'gfpgan_face': os.path.exists(MODEL_PATHS['face']) and HAS_GFPGAN,
                'scunet_real': os.path.exists(MODEL_PATHS['scunet']),
            }
        }
        return info

    def _load_realesrgan_general(self):
        if 'general' not in self.cached_models:
            print("[Enhancer] Loading RealESRGAN_x4plus model...")
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(
                scale=4,
                model_path=MODEL_PATHS['general'],
                model=model,
                tile=0,  # 0 disables tiling on RTX 3060 12GB to completely avoid tile cut seam artifacts
                tile_pad=10,
                pre_pad=0,
                half=self.half,
                device=self.device
            )
            self.cached_models['general'] = upsampler
            print("[Enhancer] RealESRGAN_x4plus loaded!")
        return self.cached_models['general']

    def _load_realesrgan_anime(self):
        if 'anime' not in self.cached_models:
            print("[Enhancer] Loading RealESRGAN_x4plus_anime_6B model (tile=0)...")
            # RealESRGAN_x4plus_anime_6B uses RRDBNet with 6 blocks
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(
                scale=4,
                model_path=MODEL_PATHS['anime'],
                model=model,
                tile=0,  # 0 disables tiling on RTX 3060 to completely eliminate seam lines
                tile_pad=10,
                pre_pad=0,
                half=self.half,
                device=self.device
            )
            self.cached_models['anime'] = upsampler
            print("[Enhancer] RealESRGAN_x4plus_anime_6B loaded with tile=0!")
        return self.cached_models['anime']

    def _reconstruct_tangent_flow(self, img_bgr, scale_ref=1.0):
        """
        Directional Tangent Flow & Contour Regularization:
        - Analyzes the local stroke gradient orientation field (tangent angles).
        - Smooths strictly ALONG the stroke curvature direction to heal pixel staircasing,
          aliasing fractures, micro-holes, and edge jitter ("tái tạo & chống vỡ nét").
        - Preserves perpendicular gradient sharpness for crisp, silky-smooth vector outlines.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 1. Gradient & Tangent vector field
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)

        denom = mag + 1e-5
        tx = -gy / denom
        ty = gx / denom

        # 2. Tangent-flow resampling along the contour
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        step = 0.75 * scale_ref
        offsets = [-2.0 * step, -step, 0.0, step, 2.0 * step]
        kernel_weights = [0.12, 0.25, 0.26, 0.25, 0.12]

        smoothed = np.zeros_like(img_bgr, dtype=np.float32)
        weights = np.zeros((h, w, 1), dtype=np.float32)

        for offset, kw in zip(offsets, kernel_weights):
            sx = np.clip(x_coords + offset * tx, 0, w - 1)
            sy = np.clip(y_coords + offset * ty, 0, h - 1)
            sample = cv2.remap(img_bgr, sx, sy, cv2.INTER_LINEAR)
            smoothed += sample.astype(np.float32) * kw
            weights += kw

        flow_bgr = smoothed / weights

        # 3. Apply only on line strokes to protect flat regions
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        is_stroke_edge = (mag > 32) & ((V < 110) | ((V < 220) & (S < 48)))
        stroke_mask = cv2.GaussianBlur(is_stroke_edge.astype(np.float32), (0, 0), 0.8 * scale_ref)[:, :, None]

        flow_reconstructed = img_bgr.astype(np.float32) * (1.0 - stroke_mask) + flow_bgr * stroke_mask
        return np.clip(flow_reconstructed, 0, 255).astype(np.uint8)

    def _clarify_vector_lineart(self, img_bgr):
        """
        Adaptive Multi-Tier Vector Cel Line Art Engine:
        - Tangent Flow Reconstruction: heals aliasing stair-steps, micro-holes, and broken contours.
        - Multi-tier Black Top-Hat: micro-tier (3x3) for fine lines (teeth gaps, squiggly text,
          eyelids) and main-tier (5x5, 7x7) for character silhouettes and bold contours.
        - Hybrid Color/Lightness Protection: combines HSV (S, V) and LAB Chroma to strictly
          protect vibrant colored fills (tongue, clothing, eyes, warm fur) while enabling
          deep inking on genuine line strokes.
        - Vector Inking: solidifies strokes towards rich black ink without fattening fine text.
        - Continuous Anti-aliased Acutance: smoothly blends contour edge sharpening across
          stroke boundaries without introducing step cutoffs or flat-fill grain.
        """
        h, w = img_bgr.shape[:2]
        scale_ref = max(1.0, min(h, w) / 512.0)

        # Step 0: Reconstruct and heal broken/aliased contour lines along tangent flow
        img_bgr = self._reconstruct_tangent_flow(img_bgr, scale_ref=scale_ref)

        k_micro_sz = max(3, int(round(3 * scale_ref))) | 1
        k_mid_sz = max(5, int(round(5 * scale_ref))) | 1
        k_out_sz = max(7, int(round(7 * scale_ref))) | 1

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Multi-scale Black Top-Hat extraction
        k_micro = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_micro_sz, k_micro_sz))
        th_micro = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_micro)

        k_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_mid_sz, k_mid_sz))
        k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_out_sz, k_out_sz))
        th_mid = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_mid)
        th_out = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_out)
        th_main = np.maximum(th_mid, th_out)

        # 2. Hybrid Color / Lightness gating
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        chroma = np.sqrt((lab[:, :, 1].astype(float) - 128)**2 + (lab[:, :, 2].astype(float) - 128)**2)

        # Dark lines (V < 85) are always ink strokes; lighter strokes must be neutral (S < 45 or chroma < 38)
        is_line_allowed = (V < 85) | ((V < 225) & ((S < 45) | (chroma < 38)))

        # 3. Two-tier response maps
        # Tier 1 (micro-detail): low activation threshold (-3.5), gentle ramp, zero line bloating
        micro_strength = np.clip((th_micro.astype(float) - 3.5) / 20.0, 0, 1.0) * is_line_allowed.astype(float)
        # Tier 2 (main outline): solid outline detection (-5.0 / 25.0)
        main_strength = np.clip((th_main.astype(float) - 5.0) / 25.0, 0, 1.0) * is_line_allowed.astype(float)

        effective_line = np.maximum(main_strength, micro_strength)

        # 4. Inking vector solidification
        # Main outlines get bold inking (0.85); micro-details get controlled inking (0.65) to prevent clumping
        inking_factor = (main_strength * 0.85 + micro_strength * (1.0 - main_strength) * 0.65)[:, :, None]
        darkened = img_bgr.astype(float) * (1.0 - inking_factor * 0.90)
        darkened = np.clip(darkened, 0, 255).astype(np.uint8)

        # 5. Continuous anti-aliased edge sharpening
        blur = cv2.GaussianBlur(darkened, (0, 0), 0.9 * scale_ref)
        sharp = cv2.addWeighted(darkened, 1.40, blur, -0.40, 0)

        # Smooth continuous blending over [0.04, 0.20] guarantees zero noise in flat fills and no banding
        blend_weight = np.clip((effective_line - 0.04) / 0.16, 0.0, 1.0)[:, :, None]
        final_bgr = (darkened.astype(float) * (1.0 - blend_weight) + sharp.astype(float) * blend_weight)
        return np.clip(final_bgr, 0, 255).astype(np.uint8)

    def _load_gfpgan(self, upscale=2):
        key = f'gfpgan_{upscale}'
        if key not in self.cached_models:
            if not HAS_GFPGAN or GFPGANer is None or not os.path.exists(MODEL_PATHS['face']):
                raise RuntimeError("GFPGAN library or model weights not available.")
            print(f"[Enhancer] Loading GFPGANv1.4 model (scale {upscale})...")
            face_restorer = GFPGANer(
                model_path=MODEL_PATHS['face'],
                upscale=upscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device
            )
            self.cached_models[key] = face_restorer
            print(f"[Enhancer] GFPGANv1.4 loaded (scale {upscale})!")
        return self.cached_models[key]

    def _load_scunet(self):
        if 'scunet' not in self.cached_models:
            path = MODEL_PATHS['scunet']
            if not os.path.exists(path):
                print("[Enhancer] Downloading SCUNet 1X restoration model (~31MB)...")
                import urllib.request
                url = 'https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth'
                os.makedirs(os.path.dirname(path), exist_ok=True)
                urllib.request.urlretrieve(url, path)
                print("[Enhancer] SCUNet downloaded!")
            import spandrel  # type: ignore
            model = spandrel.ModelLoader().load_from_file(path).to(self.device).eval()
            self.cached_models['scunet'] = model
            print(f"[Enhancer] SCUNet 1X model loaded on {self.device.upper()}!")
        return self.cached_models['scunet']

    def _enhance_scunet(self, img_bgr):
        model = self._load_scunet()
        h, w = img_bgr.shape[:2]

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)

        tile_size = 512
        tile_pad = 32

        if h <= tile_size and w <= tile_size:
            with torch.no_grad():
                out_tensor = model(tensor)
        else:
            out_tensor = torch.zeros_like(tensor)
            weights = torch.zeros((1, 1, h, w), device=self.device)
            stride = tile_size - 2 * tile_pad

            y_starts = list(range(0, max(1, h - tile_size + 1), stride))
            if y_starts[-1] + tile_size < h:
                y_starts.append(h - tile_size)
            x_starts = list(range(0, max(1, w - tile_size + 1), stride))
            if x_starts[-1] + tile_size < w:
                x_starts.append(w - tile_size)

            mask = torch.ones((1, 1, tile_size, tile_size), device=self.device)
            for y in y_starts:
                for x in x_starts:
                    tile = tensor[:, :, y:y+tile_size, x:x+tile_size]
                    with torch.no_grad():
                        out_tile = model(tile)
                    out_tensor[:, :, y:y+tile_size, x:x+tile_size] += out_tile
                    weights[:, :, y:y+tile_size, x:x+tile_size] += mask
            out_tensor = out_tensor / weights

        res_rgb = (out_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        res_bgr = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)
        return res_bgr

    def enhance_image(self, input_path, output_path, model_type='general', outscale=4, face_enhance=False):
        """
        Enhance an image using the selected model.
        
        Args:
            input_path: Path to input image
            output_path: Path to save enhanced image
            model_type: 'general', 'anime', 'face_only', or 'combo'
            outscale: Desired upscale factor (usually 2 or 4)
            face_enhance: Whether to apply GFPGAN face restoration on top
        """
        start_time = time.time()
        
        # Read image with OpenCV (handles BGR and Alpha)
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image from {input_path}")
            
        orig_h, orig_w = img.shape[:2]
        has_alpha = (len(img.shape) == 3 and img.shape[2] == 4)
        
        alpha_channel = None
        if has_alpha:
            alpha_channel = img[:, :, 3]
            bgr_channel = img[:, :, :3]
            # Replace dark transparent pixels with clean white background to prevent dark halos
            clean_bgr = bgr_channel.copy()
            clean_bgr[alpha_channel < 15] = [255, 255, 255]
            img = clean_bgr
            
        print(f"[Enhancer] Processing {os.path.basename(input_path)}: {orig_w}x{orig_h} with model '{model_type}', scale={outscale}, face_enhance={face_enhance}")

        output_img = None

        if model_type == 'scunet':
            # SCUNet Native 1X Blind Restoration (zero dimension change)
            restored = self._enhance_scunet(img)
            if face_enhance and HAS_GFPGAN and os.path.exists(MODEL_PATHS['face']):
                restorer = self._load_gfpgan(upscale=1)
                _, _, output_img = restorer.enhance(
                    restored,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True
                )
            else:
                output_img = restored

        elif model_type == 'face_only':
            # Run GFPGAN directly
            restorer = self._load_gfpgan(upscale=outscale)
            # GFPGAN expects BGR
            cropped_faces, restored_faces, restored_img = restorer.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
            output_img = restored_img

        elif model_type in ['anime', 'animesharp']:
            # Run clean RealESRGAN Anime 6B model with tile=0 (silky smooth, zero noise)
            upsampler = self._load_realesrgan_anime()
            out_4x, _ = upsampler.enhance(img, outscale=4)

            # Preserve crisp high-frequency line art using Lanczos-4 downsampling when scaling down
            if outscale == 1:
                target_img = cv2.resize(out_4x, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
            elif outscale != 4:
                target_w = max(1, int(orig_w * outscale))
                target_h = max(1, int(orig_h * outscale))
                target_img = cv2.resize(out_4x, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                target_img = out_4x

            # Apply Vector Cel Line Art Clarification (deepen & sharpen outlines without any noise)
            output_img = self._clarify_vector_lineart(target_img)

            if face_enhance and HAS_GFPGAN and os.path.exists(MODEL_PATHS['face']):
                restorer = self._load_gfpgan(upscale=1)
                _, _, output_img = restorer.enhance(output_img, has_aligned=False, only_center_face=False, paste_back=True)

        elif model_type == 'combo':
            # Combo: Real-ESRGAN General + GFPGAN Face Restoration
            upsampler = self._load_realesrgan_general()
            enhanced_bg, _ = upsampler.enhance(img, outscale=outscale)
            
            if HAS_GFPGAN and os.path.exists(MODEL_PATHS['face']):
                restorer = self._load_gfpgan(upscale=1)
                _, _, output_img = restorer.enhance(
                    enhanced_bg,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True
                )
            else:
                output_img = enhanced_bg

        else:  # default 'general'
            upsampler = self._load_realesrgan_general()
            output_img, _ = upsampler.enhance(img, outscale=outscale)
            
            if face_enhance and HAS_GFPGAN and os.path.exists(MODEL_PATHS['face']):
                restorer = self._load_gfpgan(upscale=1)
                _, _, output_img = restorer.enhance(
                    output_img,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True
                )

        # Ensure 100% exact original dimensions if outscale=1 was requested or if using SCUNet
        if (outscale == 1 or model_type == 'scunet') and (output_img.shape[0] != orig_h or output_img.shape[1] != orig_w):
            output_img = cv2.resize(output_img, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

        # Subtle adaptive edge clarity boost for 1X scale (non-anime only, anime is already clarified)
        if outscale == 1 and model_type not in ['anime', 'animesharp']:
            try:
                bgr_part = output_img[:, :, :3]
                blur = cv2.GaussianBlur(bgr_part, (0, 0), 1.2)
                output_img = cv2.addWeighted(bgr_part, 1.22, blur, -0.22, 0)
            except Exception as e:
                print(f"[Enhancer] Warning in edge boost: {e}")

        # Handle alpha channel restore if original had transparency
        if has_alpha and alpha_channel is not None:
            new_h, new_w = output_img.shape[:2]
            if new_h == orig_h and new_w == orig_w:
                resized_alpha = alpha_channel.copy()
            else:
                resized_alpha = cv2.resize(alpha_channel, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            # Protect dark line art / strokes from alpha erosion in transparent stickers
            if model_type in ['anime', 'animesharp']:
                gray = cv2.cvtColor(output_img[:, :, :3], cv2.COLOR_BGR2GRAY)
                dark_lines = (gray < 90) & (resized_alpha > 25)
                resized_alpha[dark_lines] = np.maximum(resized_alpha[dark_lines], 250)
            output_img = cv2.merge([output_img[:, :, 0], output_img[:, :, 1], output_img[:, :, 2], np.clip(resized_alpha, 0, 255).astype(np.uint8)])

        # Save output image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output_img)
        
        # Clean GPU memory cache
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        duration = time.time() - start_time
        final_h, final_w = output_img.shape[:2]

        stats = {
            'duration_sec': round(duration, 2),
            'orig_width': orig_w,
            'orig_height': orig_h,
            'orig_mp': round((orig_w * orig_h) / 1_000_000, 2),
            'final_width': final_w,
            'final_height': final_h,
            'final_mp': round((final_w * final_h) / 1_000_000, 2),
            'model_type': model_type,
            'outscale': outscale,
            'face_enhance': face_enhance
        }
        print(f"[Enhancer] Done in {duration:.2f}s: {orig_w}x{orig_h} -> {final_w}x{final_h} ({stats['final_mp']} MP)")
        return stats
