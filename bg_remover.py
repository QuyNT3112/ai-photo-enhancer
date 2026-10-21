"""
AI Background Remover Engine (Next-Gen Multi-Model & Deep Clean)
Supports:
  - briaai/RMBG-1.4 (Salient Fast & Graphic / Sticker)
  - ZhengPeng7/BiRefNet (SOTA 2024/2025 High-Resolution Segmenter on CUDA)
  - U2Net (General Object Segmenter via rembg)
Features:
  - Deep Clean Mode (Border Background Flood Purge & Fog Zeroing)
  - Adjustable Sensitivity / Threshold
  - Edge De-Fringe & Color Spill Suppression
  - Anti-aliased Sub-pixel Matting
Accelerated with NVIDIA CUDA on RTX 3060.
"""

import os
import time
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from typing import Optional, Dict, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

class BackgroundRemover:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models: Dict[str, Any] = {}
        self.rmbg_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print(f"[BgRemover] Initialized on device: {self.device.upper()}")

    def _get_rmbg14(self):
        if 'rmbg-1.4' not in self.models:
            print("[BgRemover] Loading RMBG-1.4 model...")
            m = AutoModelForImageSegmentation.from_pretrained(
                'briaai/RMBG-1.4',
                trust_remote_code=True
            )
            m.to(self.device)
            m.eval()
            self.models['rmbg-1.4'] = m
            print("[BgRemover] RMBG-1.4 model loaded successfully!")
        return self.models['rmbg-1.4']

    def _get_birefnet(self):
        if 'birefnet' not in self.models:
            print("[BgRemover] Loading BiRefNet (SOTA) model...")
            m = AutoModelForImageSegmentation.from_pretrained(
                'ZhengPeng7/BiRefNet',
                trust_remote_code=True
            )
            m.to(self.device)
            m.eval()
            self.models['birefnet'] = m
            print("[BgRemover] BiRefNet loaded successfully!")
        return self.models['birefnet']

    def _get_u2net_session(self):
        if 'u2net' not in self.models:
            print("[BgRemover] Loading U2Net session...")
            import rembg
            self.models['u2net'] = rembg.new_session('u2net')
            print("[BgRemover] U2Net session loaded successfully!")
        return self.models['u2net']

    def _predict_raw_mask(self, model_name: str, rgb_pil: Image.Image) -> np.ndarray:
        """
        Runs model inference and returns a float32 mask [0.0, 1.0] matching rgb_pil (H, W).
        """
        orig_w, orig_h = rgb_pil.size
        model_key = model_name.strip().lower()

        if model_key in ['birefnet', 'biref']:
            model = self._get_birefnet()
            inp = self.rmbg_transform(rgb_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                res = model(inp)
            if isinstance(res, (list, tuple)):
                raw = res[-1].sigmoid().squeeze().cpu().numpy()
            else:
                raw = res.sigmoid().squeeze().cpu().numpy()
            mask = cv2.resize(raw, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            return np.clip(mask, 0.0, 1.0)

        elif model_key in ['u2net', 'rembg']:
            import rembg
            session = self._get_u2net_session()
            out_img = rembg.remove(rgb_pil, session=session)
            arr = np.array(out_img)
            if arr.shape[2] == 4:
                alpha = arr[:, :, 3].astype(np.float32) / 255.0
                return alpha
            return np.ones((orig_h, orig_w), dtype=np.float32)

        else:
            # Default to RMBG-1.4
            model = self._get_rmbg14()
            inp = self.rmbg_transform(rgb_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                preds = model(inp)
            raw = preds[0][0].squeeze().cpu().numpy()
            mask = cv2.resize(raw, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            return np.clip(mask, 0.0, 1.0)

    def _purge_border_background(self, img_bgr: np.ndarray, ai_mask: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
        """
        Detects if perimeter borders share a consistent background color (e.g. cream, white, studio gray).
        If so, traces connected background pixels via FloodFill and purges them completely from the AI mask.
        """
        h, w = img_bgr.shape[:2]
        strip = max(3, min(8, min(h, w) // 50))
        top = img_bgr[:strip, :]
        bottom = img_bgr[-strip:, :]
        left = img_bgr[:, :strip]
        right = img_bgr[:, -strip:]
        border_pixels = np.concatenate([
            top.reshape(-1, 3), bottom.reshape(-1, 3),
            left.reshape(-1, 3), right.reshape(-1, 3)
        ], axis=0)

        bg_mean = np.mean(border_pixels, axis=0)
        bg_std = np.std(border_pixels, axis=0)
        max_std = np.max(bg_std)

        # Uniform or subtle gradient background detected on borders
        if max_std < 40.0:
            # Distance threshold scaled with user sensitivity and border noise
            dist_thresh = 18.0 + (sensitivity * 30.0) + (max_std * 0.4)
            color_dist = np.linalg.norm(img_bgr.astype(np.float32) - bg_mean.astype(np.float32), axis=2)

            looks_like_bg = (color_dist < dist_thresh).astype(np.uint8)

            flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            flood_img = looks_like_bg.copy()

            step_x = max(1, w // 16)
            step_y = max(1, h // 16)
            seeds = []
            for x in range(0, w, step_x):
                seeds.append((x, 0))
                seeds.append((x, h - 1))
            for y in range(0, h, step_y):
                seeds.append((0, y))
                seeds.append((w - 1, y))

            for sx, sy in seeds:
                if flood_img[sy, sx] == 1:
                    cv2.floodFill(flood_img, flood_mask, (sx, sy), 2)

            connected_bg = (flood_img == 2).astype(np.float32)

            # Soften purge boundary slightly (3px) to prevent abrupt stair steps
            connected_bg_soft = cv2.GaussianBlur(connected_bg, (5, 5), 0)
            ai_mask = ai_mask * (1.0 - connected_bg_soft)

        return ai_mask

    def _defringe_colors(self, bgr: np.ndarray, alpha_norm: np.ndarray) -> np.ndarray:
        """
        Suppresses fringe / background color bleeding on the subject's outer edge
        by inpainting true interior foreground colors into semi-transparent boundary pixels.
        """
        solid_fg = (alpha_norm > 0.95).astype(np.uint8)
        boundary = np.logical_and(alpha_norm > 0.05, alpha_norm <= 0.95)

        if np.sum(solid_fg) == 0 or np.sum(boundary) == 0:
            return bgr

        inpaint_mask = (boundary.astype(np.uint8)) * 255
        clean_bgr = cv2.inpaint(bgr, inpaint_mask, 3, cv2.INPAINT_TELEA)
        result_bgr = bgr.copy()
        result_bgr[boundary] = clean_bgr[boundary]
        return result_bgr

    def remove_background(
        self,
        input_path: str,
        output_path: str,
        model_name: str = 'rmbg-1.4',
        clean_mode: str = 'deep_clean',
        sensitivity: float = 0.5,
        de_fringe: bool = True,
        bg_color: str = 'transparent',
        feather_edge: bool = True
    ) -> Dict[str, Any]:
        """
        High-performance background removal with AI + Advanced Deep Clean.

        Args:
            input_path: Input file path
            output_path: Output file path
            model_name: 'birefnet' (SOTA), 'rmbg-1.4' (Fast/Sticker), 'u2net'
            clean_mode: 'deep_clean' (aggressive clean), 'standard' (balanced), 'soft' (soft hair matting)
            sensitivity: 0.0 to 1.0 (aggressiveness of background purge)
            de_fringe: Whether to suppress border color contamination
            bg_color: 'transparent', 'white', 'black', 'green', or hex code '#RRGGBB'
            feather_edge: Sub-pixel boundary antialiasing

        Returns:
            Dictionary containing metadata and statistics.
        """
        start_time = time.time()
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        orig_img = Image.open(input_path)
        orig_w, orig_h = orig_img.size
        rgb_img = orig_img.convert("RGB")
        bgr_np = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)

        # 1. Model inference
        raw_mask = self._predict_raw_mask(model_name, rgb_img)

        # 2. Deep Clean & Border Background Purge
        clean_mode_str = clean_mode.strip().lower()
        if clean_mode_str in ['deep_clean', 'aggressive', 'deep']:
            # Purge contiguous border color
            raw_mask = self._purge_border_background(bgr_np, raw_mask, sensitivity=sensitivity)

            # High-contrast thresholding with clean cutoff
            # Eliminates misty/cloudy dirty background residuals
            cutoff = 0.12 + (sensitivity * 0.28)  # Range ~0.15 to ~0.40
            mask_np = np.clip((raw_mask - cutoff) / max(0.01, 1.0 - cutoff), 0.0, 1.0)
            if feather_edge:
                # S-curve smoothstep
                mask_np = mask_np * mask_np * (3.0 - 2.0 * mask_np)

        elif clean_mode_str in ['soft', 'hair']:
            # Gentle curve preserving low-opacity hair details
            cutoff = 0.05
            mask_np = np.clip((raw_mask - cutoff) / 0.95, 0.0, 1.0)
            if feather_edge:
                mask_np = np.power(mask_np, 0.9)

        else:
            # Standard mode
            cutoff = 0.15
            mask_np = np.clip((raw_mask - cutoff) / 0.85, 0.0, 1.0)
            if feather_edge:
                mask_np = mask_np * mask_np * (3.0 - 2.0 * mask_np)

        # 3. Optional De-Fringe on semi-transparent edges
        if de_fringe and clean_mode_str in ['deep_clean', 'standard']:
            bgr_np = self._defringe_colors(bgr_np, mask_np)

        rgb_clean = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
        fg_pil = Image.fromarray(rgb_clean)
        alpha_pil = Image.fromarray((mask_np * 255.0).clip(0, 255).astype(np.uint8))

        # 4. Handle Background Fill
        bg_color_clean = bg_color.strip().lower()
        if bg_color_clean in ['transparent', 'alpha', 'none', '']:
            final_img = fg_pil.convert("RGBA")
            final_img.putalpha(alpha_pil)
        else:
            # Parse color
            color_rgb = (255, 255, 255)
            if bg_color_clean in ['white', '#ffffff', '#fff']:
                color_rgb = (255, 255, 255)
            elif bg_color_clean in ['black', '#000000', '#000']:
                color_rgb = (0, 0, 0)
            elif bg_color_clean in ['green', '#00ff00', '#0f0']:
                color_rgb = (0, 255, 0)
            elif bg_color_clean.startswith('#') and len(bg_color_clean) == 7:
                try:
                    color_rgb = tuple(int(bg_color_clean[i:i+2], 16) for i in (1, 3, 5))
                except Exception:
                    color_rgb = (255, 255, 255)

            bg_layer = Image.new("RGBA", (orig_w, orig_h), color_rgb + (255,))
            fg_layer = fg_pil.convert("RGBA")
            fg_layer.putalpha(alpha_pil)
            final_img = Image.alpha_composite(bg_layer, fg_layer).convert("RGB")

        # 5. Save output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if bg_color_clean in ['transparent', 'alpha', 'none', '']:
            if not output_path.lower().endswith('.png'):
                output_path = os.path.splitext(output_path)[0] + '.png'
            final_img.save(output_path, format="PNG", optimize=True)
        else:
            final_img.save(output_path)

        duration = round(time.time() - start_time, 2)
        final_w, final_h = final_img.size

        if self.device == 'cuda':
            torch.cuda.empty_cache()

        model_labels = {
            'birefnet': 'BiRefNet AI (SOTA 2025)',
            'rmbg-1.4': 'RMBG-1.4 (Bria AI)',
            'u2net': 'U2Net AI (General)'
        }

        stats = {
            'duration_sec': duration,
            'orig_width': orig_w,
            'orig_height': orig_h,
            'orig_mp': round((orig_w * orig_h) / 1_000_000, 2),
            'final_width': final_w,
            'final_height': final_h,
            'final_mp': round((final_w * final_h) / 1_000_000, 2),
            'bg_color': bg_color_clean,
            'model_name': model_labels.get(model_name.lower(), model_name.upper()),
            'clean_mode': clean_mode_str,
            'sensitivity': sensitivity,
            'de_fringe': de_fringe
        }
        print(f"[BgRemover] Done in {duration}s [{model_name} / {clean_mode}]: {orig_w}x{orig_h} -> {output_path}")
        return stats
