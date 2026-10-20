"""
Aspect Ratio Converter Module
Converts landscape or portrait images to 1:1, 9:16, or 16:9.
Supports:
- AI Generative Outpaint (LaMa neural network synthesizes new environment/scenery to expand frame without cropping)
- Smart Center Crop (subject-focused proportional cropping)
- Ambient Blur Expand (studio-grade blurred background extension with drop shadow)
- Fit & Pad (clean padding with black/white/dominant color)
- Stretch Scale
"""

import os
import time
import cv2  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore


class AspectConverter:
    RATIO_SPECS = {
        '1:1': {'aspect_w': 1, 'aspect_h': 1, 'name': 'Vuông 1:1 (Square)', 'default_res': (1080, 1080)},
        '9:16': {'aspect_w': 9, 'aspect_h': 16, 'name': 'Dọc 9:16 (TikTok/Reels/Story)', 'default_res': (1080, 1920)},
        '16:9': {'aspect_w': 16, 'aspect_h': 9, 'name': 'Ngang 16:9 (YouTube/Desktop)', 'default_res': (1920, 1080)}
    }

    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.lama_model = None

    def _load_lama(self):
        """Lazy load LaMa outpainting model."""
        if self.lama_model is not None:
            return self.lama_model

        model_path = os.path.join(os.path.dirname(__file__), 'models', 'big-lama.pt')
        if not os.path.exists(model_path):
            print(f"[AspectConverter] Downloading LaMa Outpaint model to {model_path}...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            import urllib.request
            url = 'https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt'
            urllib.request.urlretrieve(url, model_path)
            print(f"[AspectConverter] LaMa model downloaded successfully!")

        print(f"[AspectConverter] Loading LaMa Outpaint model on {self.device}...")
        self.lama_model = torch.jit.load(model_path, map_location=self.device).eval()
        print(f"[AspectConverter] LaMa Outpaint model loaded and ready!")
        return self.lama_model

    def convert(self, input_path, output_path, target_ratio='1:1', mode='ai_expand', bg_color='black', resolution_mode='standard', align='center'):
        """
        Convert aspect ratio of input image.
        
        Args:
            input_path: Path to input image
            output_path: Path to save result
            target_ratio: '1:1', '9:16', or '16:9'
            mode: 'ai_expand' (AI Sinh Không Gian), 'smart_crop' (Fill/Tràn viền), 'blur_expand', 'pad_color', 'stretch'
            bg_color: 'black', 'white', 'auto' (dominant color)
            resolution_mode: 'standard' (1080p), 'high' (2K/4K), or 'match_original'
            align: 'center', 'top', 'bottom', 'left', 'right'
        """
        t0 = time.time()
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read image from {input_path}")

        orig_h, orig_w = img.shape[:2]
        has_alpha = (len(img.shape) == 3 and img.shape[2] == 4)
        
        # Separate alpha if present
        if has_alpha:
            img = img[:, :, :3]

        if target_ratio not in self.RATIO_SPECS:
            target_ratio = '1:1'

        ratio_info = self.RATIO_SPECS[target_ratio]

        # Determine target width & height
        if resolution_mode == 'high':
            # 2K/4K target
            if target_ratio == '1:1':
                tw, th = 2160, 2160
            elif target_ratio == '9:16':
                tw, th = 1440, 2560
            else:  # 16:9
                tw, th = 2560, 1440
        elif resolution_mode == 'match_original':
            # Match largest dimension of original image
            max_dim = max(orig_w, orig_h)
            if target_ratio == '1:1':
                tw, th = max_dim, max_dim
            elif target_ratio == '9:16':
                th = max_dim
                tw = int(th * (9 / 16))
            else:  # 16:9
                tw = max_dim
                th = int(tw * (9 / 16))
        else:  # 'standard'
            tw, th = ratio_info['default_res']

        result = None

        if mode in ['ai_expand', 'ai_outpaint', 'generative']:
            # AI Generative Outpaint: generate new context without cropping
            try:
                result = self._ai_expand(img, tw, th)
            except Exception as e:
                print(f"[AspectConverter] Error in AI outpainting: {e}, falling back to smart_crop")
                result = self._crop_to_aspect(img, tw, th, align=align)
        elif mode in ['smart_crop', 'fill']:
            # Zoom & Fill entire frame with zero blurred borders
            result = self._crop_to_aspect(img, tw, th, align=align)
        elif mode == 'pad_color':
            result = self._pad_color(img, tw, th, bg_color)
        elif mode == 'stretch':
            result = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LANCZOS4)
        else:  # 'blur_expand'
            result = self._blur_expand(img, tw, th)

        # Save output image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result)

        duration = round(time.time() - t0, 3)
        final_h, final_w = result.shape[:2]

        return {
            'duration_sec': duration,
            'orig_width': orig_w,
            'orig_height': orig_h,
            'orig_ratio': f"{round(orig_w/orig_h, 2)}:1",
            'final_width': final_w,
            'final_height': final_h,
            'target_ratio': target_ratio,
            'ratio_name': ratio_info['name'],
            'mode': mode,
            'file_size_kb': round(os.path.getsize(output_path) / 1024, 1)
        }

    def _ai_expand(self, img, target_w, target_h, step_size=64):
        """
        AI Generative Outpaint: Iterative multi-step expansion
        to naturally extend window beams, sky, and environmental structures
        without receptive field decay into gray.
        """
        model = self._load_lama()
        orig_h, orig_w = img.shape[:2]

        scale = min(target_w / orig_w, target_h / orig_h)
        fg_w = int(orig_w * scale)
        fg_h = int(orig_h * scale)
        fg_resized = cv2.resize(img, (fg_w, fg_h), interpolation=cv2.INTER_LANCZOS4)
        curr = fg_resized.copy()

        target_w_aligned = ((target_w + 7) // 8) * 8
        target_h_aligned = ((target_h + 7) // 8) * 8

        while curr.shape[0] < target_h_aligned or curr.shape[1] < target_w_aligned:
            curr_h, curr_w = curr.shape[:2]
            raw_next_w = min(curr_w + step_size * 2, target_w_aligned)
            raw_next_h = min(curr_h + step_size * 2, target_h_aligned)

            next_w = ((raw_next_w + 7) // 8) * 8
            next_h = ((raw_next_h + 7) // 8) * 8

            pad_top = (next_h - curr_h) // 2
            pad_bot = next_h - curr_h - pad_top
            pad_left = (next_w - curr_w) // 2
            pad_right = next_w - curr_w - pad_left

            if pad_top == 0 and pad_bot == 0 and pad_left == 0 and pad_right == 0:
                break

            canvas = cv2.copyMakeBorder(curr, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_REPLICATE)
            canvas = canvas[:next_h, :next_w]

            mask = np.zeros((next_h, next_w), dtype=np.uint8)
            if pad_top > 0: mask[:pad_top, :] = 255
            if pad_bot > 0: mask[next_h - pad_bot:, :] = 255
            if pad_left > 0: mask[:, :pad_left] = 255
            if pad_right > 0: mask[:, next_w - pad_right:] = 255

            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)

            img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
            mask_t = torch.from_numpy(dilated_mask).unsqueeze(0).unsqueeze(0).float().to(self.device) / 255.0

            with torch.no_grad():
                out = model(img_t, mask_t)

            out_np = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            curr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)[:next_h, :next_w]

        curr = cv2.resize(curr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        offset_x = (target_w - fg_w) // 2
        offset_y = (target_h - fg_h) // 2
        inner_mask = np.ones((target_h, target_w), dtype=np.float32)
        inner_mask[offset_y:offset_y + fg_h, offset_x:offset_x + fg_w] = 0.0
        feather = cv2.GaussianBlur(inner_mask, (5, 5), 1.5)
        feather_3c = np.stack([feather] * 3, axis=-1)

        sharp_canvas = curr.copy()
        sharp_canvas[offset_y:offset_y + fg_h, offset_x:offset_x + fg_w] = fg_resized

        result = (sharp_canvas.astype(np.float32) * (1.0 - feather_3c) + curr.astype(np.float32) * feather_3c).clip(0, 255).astype(np.uint8)
        return result

    def _blur_expand(self, img, target_w, target_h):
        """Ambient blur extension: blur the zoomed background and place crisp foreground on top."""
        orig_h, orig_w = img.shape[:2]

        bg_scale = max(target_w / orig_w, target_h / orig_h)
        bg_w = int(orig_w * bg_scale)
        bg_h = int(orig_h * bg_scale)
        bg_resized = cv2.resize(img, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)

        start_x = (bg_w - target_w) // 2
        start_y = (bg_h - target_h) // 2
        bg_canvas = bg_resized[start_y:start_y + target_h, start_x:start_x + target_w].copy()

        ksize = 101
        bg_canvas = cv2.GaussianBlur(bg_canvas, (ksize, ksize), 45)
        bg_canvas = (bg_canvas.astype(np.float32) * 0.72).clip(0, 255).astype(np.uint8)

        fg_scale = min(target_w / orig_w, target_h / orig_h)
        fg_w = int(orig_w * fg_scale)
        fg_h = int(orig_h * fg_scale)
        fg_resized = cv2.resize(img, (fg_w, fg_h), interpolation=cv2.INTER_LANCZOS4)

        offset_x = (target_w - fg_w) // 2
        offset_y = (target_h - fg_h) // 2

        shadow_mask = np.zeros((target_h, target_w), dtype=np.float32)
        pad = 8
        sy1 = max(0, offset_y - pad)
        sy2 = min(target_h, offset_y + fg_h + pad)
        sx1 = max(0, offset_x - pad)
        sx2 = min(target_w, offset_x + fg_w + pad)
        shadow_mask[sy1:sy2, sx1:sx2] = 0.4
        shadow_mask = cv2.GaussianBlur(shadow_mask, (31, 31), 10)

        for c in range(3):
            bg_canvas[:, :, c] = (bg_canvas[:, :, c].astype(np.float32) * (1.0 - shadow_mask)).clip(0, 255).astype(np.uint8)

        bg_canvas[offset_y:offset_y + fg_h, offset_x:offset_x + fg_w] = fg_resized
        return bg_canvas

    def _crop_to_aspect(self, img, target_w, target_h, align='center'):
        """Smart proportional crop to fill the entire target canvas (Zoom & Fill)."""
        orig_h, orig_w = img.shape[:2]
        target_aspect = target_w / target_h
        current_aspect = orig_w / orig_h

        if current_aspect > target_aspect:
            new_w = int(orig_h * target_aspect)
            if align == 'left':
                start_x = 0
            elif align == 'right':
                start_x = max(0, orig_w - new_w)
            else:
                start_x = (orig_w - new_w) // 2
            cropped = img[:, start_x:start_x + new_w]
        else:
            new_h = int(orig_w / target_aspect)
            if align == 'top':
                start_y = 0
            elif align == 'bottom':
                start_y = max(0, orig_h - new_h)
            else:
                start_y = (orig_h - new_h) // 2
            cropped = img[start_y:start_y + new_h, :]

        return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    def _pad_color(self, img, target_w, target_h, bg_color='black'):
        """Fit image inside target dimensions and pad borders with solid color."""
        orig_h, orig_w = img.shape[:2]
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        fg_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        if bg_color == 'white':
            canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        elif bg_color == 'auto':
            border_pixels = np.concatenate([img[0, :, :], img[-1, :, :], img[:, 0, :], img[:, -1, :]], axis=0)
            avg_color = border_pixels.mean(axis=0).astype(np.uint8)
            canvas = np.full((target_h, target_w, 3), avg_color, dtype=np.uint8)
        else:
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = fg_resized
        return canvas
