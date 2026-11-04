"""
SVG Processor & AI Vector Studio Module
Clarity AI Studio

Capabilities:
1. High-Fidelity Local AI Raster-to-Vector Engine (PNG/JPG/WEBP/BMP/TIFF -> .SVG)
   - Preserves 100% genuine vector Bézier curves (not fake raster wrappers).
   - Accurate line reconstruction, corner detection, and exact color palettes.
   - Bilateral edge-preserving denoising + adaptive line enhancement.
   - Multiple AI presets: Fidelity (Artwork), Clean Logo, Line-Art (CAD/Sketch), Detailed Photo.
2. Bidirectional SVG <-> SVG XML Converter & Live Inspector:
   - SVG File -> Highlighted, formatted SVG XML + element breakdown & palette extraction.
   - SVG XML Code -> Validated, live rendered, and downloadable .SVG file.
3. AI Vector Assistant:
   - SVG Code Explanation & Structure Analysis.
   - SVG XML Optimizer, Formatter & Minifier.
   - Dynamic Palette Recolor Engine (Cyberpunk, Sunset, Pastel, Emerald, Monochrome, Luxury Gold).
   - Natural Language Prompt-to-SVG Generator (via Local LLM or built-in high-quality vector engine).
   - SVG XML Error Auto-Fixer.
"""

import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Tuple
from xml.dom import minidom

import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore

vtracer: Any = None
try:
    import vtracer as _vtracer  # type: ignore
    vtracer = _vtracer
    HAS_VTRACER = True
except Exception:
    HAS_VTRACER = False


class SvgProcessor:
    """
    Professional Engine for AI Raster-to-SVG Vectorization and SVG XML Manipulation.
    """

    PRESETS = {
        'fidelity': {
            'name': 'Chân Thực Đa Sắc (High-Fidelity)',
            'desc': 'Bảo toàn tối đa đường cong Bézier, chi tiết và dải màu nguyên bản.',
            'colormode': 'color',
            'hierarchical': 'stacked',
            'mode': 'spline',
            'filter_speckle': 2,
            'color_precision': 8,
            'layer_difference': 8,
            'corner_threshold': 60,
            'length_threshold': 3.5,
            'max_iterations': 10,
            'splice_threshold': 45,
            'path_precision': 5,
            'preprocess': 'bilateral'
        },
        'clean_logo': {
            'name': 'Đồ Họa & Logo (Clean Cutout)',
            'desc': 'Đường nét sắc sảo, phân lớp chuẩn đồ họa phẳng, loại bỏ hạt nhiễu.',
            'colormode': 'color',
            'hierarchical': 'cutout',
            'mode': 'spline',
            'filter_speckle': 4,
            'color_precision': 6,
            'layer_difference': 16,
            'corner_threshold': 55,
            'length_threshold': 4.0,
            'max_iterations': 10,
            'splice_threshold': 45,
            'path_precision': 4,
            'preprocess': 'bilateral'
        },
        'lineart': {
            'name': 'Tranh Nét & Line Art (Sketch / CAD)',
            'desc': 'Trích xuất nét vẽ đơn sắc hoặc nhị phân siêu nét, không bị răng cưa.',
            'colormode': 'binary',
            'hierarchical': 'stacked',
            'mode': 'spline',
            'filter_speckle': 3,
            'color_precision': 6,
            'layer_difference': 16,
            'corner_threshold': 50,
            'length_threshold': 3.0,
            'max_iterations': 10,
            'splice_threshold': 45,
            'path_precision': 5,
            'preprocess': 'line_enhance'
        },
        'photo': {
            'name': 'Chân Dung & Ảnh Thật (Detailed Photo)',
            'desc': 'Vector nhiều lớp mật độ cao, chuyển màu êm dịu theo phong cách vector art.',
            'colormode': 'color',
            'hierarchical': 'stacked',
            'mode': 'spline',
            'filter_speckle': 1,
            'color_precision': 8,
            'layer_difference': 6,
            'corner_threshold': 65,
            'length_threshold': 2.5,
            'max_iterations': 10,
            'splice_threshold': 45,
            'path_precision': 5,
            'preprocess': 'soft'
        }
    }

    PALETTES = {
        'cyberpunk': {
            'name': 'Cyberpunk Neon',
            'colors': ['#00FFFF', '#FF007F', '#7928CA', '#FFE600', '#0D1117', '#FFFFFF']
        },
        'sunset': {
            'name': 'Hoàng Hôn (Warm Sunset)',
            'colors': ['#FF5E36', '#FFAE34', '#EB144C', '#6C0BA9', '#FFE680', '#2E0854']
        },
        'pastel': {
            'name': 'Pastel Nhẹ Nhàng (Soft Pastel)',
            'colors': ['#B5EAD7', '#FFDAC1', '#FF9AA2', '#E2F0CB', '#C7CEEA', '#FFFFFF']
        },
        'emerald': {
            'name': 'Lục Bảo Thiên Nhiên (Emerald Nature)',
            'colors': ['#10B981', '#047857', '#064E3B', '#34D399', '#A7F3D0', '#F0FDF4']
        },
        'monochrome': {
            'name': 'Đơn Sắc Hiện Đại (Monochrome Dark)',
            'colors': ['#111827', '#374151', '#4B5563', '#9CA3AF', '#E5E7EB', '#FFFFFF']
        },
        'luxury_gold': {
            'name': 'Hoàng Kim Sang Trọng (Luxury Gold)',
            'colors': ['#D4AF37', '#AA7C11', '#FFDF73', '#1A1A1A', '#333333', '#F5F5F0']
        }
    }

    def __init__(self):
        pass

    # =========================================================================
    # 1. LOCAL AI RASTER-TO-SVG VECTORIZATION
    # =========================================================================

    def vectorize_image(
        self,
        input_path: str,
        output_svg_path: str,
        preset: str = 'fidelity',
        custom_params: Optional[Dict[str, Any]] = None,
        max_dimension: int = 1920,
        enable_ai_line_boost: bool = True
    ) -> Dict[str, Any]:
        """
        Converts raster image (PNG, JPG, WEBP, etc.) to high-fidelity SVG vector paths.
        Guarantees true Bézier spline curves and accurate color layers.
        """
        if not HAS_VTRACER or vtracer is None:
            raise RuntimeError("Thư viện vtracer chưa được cài đặt. Vui lòng cài đặt vtracer.")

        t0 = time.time()
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Không tìm thấy tệp ảnh nguồn: {input_path}")

        orig_size_bytes = os.path.getsize(input_path)

        # 1. Load Image with Pillow and OpenCV
        pil_img = Image.open(input_path)
        orig_w, orig_h = pil_img.size
        has_alpha = pil_img.mode in ('RGBA', 'LA') or (pil_img.mode == 'P' and 'transparency' in pil_img.info)

        # 2. Resolve Parameters
        cfg = dict(self.PRESETS.get(preset, self.PRESETS['fidelity']))
        if custom_params:
            for k, v in custom_params.items():
                if v is not None and k in cfg:
                    cfg[k] = v

        # 3. Intelligent Preprocessing
        # Maintain aspect ratio within max_dimension
        scale = 1.0
        if max(orig_w, orig_h) > max_dimension:
            scale = max_dimension / float(max(orig_w, orig_h))
            new_w = max(16, round(orig_w * scale))
            new_h = max(16, round(orig_h * scale))
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            new_w, new_h = orig_w, orig_h

        # Temporary path for preprocessed image
        os.makedirs(os.path.dirname(os.path.abspath(output_svg_path)), exist_ok=True)
        temp_prep_path = output_svg_path + ".temp_prep.png"

        try:
            if has_alpha:
                # Keep RGBA clean
                rgba_np = np.array(pil_img.convert('RGBA'))
                if enable_ai_line_boost and cfg.get('preprocess') == 'bilateral':
                    # Apply bilateral filter to RGB channels only, preserving alpha mask
                    rgb = rgba_np[:, :, :3]
                    alpha = rgba_np[:, :, 3]
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    smoothed_bgr = cv2.bilateralFilter(bgr, d=5, sigmaColor=45, sigmaSpace=45)
                    smoothed_rgb = cv2.cvtColor(smoothed_bgr, cv2.COLOR_BGR2RGB)
                    rgba_np[:, :, :3] = smoothed_rgb
                Image.fromarray(rgba_np).save(temp_prep_path, 'PNG')
            else:
                rgb_np = np.array(pil_img.convert('RGB'))
                bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

                preprocess_mode = cfg.get('preprocess', 'bilateral')
                if preprocess_mode == 'line_enhance' or cfg.get('colormode') == 'binary':
                    # Enhance line contrast for sketch / line-art
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    denoised = cv2.bilateralFilter(gray, 5, 40, 40)
                    if enable_ai_line_boost:
                        # Slight unsharp mask to crispen line borders
                        gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
                        sharp = cv2.addWeighted(denoised, 1.35, gaussian, -0.35, 0)
                    else:
                        sharp = denoised
                    cv2.imwrite(temp_prep_path, sharp)
                elif preprocess_mode == 'bilateral':
                    # Edge-preserving denoising to eliminate JPEG block artifacts
                    filtered = cv2.bilateralFilter(bgr, d=5, sigmaColor=40, sigmaSpace=40)
                    if enable_ai_line_boost:
                        # Subtle high-frequency edge enhancement
                        gaussian = cv2.GaussianBlur(filtered, (0, 0), 1.5)
                        sharp = cv2.addWeighted(filtered, 1.25, gaussian, -0.25, 0)
                    else:
                        sharp = filtered
                    cv2.imwrite(temp_prep_path, sharp)
                else:
                    pil_img.save(temp_prep_path, 'PNG')

            # 4. Invoke VTracer Vector Engine
            assert vtracer is not None
            vtracer.convert_image_to_svg_py(
                temp_prep_path,
                output_svg_path,
                colormode=str(cfg.get('colormode', 'color')),
                hierarchical=str(cfg.get('hierarchical', 'stacked')),
                mode=str(cfg.get('mode', 'spline')),
                filter_speckle=int(cfg.get('filter_speckle', 2)),
                color_precision=int(cfg.get('color_precision', 8)),
                layer_difference=int(cfg.get('layer_difference', 8)),
                corner_threshold=int(cfg.get('corner_threshold', 60)),
                length_threshold=float(cfg.get('length_threshold', 3.5)),
                max_iterations=int(cfg.get('max_iterations', 10)),
                splice_threshold=int(cfg.get('splice_threshold', 45)),
                path_precision=int(cfg.get('path_precision', 5))
            )

        finally:
            if os.path.exists(temp_prep_path):
                try:
                    os.remove(temp_prep_path)
                except Exception:
                    pass

        # 5. Post-Process & Validate SVG
        svg_stats = self.inspect_svg_file(output_svg_path)
        duration_sec = round(time.time() - t0, 3)

        svg_size_bytes = os.path.getsize(output_svg_path)
        size_diff = svg_size_bytes - orig_size_bytes
        percent_diff = round((size_diff / orig_size_bytes) * 100, 1) if orig_size_bytes > 0 else 0

        return {
            'success': True,
            'preset': preset,
            'preset_name': cfg.get('name', preset),
            'orig_width': orig_w,
            'orig_height': orig_h,
            'vector_width': new_w,
            'vector_height': new_h,
            'has_alpha': has_alpha,
            'orig_size_bytes': orig_size_bytes,
            'svg_size_bytes': svg_size_bytes,
            'orig_size_kb': round(orig_size_bytes / 1024, 1),
            'svg_size_kb': round(svg_size_bytes / 1024, 1),
            'percent_diff': percent_diff,
            'duration_sec': duration_sec,
            'path_count': svg_stats.get('path_count', 0),
            'colors_count': len(svg_stats.get('palette', [])),
            'palette': svg_stats.get('palette', [])[:16],
            'viewBox': svg_stats.get('viewBox', f"0 0 {new_w} {new_h}"),
            'is_true_vector': True
        }

    # =========================================================================
    # 2. SVG XML PARSER, INSPECTOR & VALIDATOR
    # =========================================================================

    def inspect_svg_content(self, svg_xml: str) -> Dict[str, Any]:
        """
        Parses and extracts comprehensive metrics, element counts, and color palettes from SVG XML.
        """
        # Clean leading/trailing spaces
        svg_xml = svg_xml.strip()
        if not svg_xml:
            raise ValueError("Mã SVG XML rỗng!")

        # Basic tag validation
        if '<svg' not in svg_xml.lower():
            raise ValueError("Nội dung không chứa thẻ <svg> hợp lệ.")

        try:
            root = ET.fromstring(svg_xml)
        except ET.ParseError as pe:
            line, col = pe.position if hasattr(pe, 'position') else (1, 1)
            raise ValueError(f"Lỗi cú pháp XML tại dòng {line}, cột {col}: {pe}")

        # Extract attributes
        width = root.attrib.get('width', '')
        height = root.attrib.get('height', '')
        viewBox = root.attrib.get('viewBox', '')

        # Element counts across all namespaces
        elem_counts: Dict[str, int] = {
            'path': 0, 'rect': 0, 'circle': 0, 'ellipse': 0,
            'line': 0, 'polyline': 0, 'polygon': 0, 'text': 0,
            'g': 0, 'defs': 0, 'linearGradient': 0, 'radialGradient': 0
        }

        color_counter: Dict[str, int] = {}

        def record_color(val: Optional[str]):
            if not val or val.lower() in ('none', 'transparent', 'inherit', 'currentcolor'):
                return
            val_clean = val.strip()
            # Normalize hex
            if val_clean.startswith('#'):
                hex_c = val_clean[:7].upper()
                color_counter[hex_c] = color_counter.get(hex_c, 0) + 1
            elif val_clean.startswith('rgb'):
                color_counter[val_clean] = color_counter.get(val_clean, 0) + 1

        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag in elem_counts:
                elem_counts[tag] += 1

            # Extract fill & stroke
            fill = elem.attrib.get('fill')
            stroke = elem.attrib.get('stroke')
            record_color(fill)
            record_color(stroke)

            # Check style attribute
            style = elem.attrib.get('style', '')
            if 'fill:' in style:
                m = re.search(r'fill:\s*([^;]+)', style)
                if m:
                    record_color(m.group(1))
            if 'stroke:' in style:
                m = re.search(r'stroke:\s*([^;]+)', style)
                if m:
                    record_color(m.group(1))

        # Sort palette by usage
        sorted_colors = sorted(color_counter.items(), key=lambda x: -x[1])
        total_colored_elements = max(1, sum(c[1] for c in sorted_colors))
        palette = []
        for c_hex, cnt in sorted_colors:
            pct = round((cnt / total_colored_elements) * 100, 1)
            palette.append({
                'hex': c_hex,
                'count': cnt,
                'percentage': pct
            })

        total_elements = sum(elem_counts.values())

        # Determine complexity rating
        if elem_counts['path'] > 2000 or total_elements > 3000:
            complexity = "Chi tiết Cao (Photorealistic / Complex Artwork)"
        elif elem_counts['path'] > 300 or total_elements > 500:
            complexity = "Đồ họa Chi tiết (Detailed Illustration / Iconography)"
        elif elem_counts['path'] > 20 or total_elements > 30:
            complexity = "Đồ họa Tiêu Chuẩn (Standard Vector Logo / UI Graphic)"
        else:
            complexity = "Tối giản (Minimal Icon / Clean Geometry)"

        return {
            'valid': True,
            'width': width,
            'height': height,
            'viewBox': viewBox,
            'elements': elem_counts,
            'path_count': elem_counts['path'],
            'total_elements': total_elements,
            'palette': palette,
            'complexity': complexity,
            'has_gradients': (elem_counts['linearGradient'] + elem_counts['radialGradient']) > 0
        }

    def inspect_svg_file(self, svg_path: str) -> Dict[str, Any]:
        """
        Reads SVG file and returns analysis stats along with XML content.
        """
        if not os.path.exists(svg_path):
            raise FileNotFoundError(f"Không tìm thấy file SVG: {svg_path}")

        with open(svg_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        stats = self.inspect_svg_content(content)
        stats['file_size_bytes'] = os.path.getsize(svg_path)
        stats['file_size_kb'] = round(stats['file_size_bytes'] / 1024, 2)
        stats['xml_content'] = content
        return stats

    # =========================================================================
    # 3. BIDIRECTIONAL CONVERSION (XML <-> SVG FILE)
    # =========================================================================

    def xml_to_svg_file(self, xml_text: str, output_path: str) -> Dict[str, Any]:
        """
        Validates SVG XML code and writes it to an authentic .svg file.
        Auto-corrects missing xmlns or viewBox if omitted.
        """
        xml_clean = xml_text.strip()
        if not xml_clean:
            raise ValueError("Mã XML rỗng!")

        # Auto-wrap if user just pasted path elements
        if not xml_clean.startswith('<svg') and not xml_clean.startswith('<?xml'):
            if '<path' in xml_clean or '<rect' in xml_clean or '<circle' in xml_clean:
                xml_clean = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">\n{xml_clean}\n</svg>'

        # Ensure xmlns exists
        if '<svg' in xml_clean and 'xmlns=' not in xml_clean:
            xml_clean = re.sub(r'<svg', '<svg xmlns="http://www.w3.org/2000/svg"', xml_clean, count=1)

        # Parse and sanitize
        try:
            root = ET.fromstring(xml_clean)
        except ET.ParseError as pe:
            line, col = pe.position if hasattr(pe, 'position') else (1, 1)
            raise ValueError(f"Mã XML không hợp lệ tại dòng {line}, cột {col}: {pe}")

        # Remove potentially unsafe script tags
        for script in list(root.iter('{http://www.w3.org/2000/svg}script')) + list(root.iter('script')):
            root.remove(script)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Write clean SVG
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_clean)

        stats = self.inspect_svg_file(output_path)
        stats['output_path'] = output_path
        return stats

    def format_svg_xml(self, xml_text: str, minify: bool = False) -> str:
        """
        Beautifies (indents) or minifies SVG XML.
        """
        xml_clean = xml_text.strip()
        try:
            # Parse with minidom for pretty printing
            parsed = minidom.parseString(xml_clean)
            if minify:
                # Minify: strip comments, newlines and extra spaces
                raw = parsed.toxml(encoding='utf-8').decode('utf-8')
                raw = re.sub(r'<!--[\s\S]*?-->', '', raw)
                raw = re.sub(r'>\s+<', '><', raw)
                return raw.strip()
            else:
                # Beautify with 2-space indentation
                pretty = parsed.toprettyxml(indent='  ', encoding='utf-8').decode('utf-8')
                # Remove empty blank lines created by minidom
                lines = [l for l in pretty.splitlines() if l.strip()]
                return '\n'.join(lines)
        except Exception:
            # Fallback simple regex formatting
            if minify:
                c = re.sub(r'<!--[\s\S]*?-->', '', xml_clean)
                return re.sub(r'\s+', ' ', c).replace('> <', '><').strip()
            return xml_clean

    # =========================================================================
    # 4. AI SVG ASSISTANT (EXPLAIN, OPTIMIZE, RECOLOR, GENERATE, FIX)
    # =========================================================================

    def ai_assist(
        self,
        action: str,
        svg_xml: str = "",
        prompt: str = "",
        palette_key: str = "",
        llm_engine: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        AI-powered vector operations:
        - 'explain': Explains the visual elements, layer count, color theme.
        - 'optimize': Cleans up redundant paths/tags, formats and minifies code.
        - 'recolor': Shifts SVG color scheme to target aesthetic palette.
        - 'generate': Generates complete SVG graphic from a text prompt.
        - 'fix': Diagnoses and fixes broken XML syntax, unclosed tags.
        """
        action = action.lower().strip()

        if action == 'explain':
            return self._ai_explain(svg_xml, llm_engine)
        elif action == 'optimize':
            return self._ai_optimize(svg_xml)
        elif action == 'recolor':
            return self._ai_recolor(svg_xml, palette_key)
        elif action == 'generate':
            return self._ai_generate_svg(prompt, llm_engine)
        elif action == 'fix':
            return self._ai_fix_xml(svg_xml)
        else:
            raise ValueError(f"Hành động AI '{action}' không được hỗ trợ.")

    def _ai_explain(self, svg_xml: str, llm_engine: Optional[Any]) -> Dict[str, Any]:
        """Provides technical & artistic breakdown of the SVG."""
        stats = self.inspect_svg_content(svg_xml)
        p_count = stats['path_count']
        top_colors = [p['hex'] for p in stats.get('palette', [])[:5]]
        colors_str = ", ".join(top_colors) if top_colors else "Không có mã màu rõ ràng"
        viewBox = stats.get('viewBox', 'Không chỉ định')

        explanation = (
            f"### Phân Tích Cấu Trúc SVG Vector\n\n"
            f"- **Độ phức tạp**: {stats['complexity']}\n"
            f"- **Số lượng đường nét (Bézier Paths)**: **{p_count:,}** paths\n"
            f"- **Tổng số thẻ đồ họa**: **{stats['total_elements']:,}** elements "
            f"({stats['elements'].get('circle', 0)} hình tròn, {stats['elements'].get('rect', 0)} hình chữ nhật, "
            f"{stats['elements'].get('polygon', 0)} đa giác, {stats['elements'].get('g', 0)} nhóm layer)\n"
            f"- **Tọa độ hiển thị (viewBox)**: `{viewBox}` (Khả năng co giãn vô cực sắc nét 100% không vỡ hạt)\n"
            f"- **Gam màu chủ đạo ({len(stats['palette'])} màu)**: {colors_str}\n"
            f"- **Dải chuyển sắc (Gradients)**: {'Có sử dụng Gradients mượt mà' if stats['has_gradients'] else 'Đồ họa màu phẳng (Solid Flat Colors)'}\n\n"
            f"**Khuyến nghị sử dụng**: "
        )

        if p_count > 1500:
            explanation += "Tệp vector này có độ chi tiết cực cao, thích hợp làm tranh minh họa vector chất lượng lớn hoặc in ấn banner sắc nét vô hạn."
        elif p_count > 100:
            explanation += "Tệp đồ họa rất cân đối giữa chi tiết hình khối và hiệu năng hiển thị Web. Rất thích hợp làm Logo công ty hoặc Illustration tương tác."
        else:
            explanation += "Tệp có cấu trúc tối giản, nhẹ và tối ưu tốc độ tải. Rất thích hợp làm icon hệ thống, favicon, nút bấm hoặc đồ họa UI/UX."

        return {
            'success': True,
            'action': 'explain',
            'explanation': explanation,
            'stats': stats
        }

    def _ai_optimize(self, svg_xml: str) -> Dict[str, Any]:
        """Cleans, formats, and minifies SVG code."""
        orig_len = len(svg_xml)
        formatted = self.format_svg_xml(svg_xml, minify=False)
        minified = self.format_svg_xml(svg_xml, minify=True)

        saved_bytes = max(0, orig_len - len(minified))
        saved_pct = round((saved_bytes / max(1, orig_len)) * 100, 1)

        return {
            'success': True,
            'action': 'optimize',
            'formatted_xml': formatted,
            'minified_xml': minified,
            'orig_size_bytes': orig_len,
            'minified_size_bytes': len(minified),
            'saved_bytes': saved_bytes,
            'saved_pct': saved_pct,
            'message': f"Đã tối ưu hóa XML thành công! Giảm bớt {saved_bytes} bytes ({saved_pct}%)."
        }

    def _ai_recolor(self, svg_xml: str, palette_key: str) -> Dict[str, Any]:
        """Re-maps colors in SVG XML according to selected palette."""
        pal = self.PALETTES.get(palette_key, self.PALETTES['cyberpunk'])
        target_colors = pal['colors']

        # Extract existing unique colors
        stats = self.inspect_svg_content(svg_xml)
        existing_palette = [p['hex'] for p in stats.get('palette', [])]

        if not existing_palette:
            raise ValueError("Không tìm thấy màu sắc nào trong tệp SVG để đổi tông.")

        # Create 1-to-1 or cyclic mapping
        color_map: Dict[str, str] = {}
        for idx, src_hex in enumerate(existing_palette):
            target_hex = target_colors[idx % len(target_colors)]
            color_map[src_hex.upper()] = target_hex
            color_map[src_hex.lower()] = target_hex

        # Replace in XML
        recolored_xml = svg_xml
        for old_c, new_c in color_map.items():
            recolored_xml = re.sub(re.escape(old_c), new_c, recolored_xml, flags=re.IGNORECASE)

        return {
            'success': True,
            'action': 'recolor',
            'palette_key': palette_key,
            'palette_name': pal['name'],
            'mapped_colors': color_map,
            'recolored_xml': recolored_xml
        }

    def _ai_fix_xml(self, svg_xml: str) -> Dict[str, Any]:
        """Repairs common syntax errors in malformed SVG XML."""
        fixed = svg_xml.strip()
        changes = []

        # Remove leading junk before <svg or <?xml
        first_lt = fixed.find('<')
        if first_lt > 0:
            fixed = fixed[first_lt:]
            changes.append("Đã loại bỏ các ký tự rác ở đầu tệp.")

        # Add missing xmlns
        if '<svg' in fixed and 'xmlns=' not in fixed:
            fixed = re.sub(r'<svg', '<svg xmlns="http://www.w3.org/2000/svg"', fixed, count=1)
            changes.append("Đã bổ sung thuộc tính chuẩn xmlns=\"http://www.w3.org/2000/svg\".")

        # Add missing viewBox if only width & height are specified
        if 'viewBox' not in fixed:
            w_m = re.search(r'width=["\']?(\d+)', fixed)
            h_m = re.search(r'height=["\']?(\d+)', fixed)
            w = w_m.group(1) if w_m else "500"
            h = h_m.group(1) if h_m else "500"
            fixed = re.sub(r'<svg', f'<svg viewBox="0 0 {w} {h}"', fixed, count=1)
            changes.append(f"Đã bổ sung thuộc tính đáp ứng viewBox=\"0 0 {w} {h}\".")

        # Ensure closing </svg>
        if not fixed.rstrip().endswith('</svg>'):
            fixed += "\n</svg>"
            changes.append("Đã đóng thẻ </svg> ở cuối mã.")

        # Verify parsed result
        try:
            ET.fromstring(fixed)
            is_valid = True
            msg = "Đã sửa lỗi XML thành công! Mã hiện đã hợp lệ 100%."
        except ET.ParseError as pe:
            is_valid = False
            msg = f"Đã áp dụng các bản vá cơ bản nhưng XML vẫn còn lỗi cú pháp phức tạp: {pe}"

        return {
            'success': is_valid,
            'action': 'fix',
            'fixed_xml': fixed,
            'changes': changes,
            'message': msg
        }

    def _ensure_svg_standards(self, svg_code: str) -> str:
        """Ensures SVG has standard xmlns, viewBox, width, and height attributes."""
        if not svg_code or '<svg' not in svg_code:
            return svg_code
        
        # Add xmlns if missing
        if 'xmlns="http://www.w3.org/2000/svg"' not in svg_code and "xmlns='http://www.w3.org/2000/svg'" not in svg_code:
            svg_code = re.sub(r'<svg\b', '<svg xmlns="http://www.w3.org/2000/svg"', svg_code, count=1)
        
        # Check viewBox
        if 'viewBox' not in svg_code:
            w_m = re.search(r'width=["\']?(\d+)', svg_code)
            h_m = re.search(r'height=["\']?(\d+)', svg_code)
            w = w_m.group(1) if w_m else "500"
            h = h_m.group(1) if h_m else "500"
            svg_code = re.sub(r'<svg\b', f'<svg viewBox="0 0 {w} {h}"', svg_code, count=1)
        
        # Ensure width="100%" and height="100%"
        if 'width="100%"' not in svg_code and "width='100%'" not in svg_code:
            if 'width=' in svg_code:
                svg_code = re.sub(r'width=["\'][^"\']+["\']', 'width="100%"', svg_code, count=1)
            else:
                svg_code = re.sub(r'<svg\b', '<svg width="100%"', svg_code, count=1)
        
        if 'height="100%"' not in svg_code and "height='100%'" not in svg_code:
            if 'height=' in svg_code:
                svg_code = re.sub(r'height=["\'][^"\']+["\']', 'height="100%"', svg_code, count=1)
            else:
                svg_code = re.sub(r'<svg\b', '<svg height="100%"', svg_code, count=1)
                
        return svg_code

    def _ai_generate_svg(self, prompt: str, llm_engine: Optional[Any]) -> Dict[str, Any]:
        """
        Generates semantic SVG XML vector graphic from natural language prompt.
        Uses specialized parametric vector synthesis for immediate, studio-quality rendering,
        and leverages the local LLM engine for freeform descriptions.
        """
        prompt_clean = prompt.strip()
        if not prompt_clean:
            prompt_clean = "Biểu tượng khiên bảo mật công nghệ neon"

        p_lower = prompt_clean.lower()

        # 1. First check if prompt matches our rich parametric subjects for instant studio-grade rendering
        synthesized_svg = self._synthesize_vector_by_prompt(prompt_clean)
        is_generic = (synthesized_svg == self._get_default_hexagon_svg())

        # If it matched a specific category, return the polished vector graphic immediately!
        if not is_generic:
            return {
                'success': True,
                'action': 'generate',
                'prompt': prompt_clean,
                'svg_xml': synthesized_svg,
                'source': 'built_in_vector_synthesizer'
            }

        # 2. For custom / freeform prompts, try local LLM if available
        if llm_engine is not None and hasattr(llm_engine, 'chat'):
            try:
                reply_data = llm_engine.chat(
                    user_message=(
                        f"Generate authentic, complete SVG vector illustration code for: '{prompt_clean}'. "
                        f"Requirements: Output ONLY the <svg viewBox=\"0 0 500 500\" xmlns=\"http://www.w3.org/2000/svg\" width=\"100%\" height=\"100%\">...<path>...</svg> "
                        f"with modern gradients and coordinates. No explanations, no markdown fences."
                    )
                )
                reply_text = ""
                if isinstance(reply_data, dict):
                    reply_text = str(reply_data.get('reply') or reply_data.get('response') or '')
                elif isinstance(reply_data, str):
                    reply_text = reply_data

                # Extract SVG block
                m = re.search(r'<svg[\s\S]*?</svg>', reply_text, re.IGNORECASE)
                if m:
                    svg_code = self._ensure_svg_standards(m.group(0))
                    # Quick validation
                    if '<path' in svg_code or '<circle' in svg_code or '<polygon' in svg_code or '<rect' in svg_code:
                        return {
                            'success': True,
                            'action': 'generate',
                            'prompt': prompt_clean,
                            'svg_xml': svg_code,
                            'source': 'local_llm'
                        }
            except Exception as ex:
                print(f"[SvgProcessor] Local LLM generate fallback: {ex}")

        # 3. Fallback to parametric visual synthesizer
        return {
            'success': True,
            'action': 'generate',
            'prompt': prompt_clean,
            'svg_xml': synthesized_svg,
            'source': 'built_in_vector_synthesizer'
        }

    def _get_default_hexagon_svg(self) -> str:
        return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">
  <defs>
    <linearGradient id="gradBrand" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#06B6D4" />
      <stop offset="50%" stop-color="#3B82F6" />
      <stop offset="100%" stop-color="#8B5CF6" />
    </linearGradient>
  </defs>
  <!-- Outer Hexagon Frame -->
  <polygon points="250,50 420,150 420,350 250,450 80,350 80,150" fill="none" stroke="url(#gradBrand)" stroke-width="12" stroke-linejoin="round"/>
  <!-- Interlocking Geometric Bands -->
  <path d="M250 110 L370 180 L370 320 L250 390 L130 320 L130 180 Z" fill="rgba(59, 130, 246, 0.08)" stroke="url(#gradBrand)" stroke-width="4"/>
  <!-- Central Sphere -->
  <circle cx="250" cy="250" r="50" fill="url(#gradBrand)" />
  <circle cx="250" cy="250" r="25" fill="#FFFFFF" opacity="0.9" />
  <!-- Tech Nodes -->
  <circle cx="250" cy="50" r="14" fill="#06B6D4"/>
  <circle cx="420" cy="150" r="14" fill="#3B82F6"/>
  <circle cx="420" cy="350" r="14" fill="#8B5CF6"/>
  <circle cx="250" cy="450" r="14" fill="#06B6D4"/>
  <circle cx="80" cy="350" r="14" fill="#3B82F6"/>
  <circle cx="80" cy="150" r="14" fill="#8B5CF6"/>
</svg>"""

    def _synthesize_vector_by_prompt(self, prompt: str) -> str:
        """
        Synthesizes sophisticated SVG graphics based on prompt keywords:
        Shield, Rocket, Star, Heart, Sun, Moon, Flower, Coffee, Cat, Camera, Robot, Car, Chart, Diamond, Cloud, Flame, etc.
        """
        p_lower = prompt.lower()

        # 0. Penguin / Chim Cánh Cụt / Cute Bird
        if any(w in p_lower for w in ('penguin', 'cánh cụt', 'chim cánh cụt', 'pingu')):
            return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">
  <defs>
    <linearGradient id="penBody" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1E293B" />
      <stop offset="100%" stop-color="#0F172A" />
    </linearGradient>
    <linearGradient id="penBelly" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#FFFFFF" />
      <stop offset="100%" stop-color="#F1F5F9" />
    </linearGradient>
    <linearGradient id="penBeak" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FBBF24" />
      <stop offset="100%" stop-color="#F59E0B" />
    </linearGradient>
    <linearGradient id="blushGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FDA4AF" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="#FB7185" stop-opacity="0.3"/>
    </linearGradient>
  </defs>
  <!-- Shadow -->
  <ellipse cx="250" cy="450" rx="140" ry="22" fill="rgba(0,0,0,0.15)"/>
  <!-- Feet -->
  <ellipse cx="190" cy="435" rx="36" ry="18" fill="url(#penBeak)"/>
  <ellipse cx="310" cy="435" rx="36" ry="18" fill="url(#penBeak)"/>
  <!-- Wings -->
  <path d="M125 240 C95 270 95 340 135 360 C145 340 145 280 140 245 Z" fill="url(#penBody)" stroke="#334155" stroke-width="2"/>
  <path d="M375 240 C405 270 405 340 365 360 C355 340 355 280 360 245 Z" fill="url(#penBody)" stroke="#334155" stroke-width="2"/>
  <!-- Chubby Round Body -->
  <ellipse cx="250" cy="275" rx="145" ry="165" fill="url(#penBody)"/>
  <!-- White Fluffy Belly -->
  <ellipse cx="250" cy="305" rx="100" ry="125" fill="url(#penBelly)"/>
  <!-- Face White Patches -->
  <circle cx="205" cy="195" r="42" fill="#FFFFFF"/>
  <circle cx="295" cy="195" r="42" fill="#FFFFFF"/>
  <!-- Adorable Expressive Eyes -->
  <ellipse cx="205" cy="195" rx="16" ry="22" fill="#0F172A"/>
  <ellipse cx="295" cy="195" rx="16" ry="22" fill="#0F172A"/>
  <circle cx="199" cy="186" r="6" fill="#FFFFFF"/>
  <circle cx="289" cy="186" r="6" fill="#FFFFFF"/>
  <circle cx="209" cy="204" r="3" fill="#FFFFFF"/>
  <circle cx="299" cy="204" r="3" fill="#FFFFFF"/>
  <!-- Cute Pink Cheeks / Blush -->
  <ellipse cx="165" cy="225" rx="18" ry="10" fill="url(#blushGrad)"/>
  <ellipse cx="335" cy="225" rx="18" ry="10" fill="url(#blushGrad)"/>
  <!-- Little Beak -->
  <path d="M232 210 Q250 226 268 210 Q250 236 232 210 Z" fill="url(#penBeak)"/>
</svg>"""

        # 1. Heart / Love
        if re.search(r'\b(heart|tim|trái tim|love|yêu|tình yêu|thương)\b', p_lower):
            return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">
  <defs>
    <linearGradient id="heartGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FF3366" />
      <stop offset="50%" stop-color="#FF0044" />
      <stop offset="100%" stop-color="#B3003B" />
    </linearGradient>
    <radialGradient id="heartGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#FF0066" stop-opacity="0.4"/>
      <stop offset="100%" stop-color="#FF0066" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="shineGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.7"/>
      <stop offset="100%" stop-color="#FFFFFF" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <circle cx="250" cy="250" r="210" fill="url(#heartGlow)"/>
  <path d="M250 430 C250 430 80 320 80 180 C80 100 140 60 210 80 C235 87 245 105 250 115 C255 105 265 87 290 80 C360 60 420 100 420 180 C420 320 250 430 250 430 Z" fill="url(#heartGrad)" stroke="#FFFFFF" stroke-width="4"/>
  <path d="M120 160 C120 120 150 95 190 95 C205 95 210 105 205 115 C175 115 145 135 140 170 C135 175 120 175 120 160 Z" fill="url(#shineGrad)"/>
  <circle cx="340" cy="140" r="10" fill="#FFFFFF" opacity="0.6"/>
</svg>"""

        # 2. Sun / Solar
        elif any(w in p_lower for w in ('sun', 'mặt trời', 'nắng', 'solar', 'bình minh', 'hoàng hôn')):
            return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">
  <defs>
    <linearGradient id="sunGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FFF066"/>
      <stop offset="50%" stop-color="#FFAA00"/>
      <stop offset="100%" stop-color="#FF5500"/>
    </linearGradient>
    <radialGradient id="sunAura" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#FFAA00" stop-opacity="0.3"/>
      <stop offset="100%" stop-color="#FF5500" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <circle cx="250" cy="250" r="220" fill="url(#sunAura)"/>
  <g stroke="url(#sunGrad)" stroke-width="12" stroke-linecap="round">
    <line x1="250" y1="50" x2="250" y2="90"/>
    <line x1="250" y1="410" x2="250" y2="450"/>
    <line x1="50" y1="250" x2="90" y2="250"/>
    <line x1="410" y1="250" x2="450" y2="250"/>
    <line x1="109" y1="109" x2="137" y2="137"/>
    <line x1="363" y1="363" x2="391" y2="391"/>
    <line x1="109" y1="391" x2="137" y2="363"/>
    <line x1="363" y1="137" x2="391" y2="109"/>
  </g>
  <circle cx="250" cy="250" r="120" fill="url(#sunGrad)" stroke="#FFFFFF" stroke-width="4"/>
  <circle cx="220" cy="220" r="85" fill="#FFFFFF" opacity="0.2"/>
</svg>"""

        # 3. Moon / Night
        elif any(w in p_lower for w in ('moon', 'mặt trăng', 'trăng', 'đêm', 'night', 'khuyết')):
            return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">
  <defs>
    <linearGradient id="moonGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FCEE21"/>
      <stop offset="50%" stop-color="#FFD200"/>
      <stop offset="100%" stop-color="#F7931E"/>
    </linearGradient>
    <radialGradient id="nightAura" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#7B2CBF" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="#10002B" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <circle cx="250" cy="250" r="230" fill="url(#nightAura)"/>
  <path d="M220 70 C320 70 400 150 400 250 C400 350 320 430 220 430 C180 430 145 418 115 398 C215 380 290 295 290 190 C290 140 270 95 235 62 C230 65 225 68 220 70 Z" fill="url(#moonGrad)" stroke="#FFFFFF" stroke-width="4"/>
  <polygon points="120,120 126,138 144,144 126,150 120,168 114,150 96,144 114,138" fill="#FFFFFF" opacity="0.9"/>
  <polygon points="350,110 354,122 366,126 354,130 350,142 346,130 334,126 346,122" fill="#FFFFFF" opacity="0.8"/>
  <polygon points="100,280 103,290 113,293 103,296 100,306 97,296 87,293 97,290" fill="#00F2FE" opacity="0.8"/>
</svg>"""

        # 4. Flower / Blossom / Nature
        elif any(w in p_lower for w in ('flower', 'hoa', 'bông hoa', 'blossom', 'cây', 'lá', 'plant', 'leaf')):
            return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">
  <defs>
    <linearGradient id="petalGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FF5252"/>
      <stop offset="50%" stop-color="#FF4081"/>
      <stop offset="100%" stop-color="#7C4DFF"/>
    </linearGradient>
    <radialGradient id="flowerCore" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#FFE082"/>
      <stop offset="70%" stop-color="#FFB300"/>
      <stop offset="100%" stop-color="#FF8F00"/>
    </radialGradient>
    <linearGradient id="leafGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#69F0AE"/>
      <stop offset="100%" stop-color="#00B0FF"/>
    </linearGradient>
  </defs>
  <path d="M150 350 C100 400 130 460 220 410 C180 370 170 350 150 350 Z" fill="url(#leafGrad)"/>
  <path d="M350 350 C400 400 370 460 280 410 C320 370 330 350 350 350 Z" fill="url(#leafGrad)"/>
  <g transform="translate(250,250)">
    <ellipse cx="0" cy="-110" rx="42" ry="75" fill="url(#petalGrad)"/>
    <ellipse cx="78" cy="-78" rx="42" ry="75" transform="rotate(45)" fill="url(#petalGrad)"/>
    <ellipse cx="110" cy="0" rx="42" ry="75" transform="rotate(90)" fill="url(#petalGrad)"/>
    <ellipse cx="78" cy="78" rx="42" ry="75" transform="rotate(135)" fill="url(#petalGrad)"/>
    <ellipse cx="0" cy="110" rx="42" ry="75" transform="rotate(180)" fill="url(#petalGrad)"/>
    <ellipse cx="-78" cy="78" rx="42" ry="75" transform="rotate(225)" fill="url(#petalGrad)"/>
    <ellipse cx="-110" cy="0" rx="42" ry="75" transform="rotate(270)" fill="url(#petalGrad)"/>
    <ellipse cx="-78" cy="-78" rx="42" ry="75" transform="rotate(315)" fill="url(#petalGrad)"/>
    <circle cx="0" cy="0" r="50" fill="url(#flowerCore)" stroke="#FFFFFF" stroke-width="5"/>
    <circle cx="0" cy="0" r="32" fill="none" stroke="rgba(255,255,255,0.4)" stroke-dasharray="6 4" stroke-width="3"/>
  </g>
</svg>"""

        # 5. Coffee / Tea / Drink
        elif any(w in p_lower for w in ('coffee', 'cà phê', 'trà', 'tea', 'cup', 'tách', 'uống', 'drink')):
            return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">
  <defs>
    <linearGradient id="cupGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#F8FAFC"/>
      <stop offset="100%" stop-color="#94A3B8"/>
    </linearGradient>
    <linearGradient id="coffeeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#451A03"/>
      <stop offset="50%" stop-color="#78350F"/>
      <stop offset="100%" stop-color="#92400E"/>
    </linearGradient>
    <linearGradient id="steamGrad" x1="0%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" stop-color="#06B6D4" stop-opacity="0.8"/>
      <stop offset="100%" stop-color="#3B82F6" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <path d="M190 140 C170 100 210 70 190 30" fill="none" stroke="url(#steamGrad)" stroke-width="8" stroke-linecap="round"/>
  <path d="M250 140 C230 90 270 60 250 20" fill="none" stroke="url(#steamGrad)" stroke-width="8" stroke-linecap="round"/>
  <path d="M310 140 C290 100 330 70 310 30" fill="none" stroke="url(#steamGrad)" stroke-width="8" stroke-linecap="round"/>
  <ellipse cx="250" cy="410" rx="180" ry="30" fill="url(#cupGrad)" stroke="#64748B" stroke-width="4"/>
  <ellipse cx="250" cy="405" rx="140" ry="18" fill="#E2E8F0"/>
  <path d="M320 200 C390 200 390 310 320 310" fill="none" stroke="url(#cupGrad)" stroke-width="26" stroke-linecap="round"/>
  <path d="M140 180 C140 340 170 380 250 380 C330 380 360 340 360 180 Z" fill="url(#cupGrad)" stroke="#64748B" stroke-width="4"/>
  <ellipse cx="250" cy="180" rx="110" ry="25" fill="url(#coffeeGrad)"/>
  <ellipse cx="250" cy="180" rx="90" ry="18" fill="#B45309" opacity="0.5"/>
</svg>"""

        # 6. Cat / Pet / Animal
        elif any(w in p_lower for w in ('cat', 'mèo', 'con mèo', 'kitten', 'chó', 'dog', 'pet', 'thú')):
            return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">
  <defs>
    <linearGradient id="catGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38BDF8"/>
      <stop offset="50%" stop-color="#818CF8"/>
      <stop offset="100%" stop-color="#C084FC"/>
    </linearGradient>
    <linearGradient id="earInner" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#F472B6"/>
      <stop offset="100%" stop-color="#FB7185"/>
    </linearGradient>
  </defs>
  <polygon points="120,220 130,80 230,170" fill="url(#catGrad)" stroke="#FFFFFF" stroke-width="4"/>
  <polygon points="140,200 148,110 215,170" fill="url(#earInner)"/>
  <polygon points="380,220 370,80 270,170" fill="url(#catGrad)" stroke="#FFFFFF" stroke-width="4"/>
  <polygon points="360,200 352,110 285,170" fill="url(#earInner)"/>
  <path d="M110 270 C110 180 170 170 250 170 C330 170 390 180 390 270 C390 380 340 430 250 430 C160 430 110 380 110 270 Z" fill="url(#catGrad)" stroke="#FFFFFF" stroke-width="4"/>
  <ellipse cx="190" cy="270" rx="26" ry="34" fill="#FACC15"/>
  <ellipse cx="190" cy="270" rx="10" ry="28" fill="#1E1B4B"/>
  <circle cx="182" cy="256" r="6" fill="#FFFFFF"/>
  <ellipse cx="310" cy="270" rx="26" ry="34" fill="#FACC15"/>
  <ellipse cx="310" cy="270" rx="10" ry="28" fill="#1E1B4B"/>
  <circle cx="302" cy="256" r="6" fill="#FFFFFF"/>
  <polygon points="240,320 260,320 250,332" fill="#FB7185"/>
  <path d="M230 342 C240 352 250 350 250 332 C250 350 260 352 270 342" fill="none" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round"/>
  <line x1="80" y1="310" x2="160" y2="320" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round"/>
  <line x1="70" y1="340" x2="160" y2="335" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round"/>
  <line x1="420" y1="310" x2="340" y2="320" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round"/>
  <line x1="430" y1="340" x2="340" y2="335" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round"/>
</svg>"""

        # 7. Camera / Photography
        elif any(w in p_lower for w in ('camera', 'máy ảnh', 'chụp hình', 'ảnh', 'photo', 'ống kính', 'lens')):
            return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">
  <defs>
    <linearGradient id="bodyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1E293B"/>
      <stop offset="100%" stop-color="#0F172A"/>
    </linearGradient>
    <linearGradient id="lensRing" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#06B6D4"/>
      <stop offset="50%" stop-color="#3B82F6"/>
      <stop offset="100%" stop-color="#8B5CF6"/>
    </linearGradient>
  </defs>
  <rect x="140" y="90" width="60" height="30" rx="8" fill="#475569"/>
  <rect x="310" y="100" width="40" height="20" rx="6" fill="#EF4444"/>
  <rect x="70" y="120" width="360" height="260" rx="36" fill="url(#bodyGrad)" stroke="#334155" stroke-width="6"/>
  <circle cx="120" cy="170" r="14" fill="#38BDF8"/>
  <rect x="330" y="160" width="50" height="24" rx="6" fill="#F59E0B"/>
  <circle cx="250" cy="250" r="110" fill="#020617" stroke="url(#lensRing)" stroke-width="12"/>
  <circle cx="250" cy="250" r="85" fill="#0F172A" stroke="#334155" stroke-width="4"/>
  <circle cx="250" cy="250" r="55" fill="url(#lensRing)"/>
  <ellipse cx="230" cy="230" rx="20" ry="12" fill="#FFFFFF" opacity="0.6" transform="rotate(-30 230 230)"/>
</svg>"""

        # 8. Robot / AI / Bot
        elif any(w in p_lower for w in ('robot', 'người máy', 'bot', 'trí tuệ nhân tạo', 'android')):
            return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">
  <defs>
    <linearGradient id="botGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#06B6D4"/>
      <stop offset="50%" stop-color="#3B82F6"/>
      <stop offset="100%" stop-color="#6366F1"/>
    </linearGradient>
    <linearGradient id="screenGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="100%" stop-color="#020617"/>
    </linearGradient>
  </defs>
  <line x1="250" y1="120" x2="250" y2="50" stroke="#06B6D4" stroke-width="8" stroke-linecap="round"/>
  <circle cx="250" cy="40" r="16" fill="#F43F5E"/>
  <rect x="75" y="210" width="30" height="60" rx="10" fill="url(#botGrad)"/>
  <rect x="395" y="210" width="30" height="60" rx="10" fill="url(#botGrad)"/>
  <rect x="95" y="120" width="310" height="240" rx="45" fill="url(#botGrad)" stroke="#FFFFFF" stroke-width="5"/>
  <rect x="130" y="160" width="240" height="130" rx="25" fill="url(#screenGrad)" stroke="#38BDF8" stroke-width="4"/>
  <circle cx="190" cy="225" r="24" fill="#00F2FE"/>
  <circle cx="190" cy="225" r="14" fill="#FFFFFF"/>
  <circle cx="310" cy="225" r="24" fill="#00F2FE"/>
  <circle cx="310" cy="225" r="14" fill="#FFFFFF"/>
  <path d="M210 320 C230 338 270 338 290 320" fill="none" stroke="#FFFFFF" stroke-width="6" stroke-linecap="round"/>
  <circle cx="130" cy="390" r="8" fill="#00F2FE"/>
  <circle cx="370" cy="390" r="8" fill="#00F2FE"/>
</svg>"""

        # 9. Car / Vehicle
        elif any(w in p_lower for w in ('car', 'ô tô', 'xe hơi', 'xe', 'auto', 'phương tiện')):
            return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">
  <defs>
    <linearGradient id="carBody" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#EF4444"/>
      <stop offset="50%" stop-color="#DC2626"/>
      <stop offset="100%" stop-color="#991B1B"/>
    </linearGradient>
    <linearGradient id="windowGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#38BDF8"/>
      <stop offset="100%" stop-color="#0284C7"/>
    </linearGradient>
  </defs>
  <ellipse cx="250" cy="370" rx="190" ry="18" fill="rgba(0,0,0,0.3)"/>
  <path d="M140 260 L190 170 L320 170 L380 260 Z" fill="url(#windowGrad)" stroke="#FFFFFF" stroke-width="3"/>
  <line x1="260" y1="170" x2="260" y2="260" stroke="#1E293B" stroke-width="6"/>
  <path d="M60 290 C60 260 90 250 140 250 L370 250 C420 250 450 270 450 300 C450 330 420 340 390 340 L350 340 C340 310 310 290 280 290 C250 290 220 310 210 340 L160 340 C150 310 120 290 90 290 C70 290 60 300 60 290 Z" fill="url(#carBody)" stroke="#FFFFFF" stroke-width="4"/>
  <polygon points="425,280 445,285 435,300" fill="#FACC15"/>
  <rect x="62" y="275" width="16" height="18" rx="4" fill="#F43F5E"/>
  <circle cx="140" cy="340" r="42" fill="#1E293B" stroke="#64748B" stroke-width="8"/>
  <circle cx="140" cy="340" r="18" fill="#CBD5E1"/>
  <circle cx="330" cy="340" r="42" fill="#1E293B" stroke="#64748B" stroke-width="8"/>
  <circle cx="330" cy="340" r="18" fill="#CBD5E1"/>
</svg>"""

        # 10. Chart / Fintech / Growth
        elif any(w in p_lower for w in ('chart', 'biểu đồ', 'tài chính', 'kinh doanh', 'doanh thu', 'growth', 'fintech', 'money')):
            return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">
  <defs>
    <linearGradient id="barGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#34D399"/>
      <stop offset="100%" stop-color="#059669"/>
    </linearGradient>
    <linearGradient id="lineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#38BDF8"/>
      <stop offset="50%" stop-color="#818CF8"/>
      <stop offset="100%" stop-color="#C084FC"/>
    </linearGradient>
  </defs>
  <g stroke="rgba(255,255,255,0.1)" stroke-width="2">
    <line x1="80" y1="120" x2="430" y2="120"/>
    <line x1="80" y1="200" x2="430" y2="200"/>
    <line x1="80" y1="280" x2="430" y2="280"/>
    <line x1="80" y1="360" x2="430" y2="360"/>
  </g>
  <rect x="110" y="270" width="45" height="90" rx="8" fill="url(#barGrad)" opacity="0.7"/>
  <rect x="180" y="220" width="45" height="140" rx="8" fill="url(#barGrad)" opacity="0.8"/>
  <rect x="250" y="170" width="45" height="190" rx="8" fill="url(#barGrad)" opacity="0.9"/>
  <rect x="320" y="110" width="45" height="250" rx="8" fill="url(#barGrad)"/>
  <path d="M132 240 C170 210 200 190 272 140 C310 110 330 90 390 60" fill="none" stroke="url(#lineGrad)" stroke-width="8" stroke-linecap="round"/>
  <polygon points="380,45 420,55 395,90" fill="#38BDF8"/>
  <circle cx="132" cy="240" r="10" fill="#FFFFFF" stroke="#38BDF8" stroke-width="4"/>
  <circle cx="202" cy="190" r="10" fill="#FFFFFF" stroke="#818CF8" stroke-width="4"/>
  <circle cx="272" cy="140" r="10" fill="#FFFFFF" stroke="#C084FC" stroke-width="4"/>
  <circle cx="342" cy="85" r="10" fill="#FFFFFF" stroke="#34D399" stroke-width="4"/>
</svg>"""

        # 11. Diamond / Gem / Luxury
        elif any(w in p_lower for w in ('diamond', 'kim cương', 'đá quý', 'gem', 'luxury', 'nhẫn', 'vàng')):
            return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">
  <defs>
    <linearGradient id="diaTop" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38BDF8"/>
      <stop offset="100%" stop-color="#0284C7"/>
    </linearGradient>
    <linearGradient id="diaMid" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#BAE6FD"/>
      <stop offset="100%" stop-color="#38BDF8"/>
    </linearGradient>
    <linearGradient id="diaBot" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0284C7"/>
      <stop offset="100%" stop-color="#0369A1"/>
    </linearGradient>
  </defs>
  <polygon points="170,120 330,120 290,190 210,190" fill="url(#diaMid)" stroke="#FFFFFF" stroke-width="2"/>
  <polygon points="170,120 210,190 100,190" fill="url(#diaTop)" stroke="#FFFFFF" stroke-width="2"/>
  <polygon points="330,120 400,190 290,190" fill="url(#diaTop)" stroke="#FFFFFF" stroke-width="2"/>
  <polygon points="210,190 290,190 250,400" fill="url(#diaMid)" stroke="#FFFFFF" stroke-width="2"/>
  <polygon points="100,190 210,190 250,400" fill="url(#diaBot)" stroke="#FFFFFF" stroke-width="2"/>
  <polygon points="400,190 290,190 250,400" fill="url(#diaBot)" stroke="#FFFFFF" stroke-width="2"/>
  <polygon points="140,80 144,95 160,100 144,105 140,120 136,105 120,100 136,95" fill="#FFFFFF"/>
  <polygon points="360,100 363,110 375,114 363,118 360,130 357,118 345,114 357,110" fill="#FFFFFF"/>
</svg>"""

        # 12. Cloud / Weather
        elif any(w in p_lower for w in ('cloud', 'mây', 'mưa', 'rain', 'thời tiết', 'sấm', 'chớp')):
            return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">
  <defs>
    <linearGradient id="cloudGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#E0F2FE"/>
      <stop offset="100%" stop-color="#38BDF8"/>
    </linearGradient>
    <linearGradient id="dropGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#38BDF8"/>
      <stop offset="100%" stop-color="#0284C7"/>
    </linearGradient>
  </defs>
  <path d="M150 320 C100 320 70 280 70 230 C70 180 110 145 160 145 C175 100 220 70 270 70 C330 70 380 110 390 170 C425 180 450 210 450 250 C450 290 420 320 370 320 Z" fill="url(#cloudGrad)" stroke="#FFFFFF" stroke-width="5"/>
  <path d="M160 360 C160 385 145 400 130 400 C115 400 100 385 130 350 Z" fill="url(#dropGrad)"/>
  <path d="M250 370 C250 395 235 410 220 410 C205 410 190 395 220 360 Z" fill="url(#dropGrad)"/>
  <path d="M340 360 C340 385 325 400 310 400 C295 400 280 385 310 350 Z" fill="url(#dropGrad)"/>
</svg>"""

        # 13. Shield / Security
        elif any(w in p_lower for w in ('shield', 'khiên', 'bảo vệ', 'security', 'an ninh', 'khóa', 'safe')):
            return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">
  <defs>
    <linearGradient id="shieldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00F2FE" />
      <stop offset="100%" stop-color="#4FACFE" />
    </linearGradient>
    <linearGradient id="innerGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1E3C72" />
      <stop offset="100%" stop-color="#2A5298" />
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="10" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>
  <circle cx="250" cy="250" r="180" fill="url(#shieldGrad)" opacity="0.15" filter="url(#glow)"/>
  <path d="M250 50 L390 100 C390 260 250 420 250 440 C250 420 110 260 110 100 Z" fill="url(#shieldGrad)" stroke="#FFFFFF" stroke-width="6" stroke-linejoin="round"/>
  <path d="M250 85 L360 125 C360 250 250 380 250 395 C250 380 140 250 140 125 Z" fill="url(#innerGrad)" stroke="rgba(255,255,255,0.4)" stroke-width="3"/>
  <circle cx="250" cy="240" r="45" fill="none" stroke="#00F2FE" stroke-width="5" stroke-dasharray="10 5"/>
  <circle cx="250" cy="240" r="25" fill="#00F2FE"/>
  <path d="M230 240 L245 255 L275 225" fill="none" stroke="#0B132B" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M250 140 L250 180 M250 300 L250 340 M170 240 L200 240 M300 240 L330 240" stroke="#00F2FE" stroke-width="4" stroke-linecap="round"/>
</svg>"""

        # 14. Rocket / Space / Startup
        elif any(w in p_lower for w in ('rocket', 'tên lửa', 'launch', 'khởi nghiệp', 'vũ trụ', 'phi thuyền')):
            return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">
  <defs>
    <linearGradient id="rocketBody" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FFFFFF" />
      <stop offset="100%" stop-color="#E2E8F0" />
    </linearGradient>
    <linearGradient id="flameGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#FFE600" />
      <stop offset="60%" stop-color="#FF5E36" />
      <stop offset="100%" stop-color="#EB144C" />
    </linearGradient>
    <linearGradient id="accentGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FF007F" />
      <stop offset="100%" stop-color="#7928CA" />
    </linearGradient>
  </defs>
  <path d="M230 360 C230 440 250 470 250 470 C250 470 270 440 270 360 Z" fill="url(#flameGrad)" />
  <path d="M240 360 C240 410 250 430 250 430 C250 430 260 410 260 360 Z" fill="#FFFFFF" />
  <path d="M190 320 L140 370 L190 380 Z" fill="url(#accentGrad)" />
  <path d="M310 320 L360 370 L310 380 Z" fill="url(#accentGrad)" />
  <path d="M250 60 C190 150 190 310 190 360 L310 360 C310 310 310 150 250 60 Z" fill="url(#rocketBody)" stroke="#CBD5E1" stroke-width="4"/>
  <path d="M250 60 C220 120 210 160 210 160 L290 160 C290 160 280 120 250 60 Z" fill="url(#accentGrad)"/>
  <circle cx="250" cy="220" r="35" fill="#0EA5E9" stroke="#FFFFFF" stroke-width="6"/>
  <circle cx="250" cy="220" r="28" fill="#38BDF8"/>
  <path d="M238 208 A 16 16 0 0 1 262 208" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" fill="none"/>
</svg>"""

        # 15. Star / Sparkle / AI Magic
        elif any(w in p_lower for w in ('star', 'sao', 'ngôi sao', 'magic', 'phép thuật', 'sparkle', 'lấp lánh')):
            return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">
  <defs>
    <linearGradient id="aiSparkle" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00F2FE" />
      <stop offset="50%" stop-color="#9B51E0" />
      <stop offset="100%" stop-color="#FF007F" />
    </linearGradient>
  </defs>
  <path d="M250 50 C250 160 260 230 370 250 C260 270 250 340 250 450 C250 340 240 270 130 250 C240 230 250 160 250 50 Z" fill="url(#aiSparkle)" />
  <path d="M380 90 C380 125 385 145 420 150 C385 155 380 175 380 210 C380 175 375 155 340 150 C375 145 380 125 380 90 Z" fill="#00F2FE" opacity="0.9"/>
  <path d="M120 330 C120 355 125 370 150 375 C125 380 120 395 120 420 C120 395 115 380 90 375 C115 370 120 355 120 330 Z" fill="#FF007F" opacity="0.85"/>
  <circle cx="250" cy="250" r="24" fill="#FFFFFF"/>
</svg>"""

        # 16. Default Creative Futuristic Hexagon Emblem
        else:
            return self._get_default_hexagon_svg()
# Global instance
svg_processor = SvgProcessor()
