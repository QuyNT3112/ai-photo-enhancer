"""
Flask Web Server for AI Image Enhancer
Runs on http://127.0.0.1:7860
"""

import os
import sys

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

import uuid
import time
import json
import zipfile
from datetime import datetime
from typing import Optional, Dict, Any, List
import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image as PILImage  # type: ignore
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for  # type: ignore
from werkzeug.utils import secure_filename  # type: ignore
from enhancer import ImageEnhancer
from aspect_converter import AspectConverter
from compressor import ImageCompressor
from generator import ImageGenerator
from llm_engine import LLMEngine
from bg_remover import BackgroundRemover
from format_converter import FormatConverter
from visual_search import visual_search_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
BATCH_DIR = os.path.join(OUTPUT_DIR, 'batch')
SAMPLES_DIR = os.path.join(BASE_DIR, 'static', 'samples')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BATCH_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024  # 256 MB max upload for multi-image batching
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# Initialize Engines
print("[Server] Initializing AI & Media Engines...")
enhancer = ImageEnhancer()
aspect_converter = AspectConverter()
compressor = ImageCompressor()
image_generator = ImageGenerator()
llm_engine = LLMEngine()
bg_remover = BackgroundRemover()
format_converter = FormatConverter()
try:
    visual_search_engine.build_or_update_index()
except Exception as _e:
    print(f"[Server] Visual search indexing deferred: {_e}")
print("[Server] AI & Media Engines ready!")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'jfif', 'tiff', 'tif', 'ico', 'pdf', 'avif', 'gif'}

# In-memory history for current session
session_history = []


def allowed_file(filename: Optional[str] = None) -> bool:
    if not filename:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


@app.route('/')
def index():
    return render_template('index.html', v=int(time.time()))


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route('/api/save-to-downloads', methods=['POST'])
def api_save_to_downloads():
    try:
        import shutil
        filename = request.form.get('filename')
        custom_name = request.form.get('custom_name')
        if not filename:
            return jsonify({'status': 'error', 'message': 'Tên tệp không hợp lệ'}), 400
        safe_name = secure_filename(filename)
        src_path = os.path.join(OUTPUT_DIR, safe_name)
        if not os.path.exists(src_path):
            return jsonify({'status': 'error', 'message': 'Không tìm thấy tệp trên máy chủ'}), 404

        downloads_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
        os.makedirs(downloads_dir, exist_ok=True)

        if custom_name:
            target_name = secure_filename(custom_name)
            ext = os.path.splitext(safe_name)[1]
            if not target_name:
                target_name = f"ClarityAI_{safe_name}"
            if not target_name.lower().endswith(ext.lower()):
                target_name += ext
        else:
            if safe_name.startswith('ClarityAI_'):
                target_name = safe_name
            else:
                target_name = f"ClarityAI_{safe_name}"

        dst_path = os.path.join(downloads_dir, target_name)
        shutil.copy2(src_path, dst_path)

        return jsonify({
            'status': 'success',
            'saved_path': dst_path,
            'saved_name': target_name,
            'filename': target_name,
            'message': f'Đã lưu trực tiếp vào thư mục Downloads: {target_name}'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/save-batch-bg-to-downloads', methods=['POST'])
def api_save_batch_bg_to_downloads():
    try:
        import shutil
        data = request.get_json(silent=True) or {}
        items = data.get('items', [])
        if not items:
            filenames = request.form.getlist('filenames')
            orig_names = request.form.getlist('original_filenames')
            items = [{'filename': f, 'original_filename': orig_names[i] if i < len(orig_names) else f} for i, f in enumerate(filenames)]

        if not items:
            return jsonify({'status': 'error', 'message': 'Không có ảnh nào để lưu'}), 400

        downloads_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
        os.makedirs(downloads_dir, exist_ok=True)

        saved_files = []
        for it in items:
            filename = secure_filename(it.get('filename', ''))
            if not filename:
                continue
            src_path = os.path.join(OUTPUT_DIR, filename)
            if not os.path.exists(src_path):
                continue

            orig_name = it.get('original_filename', filename)
            base_root, _ = os.path.splitext(orig_name)
            out_ext = os.path.splitext(filename)[1] or '.png'
            safe_base = secure_filename(base_root)
            if not safe_base:
                safe_base = 'image'
            if safe_base.startswith('ClarityAI_'):
                target_name = f"{safe_base}_nobg{out_ext}"
            else:
                target_name = f"ClarityAI_{safe_base}_nobg{out_ext}"
            dst_path = os.path.join(downloads_dir, target_name)
            shutil.copy2(src_path, dst_path)
            saved_files.append(target_name)

        return jsonify({
            'status': 'success',
            'count': len(saved_files),
            'saved_files': saved_files,
            'message': f'Đã lưu thành công {len(saved_files)} ảnh đã tách nền vào thư mục Downloads!'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/download/<path:filename>')
def api_download(filename):
    safe_name = secure_filename(filename)
    file_path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.exists(file_path):
        return jsonify({'status': 'error', 'message': 'File not found'}), 404

    # Custom friendly name parameter
    custom_name = request.args.get('name')
    ext = os.path.splitext(safe_name)[1].lower()
    if custom_name:
        custom_safe = secure_filename(custom_name)
        if not custom_safe.lower().endswith(ext):
            custom_safe += ext
        download_filename = custom_safe
    else:
        if safe_name.startswith('ClarityAI_'):
            download_filename = safe_name
        else:
            download_filename = f"ClarityAI_{safe_name}"

    # Determine accurate MIME type
    mimetypes_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
        '.ico': 'image/x-icon',
        '.pdf': 'application/pdf',
        '.avif': 'image/avif',
        '.gif': 'image/gif',
        '.zip': 'application/zip'
    }
    mimetype = mimetypes_map.get(ext, 'application/octet-stream')

    return send_from_directory(
        OUTPUT_DIR,
        safe_name,
        as_attachment=True,
        download_name=download_filename,
        mimetype=mimetype
    )


@app.route('/api/download-jpg/<path:filename>')
def api_download_jpg(filename):
    safe_name = secure_filename(filename)
    png_path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.exists(png_path):
        return jsonify({'status': 'error', 'message': 'File not found'}), 404

    jpg_name = os.path.splitext(safe_name)[0] + '.jpg'
    jpg_path = os.path.join(OUTPUT_DIR, jpg_name)
    if not os.path.exists(jpg_path):
        img = cv2.imread(png_path)
        if img is not None:
            cv2.imwrite(jpg_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return send_from_directory(
        OUTPUT_DIR,
        jpg_name,
        as_attachment=True,
        download_name=f"ClarityAI_{jpg_name}",
        mimetype='image/jpeg'
    )


@app.route('/api/convert-aspect', methods=['POST'])
def api_convert_aspect():
    try:
        target_ratio = request.form.get('target_ratio', '1:1')
        mode = request.form.get('mode', 'ai_expand')
        align = request.form.get('align', 'center')
        bg_color = request.form.get('bg_color', 'black')
        resolution_mode = request.form.get('resolution_mode', 'standard')
        apply_enhance = request.form.get('apply_enhance', 'false').lower() in ['true', '1', 'yes']

        input_path = None
        orig_filename = "image.jpg"

        # Check if reusing an existing processed file
        source_filename = request.form.get('source_filename', '')
        if source_filename:
            safe_source = secure_filename(source_filename)
            potential_path = os.path.join(OUTPUT_DIR, safe_source)
            if not os.path.exists(potential_path):
                potential_path = os.path.join(UPLOAD_DIR, safe_source)
            if os.path.exists(potential_path):
                input_path = potential_path
                orig_filename = safe_source

        # Or read newly uploaded image
        if not input_path:
            if 'image' in request.files and request.files['image'].filename != '':
                file = request.files['image']
                if not allowed_file(file.filename):
                    return jsonify({'status': 'error', 'message': 'Định dạng tệp không được hỗ trợ.'}), 400
                file_uuid = uuid.uuid4().hex[:10]
                orig_filename = secure_filename(file.filename or 'image.jpg')
                base_ext = os.path.splitext(orig_filename)[1].lower() or '.jpg'
                input_filename = f"aspect_in_{file_uuid}{base_ext}"
                input_path = os.path.join(UPLOAD_DIR, input_filename)
                file.save(input_path)

        # Or sample
        sample_id = request.form.get('sample_id', '')
        if not input_path and sample_id:
            sample_map = {'portrait': 'sample_portrait_lowres.jpg', 'anime': 'sample_anime_lowres.jpg'}
            if sample_id in sample_map:
                source_sample_path = os.path.join(SAMPLES_DIR, sample_map[sample_id])
                if os.path.exists(source_sample_path):
                    file_uuid = uuid.uuid4().hex[:10]
                    input_path = os.path.join(UPLOAD_DIR, f"aspect_sample_{sample_id}_{file_uuid}.jpg")
                    import shutil
                    shutil.copy2(source_sample_path, input_path)
                    orig_filename = sample_map[sample_id]

        if not input_path or not os.path.exists(input_path):
            return jsonify({'status': 'error', 'message': 'Vui lòng chọn hoặc tải lên một hình ảnh.'}), 400

        output_uuid = uuid.uuid4().hex[:10]
        output_filename = f"aspect_{target_ratio.replace(':', 'x')}_{output_uuid}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Run aspect ratio conversion
        stats = aspect_converter.convert(
            input_path=input_path,
            output_path=output_path,
            target_ratio=target_ratio,
            mode=mode,
            bg_color=bg_color,
            resolution_mode=resolution_mode,
            align=align
        )

        # Optional: Apply AI super-resolution on top if requested
        if apply_enhance:
            enhanced_output = os.path.join(OUTPUT_DIR, f"aspect_enh_{output_uuid}.png")
            enh_stats = enhancer.enhance_image(
                input_path=output_path,
                output_path=enhanced_output,
                model_type='general',
                outscale=2
            )
            output_path = enhanced_output
            output_filename = os.path.basename(enhanced_output)
            stats['final_width'] = enh_stats['final_width']
            stats['final_height'] = enh_stats['final_height']
            stats['duration_sec'] = round(stats['duration_sec'] + enh_stats['duration_sec'], 2)

        file_size_bytes = os.path.getsize(output_path)
        orig_size_bytes = os.path.getsize(input_path)

        result_data = {
            'id': output_uuid,
            'timestamp': datetime.now().strftime('%H:%M:%S - %d/%m/%Y'),
            'original_filename': orig_filename,
            'original_url': f"/uploads/{os.path.basename(input_path)}" if os.path.dirname(input_path) == UPLOAD_DIR else f"/outputs/{os.path.basename(input_path)}",
            'result_url': f"/outputs/{output_filename}",
            'enhanced_url': f"/outputs/{output_filename}",
            'download_url': f"/api/download/{output_filename}",
            'download_jpg_url': f"/api/download-jpg/{output_filename}",
            'filename': output_filename,
            'stats': stats,
            'file_sizes': {
                'original_kb': round(orig_size_bytes / 1024, 1),
                'result_kb': round(file_size_bytes / 1024, 1),
                'result_mb': round(file_size_bytes / (1024 * 1024), 2)
            }
        }

        # Add to history
        session_history.insert(0, result_data)

        return jsonify({'status': 'success', 'data': result_data})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/system-info')
def api_system_info():
    try:
        info = enhancer.get_system_info()
        return jsonify({
            'status': 'success',
            'info': info
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/samples')
def api_samples():
    samples = [
        {
            'id': 'portrait',
            'title': 'Ảnh chân dung cổ điển (mờ & vỡ hạt)',
            'description': 'Ảnh chân dung chụp máy film bị mờ nét, thích hợp phục hồi khuôn mặt và tóc.',
            'filename': 'sample_portrait_lowres.jpg',
            'url': '/static/samples/sample_portrait_lowres.jpg',
            'recommended_model': 'combo',
            'recommended_scale': 4,
            'face_enhance': True
        },
        {
            'id': 'anime',
            'title': 'Hình Anime Cyberpunk (nén thấp)',
            'description': 'Hình minh họa phong cách anime bị giảm chất lượng, cần làm mịn viền và sắc nét chi tiết.',
            'filename': 'sample_anime_lowres.jpg',
            'url': '/static/samples/sample_anime_lowres.jpg',
            'recommended_model': 'anime',
            'recommended_scale': 4,
            'face_enhance': False
        }
    ]
    return jsonify({'status': 'success', 'samples': samples})


@app.route('/api/enhance', methods=['POST'])
def api_enhance():
    try:
        model_type = request.form.get('model_type', 'combo')
        outscale = int(request.form.get('outscale', 4))
        face_enhance = request.form.get('face_enhance', 'true').lower() in ['true', '1', 'yes']
        sample_id = request.form.get('sample_id', '')

        input_path = None
        orig_filename = "image.jpg"

        # Check if sample was requested
        if sample_id:
            sample_map = {
                'portrait': 'sample_portrait_lowres.jpg',
                'anime': 'sample_anime_lowres.jpg'
            }
            if sample_id in sample_map:
                sample_filename = sample_map[sample_id]
                source_sample_path = os.path.join(SAMPLES_DIR, sample_filename)
                if os.path.exists(source_sample_path):
                    # Copy to uploads with unique name
                    file_uuid = uuid.uuid4().hex[:10]
                    input_filename = f"sample_{sample_id}_{file_uuid}.jpg"
                    input_path = os.path.join(UPLOAD_DIR, input_filename)
                    with open(source_sample_path, 'rb') as f_src:
                        with open(input_path, 'wb') as f_dst:
                            f_dst.write(f_src.read())
                    orig_filename = sample_filename

        # If not sample, read uploaded file
        if not input_path:
            if 'image' not in request.files:
                return jsonify({'status': 'error', 'message': 'Không tìm thấy tệp ảnh tải lên!'}), 400

            file = request.files['image']
            if file.filename == '':
                return jsonify({'status': 'error', 'message': 'Chưa chọn tệp ảnh nào!'}), 400

            if not allowed_file(file.filename):
                return jsonify({'status': 'error', 'message': 'Định dạng tệp không được hỗ trợ. Hãy chọn ảnh PNG, JPG, WEBP, BMP.'}), 400

            file_uuid = uuid.uuid4().hex[:10]
            orig_filename = secure_filename(file.filename or 'image.jpg')
            base_ext = os.path.splitext(orig_filename)[1].lower()
            if not base_ext:
                base_ext = '.jpg'
            input_filename = f"input_{file_uuid}{base_ext}"
            input_path = os.path.join(UPLOAD_DIR, input_filename)
            file.save(input_path)

        # Output file path
        output_uuid = uuid.uuid4().hex[:10]
        output_filename = f"enhanced_{output_uuid}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Run AI inference
        t0 = time.time()
        stats = enhancer.enhance_image(
            input_path=input_path,
            output_path=output_path,
            model_type=model_type,
            outscale=outscale,
            face_enhance=face_enhance
        )
        total_time = round(time.time() - t0, 2)
        stats['total_request_time'] = total_time

        # Calculate file sizes
        orig_size_bytes = os.path.getsize(input_path)
        enhanced_size_bytes = os.path.getsize(output_path)

        result_data = {
            'id': output_uuid,
            'timestamp': datetime.now().strftime('%H:%M:%S - %d/%m/%Y'),
            'original_filename': orig_filename,
            'original_url': f"/uploads/{os.path.basename(input_path)}",
            'enhanced_url': f"/outputs/{output_filename}",
            'download_url': f"/api/download/{output_filename}",
            'download_jpg_url': f"/api/download-jpg/{output_filename}",
            'filename': output_filename,
            'stats': stats,
            'file_sizes': {
                'original_kb': round(orig_size_bytes / 1024, 1),
                'enhanced_kb': round(enhanced_size_bytes / 1024, 1),
                'enhanced_mb': round(enhanced_size_bytes / (1024 * 1024), 2)
            }
        }

        # Add to history
        session_history.insert(0, result_data)
        if len(session_history) > 20:
            session_history.pop()

        return jsonify({
            'status': 'success',
            'data': result_data
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/history')
def api_history():
    return jsonify({
        'status': 'success',
        'history': session_history
    })


# =========================================================================
# BACKGROUND REMOVER APIS (Xóa Nền Ảnh AI RMBG-1.4)
# =========================================================================

@app.route('/api/remove-background', methods=['POST'])
def api_remove_background():
    try:
        files = request.files.getlist('images')
        if not files or files[0].filename == '':
            files = request.files.getlist('image')
        if not files or files[0].filename == '':
            return jsonify({'status': 'error', 'message': 'Chưa chọn tệp ảnh nào!'}), 400

        model_name = request.form.get('model_name', 'rmbg-1.4').strip().lower()
        clean_mode = request.form.get('clean_mode', 'deep_clean').strip().lower()
        try:
            sensitivity = float(request.form.get('sensitivity', 0.5))
        except Exception:
            sensitivity = 0.5
        de_fringe = request.form.get('de_fringe', 'true').lower() in ['true', '1', 'yes']
        bg_color = request.form.get('bg_color', 'transparent').strip().lower()
        feather = request.form.get('feather', 'true').lower() in ['true', '1', 'yes']

        # Single file processing
        if len(files) == 1:
            file = files[0]
            if not file.filename or not allowed_file(file.filename):
                return jsonify({'status': 'error', 'message': 'Định dạng tệp không được hỗ trợ. Hãy chọn ảnh PNG, JPG, WEBP, BMP.'}), 400

            file_uuid = uuid.uuid4().hex[:10]
            orig_filename = secure_filename(file.filename or 'image.jpg')
            base_ext = os.path.splitext(orig_filename)[1].lower() or '.jpg'
            input_filename = f"input_bg_{file_uuid}{base_ext}"
            input_path = os.path.join(UPLOAD_DIR, input_filename)
            file.save(input_path)

            output_uuid = uuid.uuid4().hex[:10]
            output_ext = '.png' if bg_color in ['transparent', 'alpha', 'none', ''] else '.jpg'
            output_filename = f"nobg_{output_uuid}{output_ext}"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            t0 = time.time()
            stats = bg_remover.remove_background(
                input_path=input_path,
                output_path=output_path,
                model_name=model_name,
                clean_mode=clean_mode,
                sensitivity=sensitivity,
                de_fringe=de_fringe,
                bg_color=bg_color,
                feather_edge=feather
            )
            total_time = round(time.time() - t0, 2)
            stats['total_request_time'] = total_time

            orig_size_bytes = os.path.getsize(input_path)
            res_size_bytes = os.path.getsize(output_path)

            result_data = {
                'id': output_uuid,
                'timestamp': datetime.now().strftime('%H:%M:%S - %d/%m/%Y'),
                'original_filename': orig_filename,
                'original_url': f"/uploads/{input_filename}",
                'result_url': f"/outputs/{output_filename}",
                'enhanced_url': f"/outputs/{output_filename}",
                'download_url': f"/api/download/{output_filename}",
                'download_jpg_url': f"/api/download-jpg/{output_filename}",
                'filename': output_filename,
                'stats': stats,
                'file_sizes': {
                    'original_kb': round(orig_size_bytes / 1024, 1),
                    'result_kb': round(res_size_bytes / 1024, 1),
                    'result_mb': round(res_size_bytes / (1024 * 1024), 2)
                }
            }

            session_history.insert(0, result_data)
            if len(session_history) > 20:
                session_history.pop()

            return jsonify({'status': 'success', 'data': result_data})

        # Multi-file batch processing
        else:
            import zipfile
            batch_uuid = uuid.uuid4().hex[:10]
            zip_filename = f"ClarityAI_Batch_NoBG_{batch_uuid}.zip"
            zip_path = os.path.join(OUTPUT_DIR, zip_filename)

            results = []
            files_to_zip = []

            for file in files:
                if not file.filename or not allowed_file(file.filename):
                    continue
                f_uuid = uuid.uuid4().hex[:10]
                orig_name = secure_filename(file.filename)
                base_ext = os.path.splitext(orig_name)[1].lower() or '.jpg'
                in_name = f"input_batch_bg_{f_uuid}{base_ext}"
                in_path = os.path.join(UPLOAD_DIR, in_name)
                file.save(in_path)

                out_ext = '.png' if bg_color in ['transparent', 'alpha', 'none', ''] else '.jpg'
                out_name = f"nobg_{f_uuid}{out_ext}"
                out_path = os.path.join(OUTPUT_DIR, out_name)

                try:
                    st = bg_remover.remove_background(
                        input_path=in_path,
                        output_path=out_path,
                        model_name=model_name,
                        clean_mode=clean_mode,
                        sensitivity=sensitivity,
                        de_fringe=de_fringe,
                        bg_color=bg_color,
                        feather_edge=feather
                    )
                    results.append({
                        'filename': out_name,
                        'original_filename': orig_name,
                        'result_url': f"/outputs/{out_name}",
                        'download_url': f"/api/download/{out_name}",
                        'stats': st
                    })
                    base_root, _ = os.path.splitext(orig_name)
                    files_to_zip.append((out_path, f"{base_root}_nobg{out_ext}"))
                except Exception as ex:
                    print(f"[BgRemover] Batch item error: {ex}")

            if not results:
                return jsonify({'status': 'error', 'message': 'Không có ảnh nào xử lý thành công!'}), 500

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path, arcname in files_to_zip:
                    zf.write(file_path, arcname)

            return jsonify({
                'status': 'success',
                'batch': True,
                'count': len(results),
                'results': results,
                'zip_url': f"/outputs/{zip_filename}",
                'zip_filename': zip_filename,
                'download_url': f"/api/download/{zip_filename}"
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# =========================================================================
# BATCH ENHANCE APIS (Làm Nét Nhiều Ảnh Cùng Lúc)
# =========================================================================

@app.route('/api/batch-enhance', methods=['POST'])
def api_batch_enhance():
    try:
        model_type = request.form.get('model_type', 'anime')
        outscale = int(request.form.get('outscale', 1))
        face_enhance = request.form.get('face_enhance', 'false').lower() in ['true', '1', 'yes']

        uploaded_files = request.files.getlist('images')
        if not uploaded_files or len(uploaded_files) == 0:
            return jsonify({'status': 'error', 'message': 'Vui lòng chọn ít nhất 1 ảnh để làm nét.'}), 400

        batch_id = uuid.uuid4().hex[:10]
        batch_folder = os.path.join(BATCH_DIR, f"enhance_{batch_id}")
        os.makedirs(batch_folder, exist_ok=True)

        results = []
        total_start = time.time()
        used_filenames = set()

        for idx, file in enumerate(uploaded_files):
            if not file or not file.filename:
                continue
            if not allowed_file(file.filename):
                continue

            orig_filename = secure_filename(file.filename)
            base_name = os.path.splitext(orig_filename)[0] or f"image_{idx+1}"
            candidate_name = f"enhanced_{base_name}.png"
            counter = 1
            while candidate_name in used_filenames:
                candidate_name = f"enhanced_{base_name}_{counter}.png"
                counter += 1
            used_filenames.add(candidate_name)

            # Save input temp
            ext = os.path.splitext(orig_filename)[1].lower() or '.png'
            in_temp_name = f"raw_{idx}_{orig_filename}"
            in_temp_path = os.path.join(batch_folder, in_temp_name)
            file.save(in_temp_path)

            out_path = os.path.join(batch_folder, candidate_name)

            try:
                stats = enhancer.enhance_image(
                    input_path=in_temp_path,
                    output_path=out_path,
                    model_type=model_type,
                    outscale=outscale,
                    face_enhance=face_enhance
                )

                orig_kb = round(os.path.getsize(in_temp_path) / 1024, 1)
                enh_kb = round(os.path.getsize(out_path) / 1024, 1)

                results.append({
                    'id': f"{batch_id}_{idx}",
                    'batch_id': batch_id,
                    'original_filename': orig_filename,
                    'enhanced_filename': candidate_name,
                    'original_url': f"/outputs/batch/enhance_{batch_id}/{in_temp_name}",
                    'enhanced_url': f"/outputs/batch/enhance_{batch_id}/{candidate_name}",
                    'download_url': f"/api/download-batch-enhance-item/{batch_id}/{candidate_name}",
                    'orig_size_kb': orig_kb,
                    'enhanced_size_kb': enh_kb,
                    'stats': stats
                })
            except Exception as item_err:
                print(f"[BatchEnhance] Error processing {orig_filename}: {item_err}")
                continue

        if not results:
            return jsonify({'status': 'error', 'message': 'Không thể làm nét các ảnh đã chọn. Vui lòng kiểm tra định dạng tệp.'}), 500

        # Create zip archive of all enhanced images
        zip_filename = f"ClarityAI_Batch_Enhanced_{batch_id}.zip"
        zip_path = os.path.join(OUTPUT_DIR, zip_filename)
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            for item in results:
                file_full = os.path.join(batch_folder, item['enhanced_filename'])
                if os.path.exists(file_full):
                    zipf.write(file_full, arcname=item['enhanced_filename'])

        total_duration = round(time.time() - total_start, 2)
        zip_size_bytes = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0

        return jsonify({
            'status': 'success',
            'data': {
                'batch_id': batch_id,
                'total_files': len(results),
                'total_duration_sec': total_duration,
                'avg_duration_sec': round(total_duration / max(1, len(results)), 2),
                'zip_filename': zip_filename,
                'zip_url': f"/api/download-batch-zip/{zip_filename}",
                'zip_size_mb': round(zip_size_bytes / (1024 * 1024), 2),
                'zip_size_kb': round(zip_size_bytes / 1024, 1),
                'model_used': model_type,
                'outscale': outscale,
                'images': results
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/download-batch-enhance-item/<batch_id>/<path:filename>')
def api_download_batch_enhance_item(batch_id, filename):
    safe_batch = secure_filename(batch_id)
    safe_file = secure_filename(filename)
    batch_dir = os.path.join(BATCH_DIR, f"enhance_{safe_batch}")
    file_path = os.path.join(batch_dir, safe_file)
    if not os.path.exists(file_path):
        return jsonify({'status': 'error', 'message': 'Không tìm thấy file ảnh'}), 404
    return send_from_directory(
        batch_dir,
        safe_file,
        as_attachment=True,
        download_name=safe_file,
        mimetype='image/png'
    )


@app.route('/api/save-batch-enhance-item-to-downloads', methods=['POST'])
def api_save_batch_enhance_item_to_downloads():
    try:
        import shutil
        batch_id = request.form.get('batch_id')
        filename = request.form.get('filename')
        if not batch_id or not filename:
            return jsonify({'status': 'error', 'message': 'Tham số không hợp lệ'}), 400
        safe_batch = secure_filename(batch_id)
        safe_file = secure_filename(filename)
        src_path = os.path.join(BATCH_DIR, f"enhance_{safe_batch}", safe_file)
        if not os.path.exists(src_path):
            return jsonify({'status': 'error', 'message': 'Không tìm thấy file ảnh trên máy chủ'}), 404

        downloads_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
        os.makedirs(downloads_dir, exist_ok=True)

        dst_path = os.path.join(downloads_dir, safe_file)
        shutil.copy2(src_path, dst_path)

        return jsonify({
            'status': 'success',
            'saved_path': dst_path,
            'filename': safe_file,
            'message': f'Đã lưu ảnh {safe_file} vào thư mục Downloads'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# =========================================================================
# BATCH RESIZE STUDIO APIS (Chức năng Resize Nhiều Ảnh Hàng Loạt -> ZIP PNG)
# =========================================================================

@app.route('/api/batch-resize', methods=['POST'])
def api_batch_resize():
    try:
        # 1. Nhận kích thước đích
        target_w = int(request.form.get('width', 1080))
        target_h = int(request.form.get('height', 1080))
        mode = request.form.get('mode', 'stretch')  # 'stretch' hoặc 'crop'

        # Kiểm tra tính hợp lệ của kích thước
        if target_w < 16 or target_h < 16 or target_w > 12000 or target_h > 12000:
            return jsonify({
                'status': 'error',
                'message': 'Kích thước phải nằm trong khoảng từ 16 đến 12000 pixel.'
            }), 400

        # 2. Lấy danh sách ảnh tải lên
        uploaded_files = request.files.getlist('images')
        if not uploaded_files or len(uploaded_files) == 0:
            return jsonify({'status': 'error', 'message': 'Vui lòng chọn ít nhất 1 ảnh để resize.'}), 400

        batch_id = uuid.uuid4().hex[:10]
        batch_folder = os.path.join(BATCH_DIR, f"batch_{batch_id}")
        os.makedirs(batch_folder, exist_ok=True)

        processed_images = []
        used_filenames = set()

        for file in uploaded_files:
            if not file or not file.filename:
                continue
            if not allowed_file(file.filename):
                continue

            orig_filename = secure_filename(file.filename)
            base_name = os.path.splitext(orig_filename)[0] or f"image_{len(processed_images)+1}"
            
            # Đảm bảo tên file duy nhất trong thư mục batch
            candidate_name = f"{base_name}.png"
            counter = 1
            while candidate_name in used_filenames:
                candidate_name = f"{base_name}_{counter}.png"
                counter += 1
            used_filenames.add(candidate_name)

            # Đọc ảnh từ stream
            file_bytes = file.read()
            if not file_bytes:
                continue
            np_arr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]

            # Xử lý resize đảm bảo KHÔNG VIỀN ĐEN:
            if mode == 'crop':
                # Center Crop: phóng tỷ lệ sao cho lấp đầy khung rồi cắt trung tâm (không viền đen, không méo)
                scale = max(target_w / orig_w, target_h / orig_h)
                nw = int(round(orig_w * scale))
                nh = int(round(orig_h * scale))
                interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
                scaled = cv2.resize(img, (nw, nh), interpolation=interp)
                
                sx = max(0, (nw - target_w) // 2)
                sy = max(0, (nh - target_h) // 2)
                result_img = scaled[sy:sy + target_h, sx:sx + target_w]
                if result_img.shape[1] != target_w or result_img.shape[0] != target_h:
                    result_img = cv2.resize(result_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                # Mặc định: Giãn vừa khít (Stretch) - 100% không viền đen, khớp đúng target_w x target_h
                interp = cv2.INTER_LANCZOS4 if (target_w > orig_w or target_h > orig_h) else cv2.INTER_AREA
                result_img = cv2.resize(img, (target_w, target_h), interpolation=interp)

            # Lưu ảnh định dạng chuẩn PNG
            save_path = os.path.join(batch_folder, candidate_name)
            cv2.imwrite(save_path, result_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 4])

            file_size_bytes = os.path.getsize(save_path)
            rel_web_path = f"batch/batch_{batch_id}/{candidate_name}"

            processed_images.append({
                'filename': candidate_name,
                'orig_filename': orig_filename,
                'orig_w': orig_w,
                'orig_h': orig_h,
                'target_w': target_w,
                'target_h': target_h,
                'size_kb': round(file_size_bytes / 1024, 1),
                'batch_id': batch_id,
                'url': f"/outputs/{rel_web_path}",
                'download_url': f"/api/download-batch-item/{batch_id}/{candidate_name}"
            })

        if not processed_images:
            return jsonify({'status': 'error', 'message': 'Không có tệp ảnh hợp lệ nào được xử lý.'}), 400

        # Đóng gói toàn bộ ảnh đã resize thành 1 file ZIP
        zip_filename = f"ClarityAI_Batch_{target_w}x{target_h}_{batch_id}.zip"
        zip_path = os.path.join(OUTPUT_DIR, zip_filename)

        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            for item in processed_images:
                file_full_path = os.path.join(batch_folder, item['filename'])
                zipf.write(file_full_path, arcname=item['filename'])

        zip_size_bytes = os.path.getsize(zip_path)

        return jsonify({
            'status': 'success',
            'batch_id': batch_id,
            'total_files': len(processed_images),
            'target_w': target_w,
            'target_h': target_h,
            'zip_filename': zip_filename,
            'zip_url': f"/api/download-batch-zip/{zip_filename}",
            'zip_size_mb': round(zip_size_bytes / (1024 * 1024), 2),
            'zip_size_kb': round(zip_size_bytes / 1024, 1),
            'images': processed_images
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/download-batch-zip/<path:filename>')
def api_download_batch_zip(filename):
    safe_name = secure_filename(filename)
    zip_path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.exists(zip_path):
        return jsonify({'status': 'error', 'message': 'Không tìm thấy tệp ZIP'}), 404
    return send_from_directory(
        OUTPUT_DIR,
        safe_name,
        as_attachment=True,
        download_name=safe_name,
        mimetype='application/zip'
    )


@app.route('/api/save-batch-zip-to-downloads', methods=['POST'])
def api_save_batch_zip_to_downloads():
    try:
        import shutil
        filename = request.form.get('filename')
        if not filename:
            return jsonify({'status': 'error', 'message': 'Tên tệp không hợp lệ'}), 400
        safe_name = secure_filename(filename)
        src_path = os.path.join(OUTPUT_DIR, safe_name)
        if not os.path.exists(src_path):
            return jsonify({'status': 'error', 'message': 'Không tìm thấy tệp ZIP trên máy chủ'}), 404

        downloads_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
        os.makedirs(downloads_dir, exist_ok=True)

        dst_path = os.path.join(downloads_dir, safe_name)
        shutil.copy2(src_path, dst_path)

        return jsonify({
            'status': 'success',
            'saved_path': dst_path,
            'filename': safe_name,
            'message': f'Đã lưu tệp ZIP vào thư mục Downloads: {safe_name}'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/download-batch-item/<batch_id>/<path:filename>')
def api_download_batch_item(batch_id, filename):
    safe_batch = secure_filename(batch_id)
    safe_file = secure_filename(filename)
    batch_dir = os.path.join(OUTPUT_DIR, 'batch', f"batch_{safe_batch}")
    file_path = os.path.join(batch_dir, safe_file)
    if not os.path.exists(file_path):
        return jsonify({'status': 'error', 'message': 'Không tìm thấy file ảnh'}), 404
    return send_from_directory(
        batch_dir,
        safe_file,
        as_attachment=True,
        download_name=safe_file,
        mimetype='image/png'
    )


@app.route('/api/save-batch-item-to-downloads', methods=['POST'])
def api_save_batch_item_to_downloads():
    try:
        import shutil
        batch_id = request.form.get('batch_id')
        filename = request.form.get('filename')
        if not batch_id or not filename:
            return jsonify({'status': 'error', 'message': 'Tham số không hợp lệ'}), 400
        safe_batch = secure_filename(batch_id)
        safe_file = secure_filename(filename)
        src_path = os.path.join(OUTPUT_DIR, 'batch', f"batch_{safe_batch}", safe_file)
        if not os.path.exists(src_path):
            return jsonify({'status': 'error', 'message': 'Không tìm thấy file ảnh trên máy chủ'}), 404

        downloads_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
        os.makedirs(downloads_dir, exist_ok=True)

        dst_path = os.path.join(downloads_dir, safe_file)
        shutil.copy2(src_path, dst_path)

        return jsonify({
            'status': 'success',
            'saved_path': dst_path,
            'filename': safe_file,
            'message': f'Đã lưu ảnh {safe_file} vào thư mục Downloads'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# =========================================================================
# SMART TARGET-MB COMPRESSOR APIS (Nén Dung Lượng Ảnh Tối Ưu Theo Mức MB)
# =========================================================================

@app.route('/api/compress', methods=['POST'])
def api_compress():
    try:
        target_mb = float(request.form.get('target_mb', 1.0))
        format_choice = request.form.get('format_choice', 'auto')

        if target_mb < 0.02 or target_mb > 500.0:
            return jsonify({'status': 'error', 'message': 'Mức dung lượng mục tiêu phải nằm trong khoảng từ 0.02 MB đến 500 MB.'}), 400

        input_path = None
        orig_filename = "image.jpg"

        # 1. Kiểm tra tái sử dụng ảnh từ tab trước
        source_filename = request.form.get('source_filename', '')
        if source_filename:
            safe_source = secure_filename(source_filename)
            potential_path = os.path.join(OUTPUT_DIR, safe_source)
            if not os.path.exists(potential_path):
                potential_path = os.path.join(UPLOAD_DIR, safe_source)
            if os.path.exists(potential_path):
                input_path = potential_path
                orig_filename = safe_source

        # 2. Đọc file tải lên
        if not input_path:
            if 'image' in request.files and request.files['image'].filename != '':
                file = request.files['image']
                if not allowed_file(file.filename):
                    return jsonify({'status': 'error', 'message': 'Định dạng tệp không được hỗ trợ. Hãy chọn PNG, JPG, WEBP, BMP.'}), 400
                file_uuid = uuid.uuid4().hex[:10]
                orig_filename = secure_filename(file.filename or 'image.jpg')
                base_ext = os.path.splitext(orig_filename)[1].lower() or '.jpg'
                input_filename = f"compress_in_{file_uuid}{base_ext}"
                input_path = os.path.join(UPLOAD_DIR, input_filename)
                file.save(input_path)

        # 3. Ảnh mẫu
        sample_id = request.form.get('sample_id', '')
        if not input_path and sample_id:
            sample_map = {'portrait': 'sample_portrait_lowres.jpg', 'anime': 'sample_anime_lowres.jpg'}
            if sample_id in sample_map:
                source_sample_path = os.path.join(SAMPLES_DIR, sample_map[sample_id])
                if os.path.exists(source_sample_path):
                    file_uuid = uuid.uuid4().hex[:10]
                    input_path = os.path.join(UPLOAD_DIR, f"compress_sample_{sample_id}_{file_uuid}.jpg")
                    import shutil
                    shutil.copy2(source_sample_path, input_path)
                    orig_filename = sample_map[sample_id]

        if not input_path or not os.path.exists(input_path):
            return jsonify({'status': 'error', 'message': 'Vui lòng chọn hoặc tải lên một hình ảnh.'}), 400

        output_uuid = uuid.uuid4().hex[:10]
        ext = '.webp' if format_choice in ('webp', 'auto') else ('.jpg' if format_choice in ('jpg', 'jpeg') else '.png')
        output_filename = f"compressed_{target_mb}mb_{output_uuid}{ext}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Thực thi nén thông minh chống loang màu và giữ nét
        stats = compressor.compress(
            input_path=input_path,
            output_path=output_path,
            target_mb=target_mb,
            format_choice=format_choice
        )

        if stats.get('ext') and not output_filename.endswith(stats['ext']):
            base = os.path.splitext(output_filename)[0]
            new_filename = f"{base}{stats['ext']}"
            new_path = os.path.join(OUTPUT_DIR, new_filename)
            if os.path.exists(output_path):
                os.replace(output_path, new_path)
            output_filename = new_filename
            output_path = new_path

        result_data = {
            'id': output_uuid,
            'timestamp': datetime.now().strftime('%H:%M:%S - %d/%m/%Y'),
            'original_filename': orig_filename,
            'original_url': f"/uploads/{os.path.basename(input_path)}" if os.path.dirname(input_path) == UPLOAD_DIR else f"/outputs/{os.path.basename(input_path)}",
            'compressed_url': f"/outputs/{output_filename}",
            'download_url': f"/outputs/{output_filename}",
            'filename': output_filename,
            'stats': stats
        }

        # Lưu vào lịch sử phiên làm việc
        session_history.insert(0, {
            'id': output_uuid,
            'timestamp': result_data['timestamp'],
            'original_filename': f"[Nén {target_mb}MB] {orig_filename}",
            'original_url': result_data['original_url'],
            'enhanced_url': result_data['compressed_url'],
            'download_url': result_data['download_url'],
            'download_jpg_url': result_data['download_url'],
            'filename': output_filename,
            'stats': {
                'orig_width': stats['orig_width'],
                'orig_height': stats['orig_height'],
                'final_width': stats['final_width'],
                'final_height': stats['final_height'],
                'duration_sec': stats['duration_sec']
            },
            'file_sizes': {
                'original_kb': stats['orig_size_kb'],
                'enhanced_kb': stats['final_size_kb'],
                'enhanced_mb': stats['final_size_mb']
            }
        })

        return jsonify({'status': 'success', 'data': result_data})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/save-compressed-to-downloads', methods=['POST'])
def api_save_compressed_to_downloads():
    try:
        import shutil
        filename = request.form.get('filename')
        save_as_name = request.form.get('save_as_name')
        if not filename:
            return jsonify({'status': 'error', 'message': 'Tên tệp không hợp lệ'}), 400
        safe_name = secure_filename(filename)
        src_path = os.path.join(OUTPUT_DIR, safe_name)
        if not os.path.exists(src_path):
            return jsonify({'status': 'error', 'message': 'Không tìm thấy tệp trên máy chủ'}), 404

        downloads_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
        os.makedirs(downloads_dir, exist_ok=True)

        target_name = secure_filename(save_as_name) if save_as_name else f"ClarityAI_{safe_name}"
        dst_path = os.path.join(downloads_dir, target_name)
        shutil.copy2(src_path, dst_path)

        return jsonify({
            'status': 'success',
            'saved_path': dst_path,
            'filename': target_name,
            'message': f'Đã lưu tệp vào thư mục Downloads: {target_name}'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/batch-compress', methods=['POST'])
def api_batch_compress():
    try:
        target_mb = float(request.form.get('target_mb', 1.0))
        format_choice = request.form.get('format_choice', 'auto')
        
        if target_mb < 0.02 or target_mb > 500.0:
            return jsonify({'status': 'error', 'message': 'Mức dung lượng mục tiêu không hợp lệ (0.02 - 500 MB).'}), 400

        uploaded_files = request.files.getlist('images') or request.files.getlist('images[]')
        if not uploaded_files and 'image' in request.files:
            uploaded_files = [request.files['image']]

        sample_id = request.form.get('sample_id', '')
        source_filename = request.form.get('source_filename', '')

        batch_id = uuid.uuid4().hex[:10]
        batch_folder = os.path.join(OUTPUT_DIR, 'compress_batch', f"batch_{batch_id}")
        os.makedirs(batch_folder, exist_ok=True)

        processed_images = []
        used_filenames = set()

        ext_map = {'auto': '.webp', 'webp': '.webp', 'jpg': '.jpg', 'jpeg': '.jpg', 'png': '.png'}
        chosen_ext = ext_map.get(format_choice.lower(), '.webp')

        # 1. Tái sử dụng ảnh từ tab trước nếu có
        if (not uploaded_files or (len(uploaded_files) == 1 and uploaded_files[0].filename == '')) and source_filename:
            safe_src = secure_filename(source_filename)
            src_path = os.path.join(OUTPUT_DIR, safe_src)
            if not os.path.exists(src_path):
                src_path = os.path.join(UPLOAD_DIR, safe_src)
            if os.path.exists(src_path):
                base_name = os.path.splitext(safe_src)[0]
                candidate_name = f"{base_name}{chosen_ext}"
                save_path = os.path.join(batch_folder, candidate_name)
                stats = compressor.compress(src_path, save_path, target_mb=target_mb, format_choice=format_choice, keep_original_resolution=True)
                processed_images.append({
                    'filename': candidate_name,
                    'orig_filename': safe_src,
                    'orig_w': stats['orig_width'],
                    'orig_h': stats['orig_height'],
                    'final_w': stats['final_width'],
                    'final_h': stats['final_height'],
                    'orig_size_mb': stats['orig_size_mb'],
                    'final_size_mb': stats['final_size_mb'],
                    'reduction_percent': stats['reduction_percent'],
                    'quality_used': stats['quality_used'],
                    'batch_id': batch_id,
                    'url': f"/outputs/compress_batch/batch_{batch_id}/{candidate_name}",
                    'download_url': f"/api/download-compress-item/{batch_id}/{candidate_name}"
                })

        # 2. Ảnh mẫu nếu có
        elif (not uploaded_files or (len(uploaded_files) == 1 and uploaded_files[0].filename == '')) and sample_id:
            sample_map = {'portrait': 'sample_portrait_lowres.jpg', 'anime': 'sample_anime_lowres.jpg'}
            if sample_id in sample_map:
                s_name = sample_map[sample_id]
                src_path = os.path.join(SAMPLES_DIR, s_name)
                if os.path.exists(src_path):
                    base_name = os.path.splitext(s_name)[0]
                    candidate_name = f"{base_name}{chosen_ext}"
                    save_path = os.path.join(batch_folder, candidate_name)
                    stats = compressor.compress(src_path, save_path, target_mb=target_mb, format_choice=format_choice, keep_original_resolution=True)
                    processed_images.append({
                        'filename': candidate_name,
                        'orig_filename': s_name,
                        'orig_w': stats['orig_width'],
                        'orig_h': stats['orig_height'],
                        'final_w': stats['final_width'],
                        'final_h': stats['final_height'],
                        'orig_size_mb': stats['orig_size_mb'],
                        'final_size_mb': stats['final_size_mb'],
                        'reduction_percent': stats['reduction_percent'],
                        'quality_used': stats['quality_used'],
                        'batch_id': batch_id,
                        'url': f"/outputs/compress_batch/batch_{batch_id}/{candidate_name}",
                        'download_url': f"/api/download-compress-item/{batch_id}/{candidate_name}"
                    })

        # 3. Duyệt toàn bộ tệp được tải lên
        for file in uploaded_files:
            if not file or not file.filename:
                continue
            if not allowed_file(file.filename):
                continue

            orig_filename = secure_filename(file.filename)
            base_name = os.path.splitext(orig_filename)[0] or f"image_{len(processed_images)+1}"
            
            # GIỮ NGUYÊN TÊN ẢNH GỐC (với định dạng được chọn)
            candidate_name = f"{base_name}{chosen_ext}"
            counter = 1
            while candidate_name in used_filenames:
                candidate_name = f"{base_name}_{counter}{chosen_ext}"
                counter += 1
            used_filenames.add(candidate_name)

            temp_input_path = os.path.join(batch_folder, f"temp_in_{counter}_{orig_filename}")
            file.save(temp_input_path)

            save_path = os.path.join(batch_folder, candidate_name)
            stats = compressor.compress(
                input_path=temp_input_path,
                output_path=save_path,
                target_mb=target_mb,
                format_choice=format_choice,
                keep_original_resolution=True
            )

            try:
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
            except Exception:
                pass

            processed_images.append({
                'filename': candidate_name,
                'orig_filename': orig_filename,
                'orig_w': stats['orig_width'],
                'orig_h': stats['orig_height'],
                'final_w': stats['final_width'],
                'final_h': stats['final_height'],
                'orig_size_mb': stats['orig_size_mb'],
                'final_size_mb': stats['final_size_mb'],
                'reduction_percent': stats['reduction_percent'],
                'quality_used': stats['quality_used'],
                'batch_id': batch_id,
                'url': f"/outputs/compress_batch/batch_{batch_id}/{candidate_name}",
                'download_url': f"/api/download-compress-item/{batch_id}/{candidate_name}"
            })

        if not processed_images:
            return jsonify({'status': 'error', 'message': 'Không có tệp ảnh hợp lệ nào được tải lên.'}), 400

        # Đóng gói ZIP trọn bộ ảnh đã nén
        zip_filename = f"ClarityAI_Compressed_{target_mb}MB_{batch_id}.zip"
        zip_path = os.path.join(OUTPUT_DIR, zip_filename)

        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            for item in processed_images:
                file_full_path = os.path.join(batch_folder, item['filename'])
                zipf.write(file_full_path, arcname=item['filename'])

        zip_size_bytes = os.path.getsize(zip_path)
        total_orig_mb = round(sum(item['orig_size_mb'] for item in processed_images), 2)
        total_comp_mb = round(sum(item['final_size_mb'] for item in processed_images), 2)
        total_reduction = round((1 - (total_comp_mb / total_orig_mb)) * 100, 1) if total_orig_mb > 0 else 0

        return jsonify({
            'status': 'success',
            'batch_id': batch_id,
            'total_files': len(processed_images),
            'target_mb': target_mb,
            'format': format_choice,
            'total_orig_mb': total_orig_mb,
            'total_compressed_mb': total_comp_mb,
            'total_reduction_percent': total_reduction,
            'zip_filename': zip_filename,
            'zip_url': f"/api/download-compress-zip/{zip_filename}",
            'zip_size_mb': round(zip_size_bytes / (1024 * 1024), 2),
            'images': processed_images
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/download-compress-zip/<path:filename>')
def api_download_compress_zip(filename):
    safe_name = secure_filename(filename)
    zip_path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.exists(zip_path):
        return jsonify({'status': 'error', 'message': 'Không tìm thấy tệp ZIP'}), 404
    return send_from_directory(
        OUTPUT_DIR,
        safe_name,
        as_attachment=True,
        download_name=safe_name,
        mimetype='application/zip'
    )


@app.route('/api/save-compress-zip-to-downloads', methods=['POST'])
def api_save_compress_zip_to_downloads():
    try:
        import shutil
        filename = request.form.get('filename')
        if not filename:
            return jsonify({'status': 'error', 'message': 'Tên tệp không hợp lệ'}), 400
        safe_name = secure_filename(filename)
        src_path = os.path.join(OUTPUT_DIR, safe_name)
        if not os.path.exists(src_path):
            return jsonify({'status': 'error', 'message': 'Không tìm thấy tệp ZIP trên máy chủ'}), 404

        downloads_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
        os.makedirs(downloads_dir, exist_ok=True)

        dst_path = os.path.join(downloads_dir, safe_name)
        shutil.copy2(src_path, dst_path)

        return jsonify({
            'status': 'success',
            'saved_path': dst_path,
            'filename': safe_name,
            'message': f'Đã lưu tệp ZIP vào thư mục Downloads: {safe_name}'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/download-compress-item/<batch_id>/<path:filename>')
def api_download_compress_item(batch_id, filename):
    safe_batch = secure_filename(batch_id)
    safe_file = secure_filename(filename)
    batch_dir = os.path.join(OUTPUT_DIR, 'compress_batch', f"batch_{safe_batch}")
    file_path = os.path.join(batch_dir, safe_file)
    if not os.path.exists(file_path):
        return jsonify({'status': 'error', 'message': 'Không tìm thấy file ảnh'}), 404
    ext = os.path.splitext(safe_file)[1].lower()
    mime = 'image/png' if ext == '.png' else ('image/jpeg' if ext in ('.jpg', '.jpeg') else 'image/webp')
    return send_from_directory(
        batch_dir,
        safe_file,
        as_attachment=True,
        download_name=safe_file,
        mimetype=mime
    )


@app.route('/api/save-compress-item-to-downloads', methods=['POST'])
def api_save_compress_item_to_downloads():
    try:
        import shutil
        batch_id = request.form.get('batch_id')
        filename = request.form.get('filename')
        if not batch_id or not filename:
            return jsonify({'status': 'error', 'message': 'Tham số không hợp lệ'}), 400
        safe_batch = secure_filename(batch_id)
        safe_file = secure_filename(filename)
        src_path = os.path.join(OUTPUT_DIR, 'compress_batch', f"batch_{safe_batch}", safe_file)
        if not os.path.exists(src_path):
            return jsonify({'status': 'error', 'message': 'Không tìm thấy file ảnh trên máy chủ'}), 404

        downloads_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
        os.makedirs(downloads_dir, exist_ok=True)

        dst_path = os.path.join(downloads_dir, safe_file)
        shutil.copy2(src_path, dst_path)

        return jsonify({
            'status': 'success',
            'saved_path': dst_path,
            'filename': safe_file,
            'message': f'Đã lưu ảnh {safe_file} vào thư mục Downloads'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ==========================================
# Image Format Converter Endpoints
# ==========================================

@app.route('/api/convert-format', methods=['POST'])
def api_convert_format():
    try:
        files = request.files.getlist('images')
        if not files or files[0].filename == '':
            files = request.files.getlist('image')
        if not files or files[0].filename == '':
            return jsonify({'status': 'error', 'message': 'Chưa chọn tệp ảnh nào!'}), 400

        target_format = request.form.get('target_format', 'png').lower().strip()
        quality_mode = request.form.get('quality_mode', 'lossless').lower().strip()
        bg_color = request.form.get('bg_color', 'white').strip()

        fmt_info = format_converter.SUPPORTED_TARGET_FORMATS.get(target_format, format_converter.SUPPORTED_TARGET_FORMATS['png'])
        out_ext = fmt_info['ext']

        if len(files) == 1:
            file = files[0]
            if not file.filename or not allowed_file(file.filename):
                return jsonify({'status': 'error', 'message': 'Định dạng tệp đầu vào không được hỗ trợ'}), 400

            file_uuid = uuid.uuid4().hex[:10]
            orig_filename = secure_filename(file.filename)
            orig_stem, _ = os.path.splitext(orig_filename)
            if not orig_stem:
                orig_stem = "image"

            # Output filename maintains original stem with new extension
            clean_filename = f"{orig_stem}{out_ext}"

            in_ext = os.path.splitext(orig_filename)[1].lower() or '.jpg'
            input_name = f"input_conv_{file_uuid}{in_ext}"
            input_path = os.path.join(UPLOAD_DIR, input_name)
            file.save(input_path)

            output_name = f"conv_{file_uuid}{out_ext}"
            output_path = os.path.join(OUTPUT_DIR, output_name)

            stats = format_converter.convert(
                input_path=input_path,
                output_path=output_path,
                target_format=target_format,
                quality_mode=quality_mode,
                bg_color=bg_color
            )

            result_data = {
                'id': file_uuid,
                'timestamp': datetime.now().strftime('%H:%M:%S - %d/%m/%Y'),
                'original_filename': orig_filename,
                'clean_filename': clean_filename,
                'target_format': target_format.upper(),
                'original_url': f"/uploads/{input_name}",
                'result_url': f"/outputs/{output_name}",
                'download_url': f"/api/download/{output_name}?name={clean_filename}",
                'filename': output_name,
                'stats': stats
            }
            return jsonify({'status': 'success', 'batch': False, 'data': result_data})

        else:
            # Batch conversion
            batch_uuid = uuid.uuid4().hex[:10]
            zip_filename = f"Converted_{target_format.upper()}_{batch_uuid}.zip"
            zip_path = os.path.join(OUTPUT_DIR, zip_filename)

            results = []
            files_to_zip = []
            used_names = {}

            for i, file in enumerate(files):
                if not file.filename or not allowed_file(file.filename):
                    continue

                f_uuid = uuid.uuid4().hex[:10]
                orig_filename = secure_filename(file.filename)
                orig_stem, _ = os.path.splitext(orig_filename)
                if not orig_stem:
                    orig_stem = f"image_{i+1}"

                # Deduplicate if duplicate names exist in the batch
                count = used_names.get(orig_stem, 0)
                used_names[orig_stem] = count + 1
                if count > 0:
                    clean_filename = f"{orig_stem}_{count}{out_ext}"
                else:
                    clean_filename = f"{orig_stem}{out_ext}"

                in_ext = os.path.splitext(orig_filename)[1].lower() or '.jpg'
                in_name = f"input_batch_conv_{f_uuid}{in_ext}"
                in_path = os.path.join(UPLOAD_DIR, in_name)
                file.save(in_path)

                out_name = f"conv_batch_{f_uuid}{out_ext}"
                out_path = os.path.join(OUTPUT_DIR, out_name)

                try:
                    st = format_converter.convert(
                        input_path=in_path,
                        output_path=out_path,
                        target_format=target_format,
                        quality_mode=quality_mode,
                        bg_color=bg_color
                    )
                    results.append({
                        'filename': out_name,
                        'original_filename': orig_filename,
                        'clean_filename': clean_filename,
                        'result_url': f"/outputs/{out_name}",
                        'download_url': f"/api/download/{out_name}?name={clean_filename}",
                        'stats': st
                    })
                    files_to_zip.append((out_path, clean_filename))
                except Exception as ex:
                    print(f"[FormatConverter] Error converting {orig_filename}: {ex}")

            if not results:
                return jsonify({'status': 'error', 'message': 'Không có ảnh nào chuyển đổi thành công!'}), 500

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for fpath, arcname in files_to_zip:
                    zf.write(fpath, arcname)

            return jsonify({
                'status': 'success',
                'batch': True,
                'count': len(results),
                'target_format': target_format.upper(),
                'results': results,
                'zip_url': f"/outputs/{zip_filename}",
                'zip_filename': zip_filename,
                'download_url': f"/api/download/{zip_filename}"
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/save-batch-convert-to-downloads', methods=['POST'])
def api_save_batch_convert_to_downloads():
    try:
        import shutil
        data = request.get_json(silent=True) or {}
        items = data.get('items', [])
        if not items:
            return jsonify({'status': 'error', 'message': 'Không có tệp nào để lưu'}), 400

        downloads_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
        os.makedirs(downloads_dir, exist_ok=True)

        saved_files = []
        for it in items:
            filename = secure_filename(it.get('filename', ''))
            custom_name = it.get('clean_filename') or it.get('original_filename', '')
            if not filename:
                continue
            src_path = os.path.join(OUTPUT_DIR, filename)
            if not os.path.exists(src_path):
                continue

            ext = os.path.splitext(filename)[1]
            if custom_name:
                stem = os.path.splitext(custom_name)[0]
                target_name = f"{secure_filename(stem)}{ext}"
                if not target_name or target_name == ext:
                    target_name = filename
            else:
                target_name = filename

            dst_path = os.path.join(downloads_dir, target_name)
            shutil.copy2(src_path, dst_path)
            saved_files.append(target_name)

        return jsonify({
            'status': 'success',
            'count': len(saved_files),
            'saved_files': saved_files,
            'message': f'Đã lưu thành công {len(saved_files)} ảnh đã chuyển đổi vào thư mục Downloads!'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ==========================================
# AI Image Generator (T2I & I2I) Endpoints
# ==========================================

@app.route('/api/generator-meta')
def api_generator_meta():
    try:
        return jsonify({
            'status': 'success',
            'models': image_generator.get_models_info(),
            'styles': image_generator.get_style_presets(),
            'random_prompt': image_generator.get_random_prompt()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/generator-random-prompt')
def api_generator_random_prompt():
    try:
        return jsonify({
            'status': 'success',
            'data': image_generator.get_random_prompt()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/translate-prompt', methods=['POST'])
def api_translate_prompt():
    """Translates and formats prompt for user inspection and fine-tuning."""
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        prompt_text = data.get('prompt', '').strip()
        style_preset = data.get('style_preset', 'auto').strip()
        model_id = data.get('model_id', 'runwayml/stable-diffusion-v1-5').strip()
        negative_prompt = data.get('negative_prompt', '').strip()

        if not prompt_text:
            return jsonify({'status': 'error', 'message': 'Vui lòng nhập nội dung prompt!'}), 400

        from generator import smart_translate_prompt
        prompt_en = smart_translate_prompt(prompt_text).strip()
        full_positive, full_negative, _, effective_style = image_generator._format_prompt(
            prompt=prompt_text,
            style_key=style_preset,
            negative_prompt=negative_prompt,
            model_id=model_id
        )

        return jsonify({
            'status': 'success',
            'data': {
                'original': prompt_text,
                'translated': prompt_en,
                'full_positive': full_positive,
                'full_negative': full_negative,
                'effective_style': effective_style
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/generate-image', methods=['POST'])
def api_generate_image():
    try:
        mode = request.form.get('mode', 't2i').strip().lower()
        prompt = request.form.get('prompt', '').strip()
        negative_prompt = request.form.get('negative_prompt', '').strip()
        style_preset = request.form.get('style_preset', 'auto').strip()
        model_id = request.form.get('model_id', 'runwayml/stable-diffusion-v1-5').strip()
        aspect_ratio = request.form.get('aspect_ratio', '1:1').strip()
        
        try:
            steps = int(request.form.get('steps', 25))
        except (ValueError, TypeError):
            steps = 25

        try:
            cfg_scale = float(request.form.get('cfg_scale', 8.0))
        except (ValueError, TypeError):
            cfg_scale = 8.0

        try:
            seed = int(request.form.get('seed', -1))
        except (ValueError, TypeError):
            seed = -1

        if not prompt:
            return jsonify({'status': 'error', 'message': 'Vui lòng nhập mô tả ảnh (Prompt)!'}), 400

        output_uuid = uuid.uuid4().hex[:10]
        output_filename = f"gen_{mode}_{output_uuid}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        if mode == 'i2i':
            try:
                strength = float(request.form.get('strength', 0.45))
            except (ValueError, TypeError):
                strength = 0.45

            input_img = None
            orig_filename = "init_image.jpg"

            # Check if source_filename provided (reusing output/upload from another tab)
            source_filename = request.form.get('source_filename', '').strip()
            if source_filename:
                safe_src = secure_filename(source_filename)
                p1 = os.path.join(OUTPUT_DIR, safe_src)
                p2 = os.path.join(UPLOAD_DIR, safe_src)
                target_src = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)
                if target_src:
                    input_img = PILImage.open(target_src).convert('RGB')
                    orig_filename = safe_src

            # Check if uploaded file provided
            if input_img is None and 'image' in request.files and request.files['image'].filename != '':
                file = request.files['image']
                input_img = PILImage.open(file.stream).convert('RGB')
                orig_filename = secure_filename(file.filename or 'image.jpg')

            # Check if sample image provided
            sample_id = request.form.get('sample_id', '').strip()
            if input_img is None and sample_id:
                sample_map = {'portrait': 'sample_portrait_lowres.jpg', 'anime': 'sample_anime_lowres.jpg'}
                if sample_id in sample_map:
                    sample_file = os.path.join(SAMPLES_DIR, sample_map[sample_id])
                    if os.path.exists(sample_file):
                        input_img = PILImage.open(sample_file).convert('RGB')
                        orig_filename = sample_map[sample_id]

            if input_img is None:
                return jsonify({'status': 'error', 'message': 'Vui lòng chọn hoặc tải lên một hình ảnh nguồn cho chế độ Image-to-Image!'}), 400

            result_img, meta = image_generator.generate_image_to_image(
                input_image=input_img,
                prompt=prompt,
                negative_prompt=negative_prompt,
                style_preset=style_preset,
                aspect_ratio=aspect_ratio,
                strength=strength,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                model_id=model_id
            )
        else:
            # Text-to-Image
            result_img, meta = image_generator.generate_text_to_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                style_preset=style_preset,
                aspect_ratio=aspect_ratio,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                model_id=model_id
            )

        # Save output image
        result_img.save(output_path, format="PNG")
        file_size_bytes = os.path.getsize(output_path)

        result_data = {
            'id': output_uuid,
            'timestamp': datetime.now().strftime('%H:%M:%S - %d/%m/%Y'),
            'mode': mode,
            'result_url': f"/outputs/{output_filename}",
            'download_url': f"/api/download/{output_filename}",
            'download_jpg_url': f"/api/download-jpg/{output_filename}",
            'filename': output_filename,
            'meta': meta,
            'file_sizes': {
                'result_kb': round(file_size_bytes / 1024, 1),
                'result_mb': round(file_size_bytes / (1024 * 1024), 2)
            }
        }

        # Insert to session history
        session_history.insert(0, {
            'id': output_uuid,
            'timestamp': result_data['timestamp'],
            'original_filename': prompt[:30] + ('...' if len(prompt) > 30 else ''),
            'original_url': result_data['result_url'],
            'enhanced_url': result_data['result_url'],
            'download_url': result_data['download_url'],
            'download_jpg_url': result_data['download_jpg_url'],
            'filename': output_filename,
            'stats': {
                'model_name': f"AI Generator ({mode.upper()})",
                'final_width': meta['width'],
                'final_height': meta['height'],
                'duration_sec': meta['duration_sec']
            },
            'file_sizes': result_data['file_sizes']
        })

        return jsonify({'status': 'success', 'data': result_data})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ============================================================================
# LLM, PROMPT OPTIMIZER & CHATBOT API ROUTES
# ============================================================================

@app.route('/api/llm/status', methods=['GET'])
def api_llm_status():
    """Returns local LLM providers status, available models, and active engine."""
    try:
        status_info = llm_engine.get_providers_status()
        return jsonify({'status': 'success', 'data': status_info})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/llm/config', methods=['POST'])
def api_llm_config():
    """Switches active provider (built_in, ollama, lm_studio) or updates host/model."""
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        provider = data.get('provider', 'built_in')
        config = {
            'ollama_host': data.get('ollama_host', 'http://127.0.0.1:11434'),
            'ollama_model': data.get('ollama_model', 'llama3.2'),
            'lm_studio_host': data.get('lm_studio_host', 'http://127.0.0.1:1234/v1'),
            'lm_studio_model': data.get('lm_studio_model', 'local-model')
        }
        res = llm_engine.set_provider_config(provider, config)
        return jsonify({'status': 'success', 'data': res})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/llm/optimize-prompt', methods=['POST'])
def api_llm_optimize_prompt():
    """Analyzes natural language and outputs a production-ready Stable Diffusion prompt."""
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        prompt_text = data.get('prompt', '').strip()
        style_preset = data.get('style_preset', 'cinematic')
        aspect_ratio = data.get('aspect_ratio', '1:1')

        if not prompt_text:
            return jsonify({'status': 'error', 'message': 'Vui lòng nhập nội dung mô tả hoặc ý tưởng ảnh.'}), 400

        result = llm_engine.optimize_prompt(
            user_input=prompt_text,
            style_preset=style_preset,
            aspect_ratio=aspect_ratio
        )
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/llm/image-to-prompt', methods=['POST'])
def api_llm_image_to_prompt():
    """Extracts dominant colors, lighting, art style, and generates a reverse prompt from an image."""
    try:
        prompt_hint = request.form.get('prompt_hint', '')
        image_path = None

        # Check for uploaded file
        if 'image' in request.files and request.files['image'] and request.files['image'].filename:
            file = request.files['image']
            fname = file.filename or 'image.jpg'
            if not allowed_file(fname):
                return jsonify({'status': 'error', 'message': 'Định dạng ảnh không được hỗ trợ'}), 400
            file_ext = fname.rsplit('.', 1)[1].lower() if '.' in fname else 'jpg'
            temp_filename = f"vision_{uuid.uuid4().hex[:10]}.{file_ext}"
            image_path = os.path.join(UPLOAD_DIR, temp_filename)
            file.save(image_path)
        else:
            # Check for image_url
            image_url = request.form.get('image_url') or (request.get_json(silent=True) or {}).get('image_url')
            if image_url:
                clean_url = image_url.split('?')[0]
                if clean_url.startswith('/outputs/'):
                    filename = clean_url.replace('/outputs/', '')
                    image_path = os.path.join(OUTPUT_DIR, secure_filename(filename))
                elif clean_url.startswith('/uploads/'):
                    filename = clean_url.replace('/uploads/', '')
                    image_path = os.path.join(UPLOAD_DIR, secure_filename(filename))
                elif clean_url.startswith('/static/'):
                    rel_path = clean_url.replace('/static/', '').replace('/', os.sep)
                    image_path = os.path.join(BASE_DIR, 'static', rel_path)

        if not image_path or not os.path.exists(image_path):
            return jsonify({'status': 'error', 'message': 'Vui lòng tải lên ảnh hoặc cung cấp URL ảnh hợp lệ.'}), 400

        result = llm_engine.image_to_prompt(image_path, prompt_hint=prompt_hint)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/llm/chat', methods=['POST'])
def api_llm_chat():
    """Handles conversation with the Photography & Color AI Assistant."""
    try:
        message = ''
        image_path = None
        history = []

        if request.is_json:
            data = request.get_json()
            message = data.get('message', '').strip()
            history = data.get('history', [])
            image_url = data.get('image_url')
            if image_url:
                clean_url = image_url.split('?')[0]
                if clean_url.startswith('/outputs/'):
                    image_path = os.path.join(OUTPUT_DIR, secure_filename(clean_url.replace('/outputs/', '')))
                elif clean_url.startswith('/uploads/'):
                    image_path = os.path.join(UPLOAD_DIR, secure_filename(clean_url.replace('/uploads/', '')))
        else:
            message = request.form.get('message', '').strip()
            history_raw = request.form.get('history')
            if history_raw:
                try:
                    history = json.loads(history_raw)
                except Exception:
                    history = []

            if 'image' in request.files and request.files['image'] and request.files['image'].filename:
                file = request.files['image']
                fname = file.filename or 'image.jpg'
                if allowed_file(fname):
                    file_ext = fname.rsplit('.', 1)[1].lower() if '.' in fname else 'jpg'
                    temp_filename = f"chat_{uuid.uuid4().hex[:10]}.{file_ext}"
                    image_path = os.path.join(UPLOAD_DIR, temp_filename)
                    file.save(image_path)
            elif request.form.get('image_url'):
                image_url_val = request.form.get('image_url') or ''
                clean_url = image_url_val.split('?')[0]
                if clean_url.startswith('/outputs/'):
                    image_path = os.path.join(OUTPUT_DIR, secure_filename(clean_url.replace('/outputs/', '')))
                elif clean_url.startswith('/uploads/'):
                    image_path = os.path.join(UPLOAD_DIR, secure_filename(clean_url.replace('/uploads/', '')))

        response_data = llm_engine.chat(
            user_message=message,
            image_path=image_path,
            chat_history=history
        )
        return jsonify(response_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ============================================================================
# VISUAL SEARCH (KEYWORD & REVERSE IMAGE SEARCH WITH CLIP)
# ============================================================================

@app.route('/api/search/text', methods=['POST'])
def api_search_text():
    """Text-to-Image Semantic Search using OpenAI CLIP with Franchise & Entity Recognition."""
    try:
        data = request.get_json() if request.is_json else request.form
        query = (data.get('query') or '').strip()
        if not query:
            return jsonify({'status': 'error', 'message': 'Vui lòng nhập từ khóa tìm kiếm'}), 400

        limit = int(data.get('limit') or 18)
        min_score = float(data.get('min_score') or 0.15)
        category_filter = data.get('category') or 'all'
        scope = data.get('scope') or 'all'  # 'all', 'web', 'local'
        web_source = data.get('web_source') or 'all'  # 'all', 'bing', 'wallhaven', 'safebooru', 'wikimedia', 'flickr'

        try:
            from entity_knowledge import detect_franchise_entity
            entity_info = detect_franchise_entity(query)
        except Exception:
            entity_info = None

        results = visual_search_engine.search_by_text(
            query=query,
            top_k=limit,
            min_similarity=min_score,
            scope=scope,
            web_source=web_source,
            category=category_filter
        )

        return jsonify({
            'status': 'success',
            'query': query,
            'scope': scope,
            'web_source': web_source,
            'entity_info': entity_info,
            'count': len(results),
            'results': results,
            'stats': visual_search_engine.get_stats()
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/search/entity-info', methods=['GET', 'POST'])
def api_search_entity_info():
    """Returns detected franchise knowledge, synopsis, and character lists."""
    try:
        q = request.args.get('query') or (request.get_json(silent=True) or {}).get('query') or request.form.get('query') or ''
        from entity_knowledge import detect_franchise_entity
        entity_info = detect_franchise_entity(q)
        return jsonify({
            'status': 'success',
            'query': q,
            'has_entity': entity_info is not None,
            'entity_info': entity_info
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/search/similar', methods=['POST'])
def api_search_similar():
    """Reverse Image Search: finds visually similar images across local files and the web."""
    try:
        limit = int(request.form.get('limit') or 18)
        min_sim = float(request.form.get('min_similarity') or 0.20)
        category_filter = request.form.get('category') or 'all'
        exclude_self = request.form.get('exclude_self', 'true').lower() == 'true'
        scope = request.form.get('scope') or 'all'  # 'all', 'web', 'local'
        web_source = request.form.get('web_source') or 'all'

        image_path = None
        # Option 1: File uploaded
        if 'image' in request.files and request.files['image'] and request.files['image'].filename:
            f = request.files['image']
            fname = f.filename or 'query.jpg'
            if allowed_file(fname):
                ext = fname.rsplit('.', 1)[1].lower() if '.' in fname else 'jpg'
                temp_name = f"query_{uuid.uuid4().hex[:10]}.{ext}"
                image_path = os.path.join(UPLOAD_DIR, temp_name)
                f.save(image_path)
        # Option 2: Image URL or filename provided
        elif request.form.get('image_url') or request.form.get('filename'):
            raw_url = request.form.get('image_url') or request.form.get('filename') or ''
            clean_name = os.path.basename(raw_url.split('?')[0])
            for d in [OUTPUT_DIR, UPLOAD_DIR, SAMPLES_DIR]:
                candidate = os.path.join(d, clean_name)
                if os.path.exists(candidate):
                    image_path = candidate
                    break

        if not image_path or not os.path.exists(image_path):
            return jsonify({'status': 'error', 'message': 'Không tìm thấy ảnh để tìm kiếm tương tự'}), 400

        results = visual_search_engine.search_by_image(
            image_input=image_path,
            top_k=limit,
            min_similarity=min_sim,
            exclude_self_path=image_path if exclude_self else None,
            scope=scope,
            web_source=web_source
        )

        query_url = f"/uploads/{os.path.basename(image_path)}" if UPLOAD_DIR in image_path else f"/outputs/{os.path.basename(image_path)}"

        return jsonify({
            'status': 'success',
            'query_image': query_url,
            'scope': scope,
            'web_source': web_source,
            'count': len(results),
            'results': results,
            'stats': visual_search_engine.get_stats()
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/search/web/import', methods=['POST'])
def api_search_web_import():
    """Downloads a chosen web image to local uploads/ for immediate use in Enhance, BgRemover, or Generator."""
    try:
        data = request.get_json(silent=True) or request.form
        image_url = data.get('url', '').strip()
        custom_name = data.get('filename', '').strip() or None
        if not image_url:
            return jsonify({'status': 'error', 'message': 'Thiếu đường dẫn URL ảnh cần tải'}), 400

        imported = visual_search_engine.download_web_image_to_local(image_url, custom_name)
        return jsonify({
            'status': 'success',
            'data': imported
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/search/stats', methods=['GET'])
def api_search_stats():
    """Returns visual search index statistics and model status."""
    try:
        return jsonify({
            'status': 'success',
            'stats': visual_search_engine.get_stats()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/search/reindex', methods=['POST'])
def api_search_reindex():
    """Triggers an incremental or forced re-indexing of library images."""
    try:
        force = request.form.get('force', 'false').lower() == 'true'
        res = visual_search_engine.build_or_update_index(force=force)
        return jsonify({
            'status': 'success',
            'info': res
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    port = 7860
    print(f"\n=======================================================")
    print(f"  AI Photo Clarity Studio Server Started!")
    print(f"  URL: http://127.0.0.1:{port}")
    print(f"  GPU Acceleration: Active")
    print(f"=======================================================\n")
    app.run(host='127.0.0.1', port=port, debug=False)
