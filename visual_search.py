"""
Visual Search Engine for Clarity AI Studio
Provides:
1. Text-to-Image Semantic Search (Keyword search using OpenAI CLIP ViT-B/32)
2. Image-to-Image Visual Similarity Search (Reverse image search)
3. Incremental NPZ Feature Caching with Automatic Re-indexing
4. Zero-disruption integration with existing Web UI and Chatbot
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
SAMPLES_DIR = os.path.join(BASE_DIR, 'static', 'samples')

INDEX_CACHE_FILE = os.path.join(OUTPUT_DIR, 'clip_visual_index.npz')
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


class VisualSearchEngine:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._is_loading = False

        # In-memory index cache
        self.file_paths: List[str] = []
        self.file_mtimes: List[float] = []
        self.embeddings: Optional[np.ndarray] = None  # Shape: (N, 512)
        self.metadata_list: List[Dict[str, Any]] = []

        print(f"[VisualSearch] Initialized engine with device={self.device}, target_model={self.model_id}")

    def load_model(self):
        """Loads CLIP model and processor lazily into VRAM/RAM."""
        if self.model is not None and self.processor is not None:
            return True

        if self._is_loading:
            while self._is_loading:
                time.sleep(0.1)
            return self.model is not None

        self._is_loading = True
        try:
            from transformers import CLIPProcessor, CLIPModel  # type: ignore

            print(f"[VisualSearch] Loading CLIP model ({self.model_id}) on {self.device}...")
            start_t = time.time()
            self.processor = CLIPProcessor.from_pretrained(self.model_id, use_fast=False)
            self.model = CLIPModel.from_pretrained(self.model_id, use_safetensors=True).to(self.device)
            self.model.eval()
            elapsed = round(time.time() - start_t, 2)
            print(f"[VisualSearch] CLIP model successfully loaded in {elapsed}s on {self.device}!")
            return True
        except Exception as e:
            print(f"[VisualSearch] Error loading CLIP model: {e}")
            return False
        finally:
            self._is_loading = False

    def scan_library_files(self) -> List[Dict[str, Any]]:
        """Scans outputs, uploads, and static/samples for all valid images."""
        found_files = []
        directories = [
            (OUTPUT_DIR, "outputs", "/outputs/"),
            (UPLOAD_DIR, "uploads", "/uploads/"),
            (SAMPLES_DIR, "samples", "/static/samples/")
        ]

        for dir_path, category, url_prefix in directories:
            if not os.path.exists(dir_path):
                continue
            for fname in os.listdir(dir_path):
                ext = os.path.splitext(fname)[1].lower()
                if ext in ALLOWED_EXTENSIONS:
                    full_p = os.path.join(dir_path, fname)
                    # Ignore zero-byte or temporary zip files
                    try:
                        sz = os.path.getsize(full_p)
                        if sz < 100:
                            continue
                        # Quick validation to filter out corrupt/incomplete files
                        with Image.open(full_p) as test_im:
                            w, h = test_im.size
                            if w < 10 or h < 10:
                                continue
                        mtime = os.path.getmtime(full_p)
                        found_files.append({
                            "path": os.path.abspath(full_p),
                            "filename": fname,
                            "category": category,
                            "url": f"{url_prefix}{fname}",
                            "size_kb": round(sz / 1024, 1),
                            "width": w,
                            "height": h,
                            "mtime": mtime
                        })
                    except Exception:
                        continue
        return found_files

    def _extract_image_features(self, pil_images: List[Image.Image]) -> np.ndarray:
        """Extracts normalized 512-dim CLIP visual embeddings for a batch of PIL Images."""
        self.load_model()
        if self.model is None or self.processor is None:
            raise RuntimeError("CLIP Model not loaded")

        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # L2 normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy().astype(np.float32)

    def _extract_text_features(self, texts: List[str]) -> np.ndarray:
        """Extracts normalized 512-dim CLIP text embeddings for a list of queries."""
        self.load_model()
        if self.model is None or self.processor is None:
            raise RuntimeError("CLIP Model not loaded")

        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # L2 normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy().astype(np.float32)

    def build_or_update_index(self, force: bool = False) -> Dict[str, Any]:
        """
        Builds or incrementally updates image embeddings cache using NPZ.
        Re-encodes only new or modified images.
        """
        start_time = time.time()
        scanned = self.scan_library_files()
        scanned_map = {item["path"]: item for item in scanned}

        cached_embeddings_map = {}
        cached_mtimes_map = {}
        cached_meta_map = {}

        if not force and os.path.exists(INDEX_CACHE_FILE):
            try:
                data = np.load(INDEX_CACHE_FILE, allow_pickle=True)
                c_paths = data.get("paths", [])
                c_mtimes = data.get("mtimes", [])
                c_embeds = data.get("embeddings", [])
                c_meta_raw = str(data.get("metadata_json", "{}"))
                c_meta_dict = json.loads(c_meta_raw) if c_meta_raw else {}

                for i, p in enumerate(c_paths):
                    p_str = str(p)
                    cached_embeddings_map[p_str] = c_embeds[i]
                    cached_mtimes_map[p_str] = float(c_mtimes[i])
                    if p_str in c_meta_dict:
                        cached_meta_map[p_str] = c_meta_dict[p_str]
            except Exception as e:
                print(f"[VisualSearch] Warning: Could not read index cache ({e}), rebuilding fresh.")

        # Identify files needing encoding
        to_encode_items = []
        final_paths = []
        final_mtimes = []
        final_embeds_list = []
        final_meta = {}

        for p, item in scanned_map.items():
            if p in cached_embeddings_map and abs(cached_mtimes_map.get(p, 0) - item["mtime"]) < 2.0:
                # Use cached embedding
                final_paths.append(p)
                final_mtimes.append(item["mtime"])
                final_embeds_list.append(cached_embeddings_map[p])
                meta = cached_meta_map.get(p, item)
                final_meta[p] = meta
            else:
                to_encode_items.append(item)

        newly_encoded_count = len(to_encode_items)

        if newly_encoded_count > 0:
            print(f"[VisualSearch] Encoding {newly_encoded_count} new/updated images with CLIP...")
            self.load_model()
            batch_size = 32

            for i in range(0, newly_encoded_count, batch_size):
                batch_items = to_encode_items[i:i + batch_size]
                batch_images = []
                valid_batch_items = []

                for it in batch_items:
                    try:
                        with Image.open(it["path"]) as img:
                            rgb_img = img.convert("RGB")
                            w, h = img.size
                            it["width"] = w
                            it["height"] = h
                            batch_images.append(rgb_img)
                            valid_batch_items.append(it)
                    except Exception as err:
                        print(f"[VisualSearch] Skipping unreadable image {it['path']}: {err}")

                if batch_images:
                    feats = self._extract_image_features(batch_images)
                    for j, it in enumerate(valid_batch_items):
                        final_paths.append(it["path"])
                        final_mtimes.append(it["mtime"])
                        final_embeds_list.append(feats[j])
                        final_meta[it["path"]] = it

        if final_embeds_list:
            self.file_paths = final_paths
            self.file_mtimes = final_mtimes
            self.embeddings = np.array(final_embeds_list, dtype=np.float32)
            self.metadata_list = [final_meta[p] for p in final_paths]

            # Save to disk
            try:
                np.savez_compressed(
                    INDEX_CACHE_FILE,
                    paths=np.array(self.file_paths),
                    mtimes=np.array(self.file_mtimes, dtype=np.float64),
                    embeddings=self.embeddings,
                    metadata_json=json.dumps(final_meta)
                )
            except Exception as e:
                print(f"[VisualSearch] Failed to save cache file: {e}")
        else:
            self.file_paths = []
            self.file_mtimes = []
            self.embeddings = np.empty((0, 512), dtype=np.float32)
            self.metadata_list = []

        elapsed = round(time.time() - start_time, 3)
        res_info = {
            "status": "success",
            "total_indexed": len(self.file_paths),
            "newly_encoded": newly_encoded_count,
            "elapsed_sec": elapsed,
            "device": self.device,
            "model": self.model_id
        }
        print(f"[VisualSearch] Index ready: {len(self.file_paths)} images indexed ({newly_encoded_count} new) in {elapsed}s.")
        return res_info

    def search_by_text(
        self,
        query: str,
        top_k: int = 12,
        min_similarity: float = 0.15
    ) -> List[Dict[str, Any]]:
        """
        Searches images matching text query using cosine similarity between text & image embeddings.
        Supports both Vietnamese & English queries with automatic semantic enhancement.
        """
        q_clean = query.strip()
        if not q_clean:
            return []

        # Ensure index exists
        if self.embeddings is None or len(self.file_paths) == 0:
            self.build_or_update_index()

        if self.embeddings is None or len(self.file_paths) == 0:
            return []

        # Prepare queries: original + smart English translation if query has Vietnamese
        queries = [q_clean]
        try:
            from generator import smart_translate_prompt  # type: ignore
            translated = smart_translate_prompt(q_clean)
            if translated and translated.lower() != q_clean.lower():
                queries.append(translated)
        except Exception:
            pass

        # Also add descriptive prompt prefixes for CLIP accuracy
        expanded_texts = []
        for q in queries:
            expanded_texts.append(q)
            if not q.lower().startswith("a photo of") and not q.lower().startswith("an image of"):
                expanded_texts.append(f"a photo of {q}")
                expanded_texts.append(f"an artistic rendering of {q}")

        text_feats = self._extract_text_features(expanded_texts)  # (M, 512)

        # Dot product with all indexed embeddings (N, 512) @ (512, M) -> (N, M)
        sim_matrix = np.dot(self.embeddings, text_feats.T)
        # Take maximum similarity score across prompt variations
        max_sims = np.max(sim_matrix, axis=1)  # (N,)

        # Sort descending
        sorted_indices = np.argsort(-max_sims)

        results = []
        for idx in sorted_indices:
            raw_score = float(max_sims[idx])
            if raw_score < min_similarity:
                continue

            # Convert cosine score (typically 0.18 - 0.35) into an intuitive percentage (50% - 99%)
            # Baseline offset 0.12, dynamic range 0.22
            norm_score = max(0.0, min(1.0, (raw_score - 0.12) / 0.22))
            pct_score = round(50.0 + norm_score * 49.0, 1)

            item = dict(self.metadata_list[idx])
            item["score"] = raw_score
            item["match_percent"] = pct_score
            item["confidence_badge"] = (
                "Rất cao" if pct_score >= 85 else ("Cao" if pct_score >= 70 else "Tương đối")
            )
            results.append(item)

            if len(results) >= top_k:
                break

        return results

    def search_by_image(
        self,
        image_input: Union[str, Image.Image],
        top_k: int = 12,
        min_similarity: float = 0.20,
        exclude_self_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Reverse image search: finds visually and semantically similar images in the library.
        """
        # Ensure index exists
        if self.embeddings is None or len(self.file_paths) == 0:
            self.build_or_update_index()

        if self.embeddings is None or len(self.file_paths) == 0:
            return []

        # Load input image
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            pil_img = Image.open(image_input).convert("RGB")
            query_abs_path = os.path.abspath(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            query_abs_path = None
        else:
            raise ValueError("image_input must be a filepath string or PIL.Image")

        # Extract normalized embedding
        query_feat = self._extract_image_features([pil_img])[0]  # (512,)

        # Cosine similarities
        sims = np.dot(self.embeddings, query_feat)  # (N,)

        sorted_indices = np.argsort(-sims)

        results = []
        for idx in sorted_indices:
            path = self.file_paths[idx]
            # Exclude self if query is already an indexed file
            if exclude_self_path and os.path.abspath(path) == os.path.abspath(exclude_self_path):
                continue
            if query_abs_path and os.path.abspath(path) == query_abs_path:
                continue

            raw_score = float(sims[idx])
            # If it's a completely identical image (score > 0.999), skip if excluding self
            if raw_score > 0.999 and (exclude_self_path or query_abs_path):
                continue

            if raw_score < min_similarity:
                continue

            # Visual similarity percentage: in image-to-image, cosine ranges 0.3 to 1.0
            pct_score = round(max(0.0, min(99.9, raw_score * 100.0)), 1)

            item = dict(self.metadata_list[idx])
            item["score"] = raw_score
            item["match_percent"] = pct_score
            item["confidence_badge"] = (
                "Rất cao" if pct_score >= 85 else ("Cao" if pct_score >= 70 else "Tương đối")
            )
            results.append(item)

            if len(results) >= top_k:
                break

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Returns current indexing and model health metrics."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "is_model_loaded": self.model is not None,
            "total_indexed": len(self.file_paths) if self.file_paths else 0,
            "cache_file": INDEX_CACHE_FILE,
            "cache_exists": os.path.exists(INDEX_CACHE_FILE)
        }


# Global singleton instance
visual_search_engine = VisualSearchEngine()
