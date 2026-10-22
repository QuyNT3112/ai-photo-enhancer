"""
Visual Search Engine for Clarity AI Studio
Provides:
1. Text-to-Image Semantic Search (Keyword search using OpenAI CLIP ViT-B/32)
2. Image-to-Image Visual Similarity Search (Reverse image search)
3. Multi-Engine Web Image Search (Bing Web, Wikimedia Commons, Flickr)
4. Local CLIP ViT Visual Re-ranking for Web Candidates
5. Reverse Image Search across the Web using Zero-Shot Visual Concept Probes
6. 1-Click Import of Web Images into Studio Tools (Enhance, Bg Remover, I2I, Chat, Library)
7. Incremental NPZ Feature Caching with Automatic Re-indexing
"""

import os
import sys
import time
import json
import re
import hashlib
import urllib.parse
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

from PIL import Image
import numpy as np
import requests
from bs4 import BeautifulSoup

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import torch

try:
    from entity_knowledge import detect_franchise_entity
except Exception:
    detect_franchise_entity = lambda q: None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
SAMPLES_DIR = os.path.join(BASE_DIR, 'static', 'samples')

INDEX_CACHE_FILE = os.path.join(OUTPUT_DIR, 'clip_visual_index.npz')
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

STANDARD_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# 20 high-coverage zero-shot concept probes for reverse web search
VISUAL_CONCEPT_PROBES = [
    "portrait photo of a beautiful person face",
    "anime manga 2d character illustration",
    "landscape scenic nature mountain river sunset",
    "futuristic cyberpunk neon city street night",
    "mythical fantasy dragon beast creature",
    "supercar luxury sports automobile vehicle",
    "cute cat dog animal pet wildlife",
    "classical fine art oil painting",
    "3d animated character disney pixar render",
    "modern architecture urban skyscraper interior",
    "steampunk retro vintage historical",
    "outer space galaxy stars nebula sci-fi",
    "macro floral flowers botanical plant garden",
    "delicious culinary food dessert meal",
    "dark gothic horror eerie mystery",
    "cozy aesthetic bedroom coffee study",
    "underwater marine ocean sea coral fish",
    "martial arts warrior samurai armor knight",
    "abstract geometric colorful modern graphic art",
    "cybernetic robot cyborg android mecha"
]


class VisualSearchEngine:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._is_loading = False

        # In-memory index cache for local files
        self.file_paths: List[str] = []
        self.file_mtimes: List[float] = []
        self.embeddings: Optional[np.ndarray] = None  # Shape: (N, 512)
        self.metadata_list: List[Dict[str, Any]] = []

        # Precomputed concept probe embeddings
        self._concept_features: Optional[np.ndarray] = None

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
        """Scans outputs, uploads, and static/samples for all valid local images."""
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
                    try:
                        sz = os.path.getsize(full_p)
                        if sz < 100:
                            continue
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
                            "mtime": mtime,
                            "is_web": False,
                            "source_site": "Thư viện máy"
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

    def _get_concept_features(self) -> np.ndarray:
        """Cached embeddings for the zero-shot concept probes."""
        if self._concept_features is None:
            self._concept_features = self._extract_text_features(VISUAL_CONCEPT_PROBES)
        return self._concept_features

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
        cached_metadata_map = {}

        if not force and os.path.exists(INDEX_CACHE_FILE):
            try:
                with np.load(INDEX_CACHE_FILE, allow_pickle=True) as data:
                    paths = list(data["file_paths"])
                    mtimes = list(data["file_mtimes"])
                    embs = data["embeddings"]
                    raw_meta = data["metadata_json"]
                    meta_list = json.loads(str(raw_meta))

                    for p, mt, emb, meta in zip(paths, mtimes, embs, meta_list):
                        cached_embeddings_map[p] = emb
                        cached_mtimes_map[p] = float(mt)
                        cached_metadata_map[p] = meta
            except Exception as e:
                print(f"[VisualSearch] Warning: Could not read cache file ({e}). Starting fresh.")

        valid_paths = []
        valid_mtimes = []
        valid_embeddings = []
        valid_metadata = []

        paths_to_encode = []

        for p, item in scanned_map.items():
            item_mtime = item["mtime"]
            if p in cached_embeddings_map:
                cached_mtime = cached_mtimes_map.get(p, 0.0)
                if abs(cached_mtime - item_mtime) < 2.0:
                    valid_paths.append(p)
                    valid_mtimes.append(item_mtime)
                    valid_embeddings.append(cached_embeddings_map[p])
                    meta = dict(item)
                    valid_metadata.append(meta)
                    continue

            paths_to_encode.append(p)

        newly_encoded_count = 0
        if paths_to_encode:
            print(f"[VisualSearch] Encoding {len(paths_to_encode)} new/modified images with CLIP ViT...")
            batch_size = 16
            for i in range(0, len(paths_to_encode), batch_size):
                batch_paths = paths_to_encode[i:i + batch_size]
                batch_images = []
                loaded_paths = []

                for bp in batch_paths:
                    try:
                        im = Image.open(bp).convert("RGB")
                        batch_images.append(im)
                        loaded_paths.append(bp)
                    except Exception as err:
                        print(f"[VisualSearch] Skip invalid image {bp}: {err}")

                if batch_images:
                    try:
                        batch_feats = self._extract_image_features(batch_images)
                        for lp, feat in zip(loaded_paths, batch_feats):
                            item = scanned_map[lp]
                            valid_paths.append(lp)
                            valid_mtimes.append(item["mtime"])
                            valid_embeddings.append(feat)
                            valid_metadata.append(dict(item))
                            newly_encoded_count += 1
                    except Exception as batch_err:
                        print(f"[VisualSearch] Batch encoding error: {batch_err}")

        if valid_embeddings:
            final_embeddings = np.vstack(valid_embeddings).astype(np.float32)
            self.file_paths = valid_paths
            self.file_mtimes = valid_mtimes
            self.embeddings = final_embeddings
            self.metadata_list = valid_metadata

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            try:
                np.savez_compressed(
                    INDEX_CACHE_FILE,
                    file_paths=np.array(valid_paths),
                    file_mtimes=np.array(valid_mtimes, dtype=np.float64),
                    embeddings=final_embeddings,
                    metadata_json=json.dumps(valid_metadata, ensure_ascii=False)
                )
            except Exception as save_err:
                print(f"[VisualSearch] Error saving cache to NPZ: {save_err}")
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

    # =========================================================================
    # MULTI-ENGINE WEB IMAGE SEARCH (BING, WIKIMEDIA, FLICKR)
    # =========================================================================

    def _search_bing_images(
        self,
        query: str,
        limit: int = 25,
        character_tag: Optional[str] = None,
        character_icon: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Searches Bing Images across the entire internet with high-res filter."""
        results = []
        try:
            url = f"https://www.bing.com/images/search?q={urllib.parse.quote(query)}&qft=+filterui:photo-photo&form=IRFLTR"
            headers = {
                "User-Agent": STANDARD_USER_AGENT,
                "Accept-Language": "en-US,en;q=0.9",
            }
            r = requests.get(url, headers=headers, timeout=6)
            if r.status_code != 200:
                return []

            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", class_="iusc"):
                m_str = a.get("m")
                if not m_str:
                    continue
                try:
                    data = json.loads(m_str)
                    murl = data.get("murl")
                    turl = data.get("turl") or murl
                    purl = data.get("purl") or ""
                    title = data.get("t") or query

                    if murl and murl.startswith("http"):
                        domain = urllib.parse.urlparse(purl).netloc.replace("www.", "") if purl else "Web"
                        lower_title = title.lower()
                        # For anime/dragon searches, ignore stock reptile dragon sites
                        if "dragon" in query.lower() and ("publicdomainpictures.net" in domain or "fairy tale" in lower_title or "chinese dragon" in lower_title or "lizard" in lower_title):
                            continue

                        ext = os.path.splitext(urllib.parse.urlparse(murl).path)[1].lower()
                        if ext not in ALLOWED_EXTENSIONS:
                            ext = ".jpg"
                        synth_fname = f"web_{hashlib.md5(murl.encode()).hexdigest()[:10]}{ext}"

                        results.append({
                            "title": title,
                            "filename": synth_fname,
                            "url": murl,
                            "thumbnail": turl,
                            "source_site": domain,
                            "source_url": purl or murl,
                            "engine": "Bing Web",
                            "is_web": True,
                            "category": "web",
                            "character_tag": character_tag,
                            "character_icon": character_icon
                        })
                        if len(results) >= limit:
                            break
                except Exception:
                    continue
        except Exception as e:
            print(f"[VisualSearch] Bing search error: {e}")
        return results

    def _search_wallhaven_images(
        self,
        query: str,
        limit: int = 16,
        character_tag: Optional[str] = None,
        character_icon: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Searches Wallhaven for ultra-high-resolution 4K/1080p anime and character artwork."""
        results = []
        try:
            url = f"https://wallhaven.cc/api/v1/search?q={urllib.parse.quote(query)}&categories=111&purity=100&sorting=relevance"
            headers = {"User-Agent": STANDARD_USER_AGENT}
            r = requests.get(url, headers=headers, timeout=6)
            if r.status_code == 200:
                data = r.json().get("data", [])
                for it in data[:limit]:
                    full_url = it.get("path")
                    if not full_url:
                        continue
                    thumb_url = it.get("thumbs", {}).get("large") or it.get("thumbs", {}).get("small") or full_url
                    res_str = it.get("resolution", "")
                    w, h = 0, 0
                    if "x" in res_str:
                        try:
                            parts = res_str.split("x")
                            w, h = int(parts[0]), int(parts[1])
                        except Exception:
                            pass
                    synth_fname = f"wh_{it.get('id')}.jpg"
                    results.append({
                        "title": f"{query.title()} Artwork ({res_str})",
                        "filename": synth_fname,
                        "url": full_url,
                        "thumbnail": thumb_url,
                        "source_site": "Wallhaven HD",
                        "source_url": it.get("url") or full_url,
                        "width": w,
                        "height": h,
                        "engine": "Wallhaven",
                        "is_web": True,
                        "category": "web",
                        "character_tag": character_tag,
                        "character_icon": character_icon
                    })
        except Exception as e:
            print(f"[VisualSearch] Wallhaven search error: {e}")
        return results

    def _search_safebooru_images(
        self,
        query: str,
        limit: int = 12,
        character_tag: Optional[str] = None,
        character_icon: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Searches Safebooru for dedicated anime/manga character illustrations."""
        results = []
        try:
            clean_tag = query.lower().strip().replace(" ", "_")
            url = f"https://safebooru.org/index.php?page=dapi&s=post&q=index&json=1&limit={limit}&tags={urllib.parse.quote(clean_tag)}"
            headers = {"User-Agent": STANDARD_USER_AGENT}
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200 and r.text.strip().startswith("["):
                try:
                    items = r.json()
                except Exception:
                    items = []
                if isinstance(items, list):
                    for it in items:
                        img_name = it.get("image")
                        dir_name = it.get("directory")
                        if not img_name or not dir_name:
                            continue
                        full_url = f"https://safebooru.org/images/{dir_name}/{img_name}"
                        thumb_url = f"https://safebooru.org/thumbnails/{dir_name}/thumbnail_{img_name}"
                        synth_fname = f"sb_{it.get('id', hashlib.md5(full_url.encode()).hexdigest()[:8])}.jpg"
                        results.append({
                            "title": f"Safebooru Anime ({query})",
                            "filename": synth_fname,
                            "url": full_url,
                            "thumbnail": thumb_url,
                            "source_site": "Safebooru Anime",
                            "source_url": f"https://safebooru.org/index.php?page=post&s=view&id={it.get('id')}",
                            "width": it.get("width"),
                            "height": it.get("height"),
                            "engine": "Safebooru",
                            "is_web": True,
                            "category": "web",
                            "character_tag": character_tag,
                            "character_icon": character_icon
                        })
        except Exception as e:
            print(f"[VisualSearch] Safebooru search error: {e}")
        return results

    def _search_wikimedia_images(
        self,
        query: str,
        limit: int = 15,
        character_tag: Optional[str] = None,
        character_icon: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Searches Wikimedia Commons for high-resolution public domain / free license images."""
        results = []
        try:
            url = "https://commons.wikimedia.org/w/api.php"
            params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": "6",
                "gsrlimit": str(limit * 2),
                "prop": "imageinfo",
                "iiprop": "url|size",
                "format": "json"
            }
            headers = {"User-Agent": "ClarityAIStudio/2.0 (contact@clarityai.local)"}
            r = requests.get(url, params=params, headers=headers, timeout=6)
            if r.status_code != 200:
                return []

            pages = r.json().get("query", {}).get("pages", {})
            for _, page in pages.items():
                imageinfo = page.get("imageinfo", [])
                if not imageinfo:
                    continue
                info = imageinfo[0]
                img_url = info.get("url")
                if not img_url:
                    continue

                lower_u = img_url.lower()
                if lower_u.endswith(".svg") or lower_u.endswith(".ogg") or lower_u.endswith(".pdf"):
                    continue

                title = page.get("title", "").replace("File:", "")
                w = info.get("width")
                h = info.get("height")
                ext = os.path.splitext(urllib.parse.urlparse(img_url).path)[1].lower()
                synth_fname = f"wiki_{hashlib.md5(img_url.encode()).hexdigest()[:10]}{ext or '.jpg'}"

                results.append({
                    "title": title,
                    "filename": synth_fname,
                    "url": img_url,
                    "thumbnail": info.get("thumburl") or img_url,
                    "source_site": "Wikimedia Commons",
                    "source_url": info.get("descriptionurl") or img_url,
                    "width": w,
                    "height": h,
                    "engine": "Wikimedia",
                    "is_web": True,
                    "category": "web",
                    "character_tag": character_tag,
                    "character_icon": character_icon
                })
                if len(results) >= limit:
                    break
        except Exception as e:
            print(f"[VisualSearch] Wikimedia search error: {e}")
        return results

    def _search_flickr_images(
        self,
        query: str,
        limit: int = 15,
        character_tag: Optional[str] = None,
        character_icon: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Searches Flickr for creative, artistic, and photography images."""
        results = []
        try:
            clean_tag = re.sub(r'[^a-zA-Z0-9]+', ',', query.strip())
            url = f"https://www.flickr.com/services/feeds/photos_public.gne?tags={clean_tag}&format=json&nojsoncallback=1"
            headers = {"User-Agent": STANDARD_USER_AGENT}
            r = requests.get(url, headers=headers, timeout=6)
            if r.status_code != 200:
                return []

            items = r.json().get("items", [])
            for it in items[:limit]:
                m = it.get("media", {}).get("m")
                if not m:
                    continue
                large_url = m.replace("_m.", "_b.")
                synth_fname = f"flickr_{hashlib.md5(large_url.encode()).hexdigest()[:10]}.jpg"

                results.append({
                    "title": it.get("title") or query,
                    "filename": synth_fname,
                    "url": large_url,
                    "thumbnail": m,
                    "source_site": "Flickr",
                    "source_url": it.get("link") or large_url,
                    "engine": "Flickr",
                    "is_web": True,
                    "category": "web",
                    "character_tag": character_tag,
                    "character_icon": character_icon
                })
        except Exception as e:
            print(f"[VisualSearch] Flickr search error: {e}")
        return results

    def search_web_images(
        self,
        query: str,
        source: str = "all",
        limit: int = 24,
        rerank_with_clip: bool = True,
        query_image: Optional[Image.Image] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves web images across Google-like multi-sources (Bing, Wallhaven HD,
        Safebooru Anime, Wikimedia Commons, Flickr) with AI Franchise & Character
        Recognition, then re-ranks them with local CLIP ViT on CUDA.
        """
        q_clean = query.strip()
        if not q_clean and query_image is None:
            return []

        # 1. AI Franchise & Entity Knowledge Detection
        entity_info = None
        if query_image is None:
            entity_info = detect_franchise_entity(q_clean)

        # Automatic prompt translation for global search engines
        translated_q = q_clean
        try:
            from generator import smart_translate_prompt  # type: ignore
            t = smart_translate_prompt(q_clean)
            if t and t.lower() != q_clean.lower():
                translated_q = t
        except Exception:
            pass

        web_candidates = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []

            if entity_info:
                # Franchise universe recognized! (e.g. Dragon Ball, Naruto, One Piece, etc.)
                active_char_id = entity_info.get("active_character_id", "all")
                clean_franchise = entity_info.get("canonical_name") or entity_info["title"].split("(")[0].strip()

                def _get_safebooru_tag(ch_dict, clean_name):
                    safe_tag = ch_dict.get("safe_tag") or clean_name.lower().replace(" ", "_")
                    if "goku" in safe_tag:
                        return "son_goku"
                    if "gohan" in safe_tag:
                        return "son_gohan"
                    if "trunks" in safe_tag:
                        return "trunks"
                    if "shenron" in safe_tag:
                        return "shenron"
                    if "frieza" in safe_tag or "fide" in safe_tag:
                        return "freeza"
                    if "cell" in safe_tag:
                        return "cell"
                    if "buu" in safe_tag:
                        return "majin_buu"
                    if "piccolo" in safe_tag:
                        return "piccolo"
                    if "vegeta" in safe_tag:
                        return "vegeta"
                    return safe_tag

                if active_char_id == "all":
                    # Mode A: Search entire franchise + parallel sub-searches for all major characters
                    primary_query = f"{clean_franchise} anime characters wallpaper official art"

                    # 1. Primary franchise queries across engines
                    if source in ("all", "wallhaven"):
                        futures.append(executor.submit(self._search_wallhaven_images, clean_franchise, 16))
                    if source in ("all", "safebooru"):
                        futures.append(executor.submit(self._search_safebooru_images, clean_franchise.lower().replace(" ", "_"), 12))
                    if source in ("all", "bing"):
                        futures.append(executor.submit(self._search_bing_images, f"{clean_franchise} anime wallpaper hd", 12))
                    if source in ("all", "wikimedia"):
                        futures.append(executor.submit(self._search_wikimedia_images, clean_franchise, 6))

                    # 2. Concurrently dispatch key franchise characters
                    chars = [c for c in entity_info.get("characters", []) if c.get("id") != "all"]
                    for ch in chars[:7]:
                        clean_ch_name = ch["name"].split("(")[0].strip()
                        safe_ch_tag = _get_safebooru_tag(ch, clean_ch_name)

                        if source in ("all", "wallhaven"):
                            futures.append(executor.submit(
                                self._search_wallhaven_images,
                                f"{clean_franchise} {clean_ch_name}",
                                4,
                                ch["name"],
                                ch["icon"]
                            ))
                        if source in ("all", "safebooru"):
                            futures.append(executor.submit(
                                self._search_safebooru_images,
                                safe_ch_tag,
                                4,
                                ch["name"],
                                ch["icon"]
                            ))
                        if source in ("all", "bing"):
                            futures.append(executor.submit(
                                self._search_bing_images,
                                f"{clean_franchise} {clean_ch_name} anime wallpaper hd",
                                3,
                                ch["name"],
                                ch["icon"]
                            ))
                else:
                    # Mode B: Specific character selected within franchise (e.g. Vegeta)
                    target_ch = next((c for c in entity_info.get("characters", []) if c["id"] == active_char_id), None)
                    clean_ch_name = target_ch["name"].split("(")[0].strip() if target_ch else translated_q
                    ch_name = target_ch["name"] if target_ch else translated_q
                    ch_icon = target_ch["icon"] if target_ch else "⚡"
                    safe_ch_tag = _get_safebooru_tag(target_ch or {}, clean_ch_name)

                    if source in ("all", "wallhaven"):
                        futures.append(executor.submit(
                            self._search_wallhaven_images,
                            f"{clean_franchise} {clean_ch_name}",
                            12,
                            ch_name,
                            ch_icon
                        ))
                    if source in ("all", "safebooru"):
                        futures.append(executor.submit(
                            self._search_safebooru_images,
                            safe_ch_tag,
                            10,
                            ch_name,
                            ch_icon
                        ))
                    if source in ("all", "bing"):
                        futures.append(executor.submit(
                            self._search_bing_images,
                            f"{clean_franchise} {clean_ch_name} anime wallpaper hd",
                            limit,
                            ch_name,
                            ch_icon
                        ))
                    if source in ("all", "wikimedia"):
                        futures.append(executor.submit(
                            self._search_wikimedia_images,
                            f"{clean_franchise} {clean_ch_name}",
                            6,
                            ch_name,
                            ch_icon
                        ))
                    if source in ("all", "flickr"):
                        futures.append(executor.submit(
                            self._search_flickr_images,
                            f"{clean_franchise} {clean_ch_name}",
                            6,
                            ch_name,
                            ch_icon
                        ))
            else:
                # Standard keyword search
                if source in ("all", "bing"):
                    futures.append(executor.submit(self._search_bing_images, translated_q, limit))
                if source in ("all", "wallhaven"):
                    futures.append(executor.submit(self._search_wallhaven_images, translated_q, limit // 2 or 8))
                if source in ("all", "safebooru"):
                    futures.append(executor.submit(self._search_safebooru_images, translated_q, 8))
                if source in ("all", "wikimedia"):
                    futures.append(executor.submit(self._search_wikimedia_images, translated_q, limit // 2 or 6))
                if source in ("all", "flickr"):
                    futures.append(executor.submit(self._search_flickr_images, translated_q, limit // 2 or 6))

            for f in as_completed(futures):
                try:
                    web_candidates.extend(f.result())
                except Exception:
                    pass

        # Auto-tag any candidate whose title/url/filename mentions a known franchise character
        if entity_info:
            for c in web_candidates:
                if not c.get("character_tag"):
                    c_text = f"{c.get('title', '')} {c.get('filename', '')} {c.get('url', '')}".lower()
                    for ch in entity_info.get("characters", []):
                        if ch.get("id") == "all":
                            continue
                        for kw in ch.get("keywords", []):
                            if len(kw) >= 3 and kw.lower() in c_text:
                                c["character_tag"] = ch["name"]
                                c["character_icon"] = ch["icon"]
                                break
                        if c.get("character_tag"):
                            break

        # Deduplicate by image URL
        seen_urls = set()
        unique_candidates = []
        for c in web_candidates:
            u = c.get("url")
            if u and u not in seen_urls:
                seen_urls.add(u)
                unique_candidates.append(c)

        if not unique_candidates:
            return []

        # If CLIP Re-ranking is enabled, download thumbnails in parallel and compute cosine similarity
        if rerank_with_clip and len(unique_candidates) > 1:
            try:
                # Prepare query feature
                if query_image is not None:
                    query_feat = self._extract_image_features([query_image.convert("RGB")])[0]  # (512,)
                else:
                    # Domain-aware text feature
                    if entity_info:
                        clean_franchise = entity_info.get("canonical_name") or entity_info["title"].split("(")[0].strip()
                        expanded = [
                            f"official anime wallpaper of {clean_franchise} by Akira Toriyama characters",
                            f"Japanese anime series {clean_franchise} high quality official illustration artwork",
                            f"anime character battle scene in {clean_franchise} anime"
                        ]
                    else:
                        expanded = [translated_q]
                        if not translated_q.lower().startswith("a photo of"):
                            expanded.append(f"a high quality photo of {translated_q}")

                    t_feats = self._extract_text_features(expanded)
                    query_feat = np.mean(t_feats, axis=0)
                    query_feat = query_feat / np.linalg.norm(query_feat)

                # Fetch candidate thumbnails concurrently
                def fetch_thumb(item):
                    thumb_url = item.get("thumbnail") or item.get("url")
                    try:
                        resp = requests.get(thumb_url, headers={"User-Agent": STANDARD_USER_AGENT}, timeout=3.5)
                        if resp.status_code == 200:
                            im = Image.open(BytesIO(resp.content)).convert("RGB")
                            return item, im
                    except Exception:
                        pass
                    return item, None

                fetched_pairs = []
                with ThreadPoolExecutor(max_workers=10) as ex:
                    fetched_pairs = list(ex.map(fetch_thumb, unique_candidates[:limit + 15]))

                valid_items = []
                valid_images = []
                fallback_items = []

                for it, im in fetched_pairs:
                    if im is not None:
                        valid_items.append(it)
                        valid_images.append(im)
                    else:
                        fallback_items.append(it)

                if valid_images:
                    cand_feats = self._extract_image_features(valid_images)  # (K, 512)
                    sims = np.dot(cand_feats, query_feat)  # (K,)

                    for it, sim in zip(valid_items, sims):
                        raw_score = float(sim)
                        if query_image is not None:
                            pct_score = round(max(0.0, min(99.9, raw_score * 100.0)), 1)
                        else:
                            norm_score = max(0.0, min(1.0, (raw_score - 0.12) / 0.22))
                            pct_score = round(50.0 + norm_score * 49.0, 1)

                        it["score"] = raw_score
                        it["match_percent"] = pct_score
                        it["confidence_badge"] = (
                            "Rất cao" if pct_score >= 85 else ("Cao" if pct_score >= 70 else "Tương đối")
                        )

                    # Sort valid items descending by score
                    valid_items.sort(key=lambda x: -x.get("score", 0.0))

                # Add fallback items with decreasing default scores
                for idx, fb in enumerate(fallback_items):
                    fb["score"] = 0.22 - idx * 0.005
                    fb["match_percent"] = round(75.0 - idx * 1.5, 1)
                    fb["confidence_badge"] = "Tương đối"

                unique_candidates = valid_items + fallback_items
            except Exception as rerank_err:
                print(f"[VisualSearch] Re-ranking error: {rerank_err}")
                for idx, it in enumerate(unique_candidates):
                    it["score"] = 0.25 - idx * 0.005
                    it["match_percent"] = round(80.0 - idx * 1.5, 1)
                    it["confidence_badge"] = "Cao" if idx < 5 else "Tương đối"
        else:
            for idx, it in enumerate(unique_candidates):
                it["score"] = 0.25 - idx * 0.005
                it["match_percent"] = round(82.0 - idx * 1.5, 1)
                it["confidence_badge"] = "Cao" if idx < 4 else "Tương đối"

        return unique_candidates[:limit]

    def search_web_by_image(
        self,
        image_input: Union[str, Image.Image],
        source: str = "all",
        limit: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Reverse Image Search across the Web:
        1. Employs zero-shot CLIP concept probes on CUDA to detect image subject/style.
        2. Queries web search engines (Bing, Wikimedia, Flickr) for candidates.
        3. Re-ranks web candidates using visual cosine similarity against the input image.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            pil_img = Image.open(image_input).convert("RGB")
            # Also extract keywords from filename if possible
            base_hint = os.path.splitext(os.path.basename(image_input))[0]
            base_hint = re.sub(r'[^a-zA-Z0-9]+', ' ', base_hint).strip()
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            base_hint = ""
        else:
            raise ValueError("image_input must be a filepath string or PIL.Image")

        # Step 1: Detect visual concepts via zero-shot classification
        img_feat = self._extract_image_features([pil_img])[0]  # (512,)
        concept_feats = self._get_concept_features()  # (20, 512)

        concept_sims = np.dot(concept_feats, img_feat)
        top_concept_indices = np.argsort(-concept_sims)[:2]
        top_concepts = [VISUAL_CONCEPT_PROBES[i] for i in top_concept_indices]

        # Formulate web search query
        search_terms = []
        if base_hint and len(base_hint) > 3 and not base_hint.startswith("input_"):
            search_terms.append(base_hint)
        search_terms.extend(top_concepts)
        web_query = " ".join(search_terms)

        print(f"[VisualSearch] Reverse Web Search concept query: '{web_query}'")

        # Step 2: Fetch and CLIP re-rank web candidates
        return self.search_web_images(
            query=web_query,
            source=source,
            limit=limit,
            rerank_with_clip=True,
            query_image=pil_img
        )

    def download_web_image_to_local(
        self,
        image_url: str,
        custom_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Downloads a web image safely into `uploads/`, verifies with Pillow,
        registers it in the local index, and returns local paths and URLs.
        """
        if not image_url or not image_url.startswith("http"):
            raise ValueError("Invalid image URL")

        os.makedirs(UPLOAD_DIR, exist_ok=True)

        headers = {"User-Agent": STANDARD_USER_AGENT}
        r = requests.get(image_url, headers=headers, timeout=12, stream=True)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to download image (HTTP {r.status_code})")

        img_bytes = BytesIO(r.content)
        with Image.open(img_bytes) as im:
            w, h = im.size
            fmt = (im.format or "JPEG").lower()
            ext = ".png" if fmt == "png" else (".webp" if fmt == "webp" else ".jpg")

        url_hash = hashlib.md5(image_url.encode()).hexdigest()[:10]
        fname = custom_filename or f"web_import_{url_hash}{ext}"
        if not fname.endswith(tuple(ALLOWED_EXTENSIONS)):
            fname += ext

        local_path = os.path.join(UPLOAD_DIR, fname)
        with open(local_path, "wb") as f:
            f.write(r.content)

        # Trigger incremental re-index in background/fast
        self.build_or_update_index()

        return {
            "status": "success",
            "filename": fname,
            "local_url": f"/uploads/{fname}",
            "file_path": local_path,
            "width": w,
            "height": h,
            "size_kb": round(len(r.content) / 1024, 1)
        }

    # =========================================================================
    # UNIFIED PUBLIC SEARCH APIS (LOCAL + WEB + ALL)
    # =========================================================================

    def search_by_text(
        self,
        query: str,
        top_k: int = 16,
        min_similarity: float = 0.15,
        scope: str = "all",  # "all", "web", "local"
        web_source: str = "all",  # "all", "bing", "wikimedia", "flickr"
        category: str = "all"
    ) -> List[Dict[str, Any]]:
        """
        Unified Keyword Search:
        - scope="web": Searches global websites (Bing, Wikimedia, Flickr).
        - scope="local": Searches local computer library.
        - scope="all": Searches both and merges with similarity ranking.
        """
        q_clean = query.strip()
        if not q_clean:
            return []

        # 1. Web Only
        if scope == "web":
            return self.search_web_images(query=q_clean, source=web_source, limit=top_k)

        # 2. Local Search
        local_results = self._search_local_by_text(query=q_clean, top_k=top_k, min_similarity=min_similarity, category=category)

        if scope == "local":
            return local_results

        # 3. Scope "all": Parallel Web + Local
        web_limit = max(8, top_k // 2)
        web_results = []
        try:
            web_results = self.search_web_images(query=q_clean, source=web_source, limit=web_limit)
        except Exception as web_err:
            print(f"[VisualSearch] Error in concurrent web search: {web_err}")

        # Merge and sort by match_percent
        combined = local_results + web_results
        combined.sort(key=lambda x: -x.get("match_percent", 0.0))
        return combined[:top_k]

    def _search_local_by_text(
        self,
        query: str,
        top_k: int = 16,
        min_similarity: float = 0.15,
        category: str = "all"
    ) -> List[Dict[str, Any]]:
        """Internal local semantic text search."""
        if self.embeddings is None or len(self.file_paths) == 0:
            self.build_or_update_index()

        if self.embeddings is None or len(self.file_paths) == 0:
            return []

        queries = [query]
        try:
            from generator import smart_translate_prompt  # type: ignore
            translated = smart_translate_prompt(query)
            if translated and translated.lower() != query.lower():
                queries.append(translated)
        except Exception:
            pass

        entity_info = detect_franchise_entity(query)
        if entity_info:
            expanded_texts = [
                f"{entity_info['title']} {query} anime manga series characters official art",
                f"anime character illustration of {query}",
                f"{entity_info['title']} universe official anime scene"
            ]
            effective_min_sim = max(min_similarity, 0.22)
        else:
            expanded_texts = []
            for q in queries:
                expanded_texts.append(q)
                if not q.lower().startswith("a photo of") and not q.lower().startswith("an image of"):
                    expanded_texts.append(f"a photo of {q}")
                    expanded_texts.append(f"an artistic rendering of {q}")
            effective_min_sim = min_similarity

        text_feats = self._extract_text_features(expanded_texts)  # (M, 512)
        sim_matrix = np.dot(self.embeddings, text_feats.T)
        max_sims = np.max(sim_matrix, axis=1)  # (N,)

        sorted_indices = np.argsort(-max_sims)

        results = []
        for idx in sorted_indices:
            item = dict(self.metadata_list[idx])
            if category != "all" and item.get("category") != category:
                continue

            raw_score = float(max_sims[idx])
            if raw_score < effective_min_sim:
                continue

            norm_score = max(0.0, min(1.0, (raw_score - 0.12) / 0.22))
            pct_score = round(50.0 + norm_score * 49.0, 1)

            item["score"] = raw_score
            item["match_percent"] = pct_score
            item["confidence_badge"] = (
                "Rất cao" if pct_score >= 85 else ("Cao" if pct_score >= 70 else "Tương đối")
            )
            item["is_web"] = False
            results.append(item)

            if len(results) >= top_k:
                break

        return results

    def search_by_image(
        self,
        image_input: Union[str, Image.Image],
        top_k: int = 16,
        min_similarity: float = 0.20,
        exclude_self_path: Optional[str] = None,
        scope: str = "all",  # "all", "web", "local"
        web_source: str = "all"
    ) -> List[Dict[str, Any]]:
        """
        Unified Reverse Image Search:
        - scope="web": Finds visually similar images across internet sites.
        - scope="local": Finds similar images in local library.
        - scope="all": Combines both local and web visually similar results.
        """
        if scope == "web":
            return self.search_web_by_image(image_input=image_input, source=web_source, limit=top_k)

        local_results = self._search_local_by_image(
            image_input=image_input,
            top_k=top_k,
            min_similarity=min_similarity,
            exclude_self_path=exclude_self_path
        )

        if scope == "local":
            return local_results

        # Scope "all": Combine local and web
        web_limit = max(8, top_k // 2)
        web_results = []
        try:
            web_results = self.search_web_by_image(image_input=image_input, source=web_source, limit=web_limit)
        except Exception as web_err:
            print(f"[VisualSearch] Error in concurrent web image search: {web_err}")

        combined = local_results + web_results
        combined.sort(key=lambda x: -x.get("match_percent", 0.0))
        return combined[:top_k]

    def _search_local_by_image(
        self,
        image_input: Union[str, Image.Image],
        top_k: int = 16,
        min_similarity: float = 0.20,
        exclude_self_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Internal local reverse image search."""
        if self.embeddings is None or len(self.file_paths) == 0:
            self.build_or_update_index()

        if self.embeddings is None or len(self.file_paths) == 0:
            return []

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

        query_feat = self._extract_image_features([pil_img])[0]  # (512,)
        sims = np.dot(self.embeddings, query_feat)  # (N,)

        sorted_indices = np.argsort(-sims)

        results = []
        for idx in sorted_indices:
            path = self.file_paths[idx]
            if exclude_self_path and os.path.abspath(path) == os.path.abspath(exclude_self_path):
                continue
            if query_abs_path and os.path.abspath(path) == query_abs_path:
                continue

            raw_score = float(sims[idx])
            if raw_score > 0.999 and (exclude_self_path or query_abs_path):
                continue

            if raw_score < min_similarity:
                continue

            pct_score = round(max(0.0, min(99.9, raw_score * 100.0)), 1)
            item = dict(self.metadata_list[idx])
            item["score"] = raw_score
            item["match_percent"] = pct_score
            item["confidence_badge"] = (
                "Rất cao" if pct_score >= 85 else ("Cao" if pct_score >= 70 else "Tương đối")
            )
            item["is_web"] = False
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
            "cache_exists": os.path.exists(INDEX_CACHE_FILE),
            "web_search_available": True,
            "web_sources": ["Bing Web", "Wikimedia Commons", "Flickr"]
        }


# Global singleton instance
visual_search_engine = VisualSearchEngine()
