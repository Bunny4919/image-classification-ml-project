import os
import sys
import json
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.vision.preprocessing import fix_exif_orientation
from app.vision.embedding import DINOv2EmbeddingService
from app.retrieval.faiss_index import FAISSGalleryIndex

GALLERY_JSON = os.path.join(BASE_DIR, "data", "metadata", "reference_gallery.json")
INDEX_DIR = os.path.join(BASE_DIR, "indexes", "image_index")


def build_faiss_gallery_index(limit: int = None, batch_size: int = 32):
    if not os.path.exists(GALLERY_JSON):
        raise FileNotFoundError(f"Gallery JSON not found at {GALLERY_JSON}. Run data/prepare_dataset.py first.")

    with open(GALLERY_JSON, "r", encoding="utf-8") as f:
        gallery = json.load(f)

    if limit:
        gallery = gallery[:limit]

    print(f"Building FAISS visual index for {len(gallery)} gallery designs...", flush=True)

    embedder = DINOv2EmbeddingService(model_name="dinov2_vits14")
    print(f"DINOv2 ready. Embedding dimension: {embedder.dimension}", flush=True)

    faiss_gallery = FAISSGalleryIndex(dimension=embedder.dimension)

    # Flatten all views into a batchable task list
    item_tasks = []
    for item in gallery:
        design_id = item["design_id"]
        category = item["category"]
        views = item.get("views", [])
        for view in views:
            img_path = view["image_path"]
            if os.path.exists(img_path):
                meta = {
                    "design_id": design_id,
                    "category": category,
                    "view_id": view.get("view_id"),
                    "view_type": view.get("view_type"),
                    "image_path": img_path,
                    "filename": view.get("filename"),
                    "title": item.get("title"),
                    "material": item.get("material"),
                    "karat": item.get("karat"),
                    "description": item.get("description")
                }
                item_tasks.append((img_path, meta))

    print(f"Collected {len(item_tasks)} valid view images for embedding...", flush=True)

    total_indexed = 0
    skipped_count = 0

    for i in range(0, len(item_tasks), batch_size):
        batch = item_tasks[i:i + batch_size]
        pil_images = []
        valid_metas = []

        for img_path, meta in batch:
            try:
                img = Image.open(img_path)
                img = fix_exif_orientation(img).convert("RGB")
                pil_images.append(img)
                valid_metas.append(meta)
            except Exception as e:
                print(f"  Warning: Failed reading {img_path}: {e}")
                skipped_count += 1

        if pil_images:
            vectors = embedder.embed_batch(pil_images)
            for vec, meta in zip(vectors, valid_metas):
                faiss_gallery.add_item(vec, meta)
                total_indexed += 1

        if (i + batch_size) % (batch_size * 10) == 0 or (i + batch_size) >= len(item_tasks):
            print(f"  Progress: {min(i + batch_size, len(item_tasks))}/{len(item_tasks)} views embedded | {total_indexed} indexed", flush=True)

    faiss_gallery.save(INDEX_DIR)
    print(f"\nFAISS Gallery Index COMPLETE!", flush=True)
    print(f"  Vectors indexed : {total_indexed}", flush=True)
    print(f"  Skipped views   : {skipped_count}", flush=True)
    print(f"  Saved to        : {INDEX_DIR}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of designs to embed")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding extraction")
    args = parser.parse_args()
    build_faiss_gallery_index(limit=args.limit, batch_size=args.batch_size)
