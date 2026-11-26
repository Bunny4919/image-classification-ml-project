import os
import sys
import json
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

GALLERY_JSON = os.path.join(BASE_DIR, "data", "metadata", "reference_gallery.json")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "indexes", "rag_index")


def ingest_metadata_to_chromadb(collection_name: str = "jewellery_metadata"):
    """Ingests reference gallery metadata into ChromaDB vector store."""
    if not os.path.exists(GALLERY_JSON):
        raise FileNotFoundError(f"Gallery metadata JSON not found at {GALLERY_JSON}. Run data/prepare_dataset.py first.")

    with open(GALLERY_JSON, "r", encoding="utf-8") as f:
        gallery = json.load(f)

    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    documents = []
    metadatas = []
    ids = []

    for item in gallery:
        design_id = item["design_id"]
        title = item.get("title", f"Design {design_id}")
        category = item.get("category", "Jewellery")
        material = item.get("material", "Gold")
        karat = item.get("karat", 18)
        stone = item.get("stone", "None")
        style = item.get("style", "Luxury")
        collection_name_val = item.get("collection", "Classic")
        description = item.get("description", "")

        doc_text = (
            f"Design ID: {design_id}\n"
            f"Title: {title}\n"
            f"Category: {category}\n"
            f"Material: {material} ({karat}K)\n"
            f"Gemstone / Accent: {stone}\n"
            f"Design Style: {style}\n"
            f"Collection: {collection_name_val}\n"
            f"Description: {description}"
        )

        documents.append(doc_text)
        ids.append(design_id)
        metadatas.append({
            "design_id": design_id,
            "title": title,
            "category": category,
            "material": material,
            "karat": int(karat),
            "stone": stone,
            "style": style,
            "collection": collection_name_val,
            "description": description
        })

    # Add to ChromaDB in batch
    if documents:
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    print(f"Successfully ingested {len(documents)} design metadata documents into ChromaDB at {CHROMA_DB_DIR}")


if __name__ == "__main__":
    ingest_metadata_to_chromadb()
