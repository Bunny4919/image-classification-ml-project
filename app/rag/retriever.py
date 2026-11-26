import os
from typing import Dict, Any, Optional, List
import chromadb

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "indexes", "rag_index")


class RAGMetadataRetriever:
    """Retriever service for accessing structured jewellery metadata stored in ChromaDB."""

    def __init__(self, chroma_dir: str = CHROMA_DB_DIR, collection_name: str = "jewellery_metadata"):
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.client = None
        self.collection = None

        if os.path.exists(chroma_dir):
            try:
                self.client = chromadb.PersistentClient(path=chroma_dir)
                self.collection = self.client.get_or_create_collection(name=collection_name)
            except Exception as e:
                print(f"Warning: Could not connect to ChromaDB at {chroma_dir}: {e}")

    def get_by_design_id(self, design_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves exact metadata record for a specific design_id."""
        if not self.collection:
            return None

        try:
            results = self.collection.get(ids=[design_id])
            if results and results.get("metadatas") and len(results["metadatas"]) > 0:
                meta = results["metadatas"][0]
                doc = results["documents"][0] if results.get("documents") else ""
                return {
                    "design_id": design_id,
                    "metadata": meta,
                    "document": doc
                }
        except Exception as e:
            print(f"Error fetching metadata for design_id '{design_id}': {e}")

        return None

    def search_query(self, query: str, design_id: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """Performs semantic search over metadata, optionally filtered by design_id."""
        if not self.collection:
            return []

        where_clause = {"design_id": design_id} if design_id else None

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause
            )

            retrieved = []
            if results and results.get("documents") and len(results["documents"]) > 0:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else []

                for d, m in zip(docs, metas):
                    retrieved.append({"metadata": m, "document": d})

            return retrieved
        except Exception as e:
            print(f"Error querying RAG metadata: {e}")
            return []
