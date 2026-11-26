from typing import List, Dict, Any, Optional
import numpy as np

from app.retrieval.faiss_index import FAISSGalleryIndex


class VisualSearchService:
    """
    Search FAISS and aggregate multi-view results by Design ID.

    Important:
    - If a category is supplied, only catalog entries belonging to that
      category are considered.
    - Category filtering is done BEFORE design aggregation.
    - Category comparison is case-insensitive.
    """

    def __init__(self, faiss_index: FAISSGalleryIndex):
        self.faiss_index = faiss_index

    @staticmethod
    def _normalize_category(value: Optional[str]) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()

    def search_design(
        self,
        query_embedding: np.ndarray,
        top_k_designs: int = 5,
        raw_top_k: int = 200,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the visual index.

        Args:
            query_embedding:
                DINOv2 embedding of the query image.

            top_k_designs:
                Number of unique catalogue designs to return.

            raw_top_k:
                Number of FAISS image/view candidates examined before
                grouping by design.

            category:
                Optional predicted category. When provided, only catalogue
                images belonging to this category are considered.

        Returns:
            List of design-level results:
            [
                {
                    "design_id": "...",
                    "similarity": 0.82,
                    "best_view": {...},
                    "all_matched_views": [...]
                }
            ]
        """

        if top_k_designs <= 0:
            return []

        requested_category = self._normalize_category(category)

        # Search a reasonably large candidate pool so category filtering
        # does not accidentally discard all relevant candidates.
        raw_matches = self.faiss_index.search(
            query_embedding,
            top_k=max(raw_top_k, top_k_designs * 20)
        )

        if not raw_matches:
            return []

        design_groups: Dict[str, Dict[str, Any]] = {}

        for match in raw_matches:
            score = float(match.get("score", 0.0))
            meta = match.get("metadata") or {}

            metadata_category = self._normalize_category(
                meta.get("category")
            )

            # ---------------------------------------------------------
            # CATEGORY FILTER
            # ---------------------------------------------------------
            # If classifier says Ring, do not allow Necklace/Earring/etc.
            # catalogue entries into the candidate set.
            if requested_category:
                if metadata_category != requested_category:
                    continue

            design_id = meta.get("design_id", "UNKNOWN")

            if design_id == "UNKNOWN":
                continue

            if design_id not in design_groups:
                design_groups[design_id] = {
                    "design_id": design_id,
                    "similarity": score,
                    "best_view": meta,
                    "all_matched_views": [
                        {
                            "score": score,
                            "view": meta
                        }
                    ]
                }
            else:
                design_groups[design_id]["all_matched_views"].append(
                    {
                        "score": score,
                        "view": meta
                    }
                )

                if score > design_groups[design_id]["similarity"]:
                    design_groups[design_id]["similarity"] = score
                    design_groups[design_id]["best_view"] = meta

        aggregated_results = sorted(
            design_groups.values(),
            key=lambda x: float(x.get("similarity", 0.0)),
            reverse=True
        )

        return aggregated_results[:top_k_designs]
