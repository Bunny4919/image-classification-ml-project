import os
from typing import Dict, Any, Optional
import yaml


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "config",
    "thresholds.yaml"
)


def load_threshold_config(path: str) -> Dict[str, Any]:
    """Loads the similarity and confidence thresholds from a YAML file."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


class RejectionDecisionService:
    """
    Final decision layer for jewellery identification.

    Important distinction:

        Unsupported object
            !=
        Low classifier confidence
            !=
        Unknown catalogue design

    A supported jewellery category with low confidence should not
    automatically be labelled UNSUPPORTED_OBJECT.
    """

    def __init__(self, config_path: str = CONFIG_PATH):

        # Defaults
        self.cat_threshold = 0.60
        self.sim_threshold_default = 0.75
        self.sim_thresholds = {}

        self.supported_categories = {
            "Ring",
            "Earring",
            "Necklace",
            "Bracelet",
        }

        # Optional YAML configuration
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}

                self.cat_threshold = float(
                    config.get(
                        "category_confidence_threshold",
                        config.get("cat_threshold", self.cat_threshold)
                    )
                )

                # Handle both legacy single threshold and new category-specific map
                sim_cfg = config.get("similarity_thresholds")
                if isinstance(sim_cfg, dict):
                    self.sim_thresholds = {k: float(v) for k, v in sim_cfg.items()}
                else:
                    # Fallback to global threshold
                    self.sim_threshold_default = float(
                        config.get(
                            "similarity_threshold",
                            config.get("sim_threshold", self.sim_threshold_default)
                        )
                    )

            except Exception as exc:
                print(
                    f"[RejectionDecisionService] "
                    f"Could not load thresholds.yaml: {exc}. "
                    f"Using defaults."
                )

    @staticmethod
    def _normalize_category(value: Optional[str]) -> str:
        if value is None:
            return ""

        value = str(value).strip().lower()

        mapping = {
            "ring": "Ring",
            "rings": "Ring",

            "earring": "Earring",
            "earrings": "Earring",

            "necklace": "Necklace",
            "necklaces": "Necklace",
            "chain": "Necklace",
            "chains": "Necklace",

            "bracelet": "Bracelet",
            "bracelets": "Bracelet",
        }

        return mapping.get(value, str(value).strip())

    def evaluate(
        self,
        category_result: Dict[str, Any],
        search_results: list
    ) -> Dict[str, Any]:

        raw_category = category_result.get("category", "Unknown")
        category = self._normalize_category(raw_category)

        cat_confidence = float(
            category_result.get("confidence", 0.0)
        )

        top_match = search_results[0] if search_results else None

        top_design_id = (
            top_match.get("design_id")
            if top_match
            else None
        )

        sim_score = (
            float(top_match.get("similarity", 0.0))
            if top_match
            else 0.0
        )

        # -------------------------------------------------------------
        # 1. Truly unsupported category
        # -------------------------------------------------------------
        if category not in self.supported_categories:

            return {
                "status": "UNSUPPORTED_OBJECT",
                "category": category,
                "category_confidence": cat_confidence,
                "best_match_design_id": None,
                "similarity_score": sim_score,
                "best_match": None,
                "explanation": (
                    "The uploaded image could not be classified as "
                    "Ring, Earring, Necklace, or Bracelet."
                )
            }

        # -------------------------------------------------------------
        # 2. Category is supported but confidence is low
        #
        # DO NOT call this UNSUPPORTED_OBJECT.
        # -------------------------------------------------------------
        if cat_confidence < self.cat_threshold:

            return {
                "status": "LOW_CONFIDENCE",
                "category": category,
                "category_confidence": cat_confidence,
                "best_match_design_id": None,
                "similarity_score": 0.0,
                "best_match": None,
                "explanation": (
                    f"The image appears to contain a supported "
                    f"{category.lower()}, but the classifier confidence "
                    f"is too low for a reliable identification."
                )
            }

        # -------------------------------------------------------------
        # 3. Defensive category validation
        #
        # Even though search.py already filters by category, verify
        # again before accepting a catalogue design.
        # -------------------------------------------------------------
        if top_match:

            match_category = self._normalize_category(
                (top_match.get("best_view") or {}).get("category")
            )

            if match_category and match_category != category:

                return {
                    "status": "KNOWN_CATEGORY_UNKNOWN_DESIGN",
                    "category": category,
                    "category_confidence": cat_confidence,
                    "best_match_design_id": None,
                    "similarity_score": 0.0,
                    "best_match": None,
                    "explanation": (
                        f"The classifier identified the image as "
                        f"{category}, but the retrieved catalogue "
                        f"candidate belongs to a different category."
                    )
                }

        # -------------------------------------------------------------
        # 4. Strong same-category catalogue match
        # -------------------------------------------------------------
        # Get the category-specific threshold or fallback to default
        current_sim_threshold = self.sim_thresholds.get(category, self.sim_threshold_default)

        if (
            sim_score >= current_sim_threshold
            and top_design_id
        ):

            return {
                "status": "KNOWN_DESIGN",
                "category": category,
                "category_confidence": cat_confidence,
                "best_match_design_id": top_design_id,
                "similarity_score": sim_score,
                "best_match": top_match,
                "explanation": (
                    f"Successfully identified a known "
                    f"{category} design '{top_design_id}'."
                )
            }

        # -------------------------------------------------------------
        # 5. Correct category, but no sufficiently similar catalogue design.
        # -------------------------------------------------------------
        return {
            "status": "KNOWN_CATEGORY_UNKNOWN_DESIGN",
            "category": category,
            "category_confidence": cat_confidence,
            "best_match_design_id": None,
            "similarity_score": 0.0,
            "best_match": None,
            "explanation": (
                f"Recognized as a supported {category}, but no "
                f"same-category catalogue match exceeded the "
                f"similarity threshold."
            )
        }
