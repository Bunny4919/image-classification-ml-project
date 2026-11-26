import os
from typing import List, Tuple

from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    status,
)

from PIL import Image

from app.vision.preprocessing import (
    load_and_verify_image,
    ALLOWED_MIME_TYPES,
    MAX_FILE_SIZE_BYTES,
)

from app.vision.classifier import ClassifierService
from app.vision.embedding import DINOv2EmbeddingService

from app.retrieval.faiss_index import FAISSGalleryIndex
from app.retrieval.search import VisualSearchService

from app.vision.similarity import RejectionDecisionService
from app.vision.guardrails import PhysicalTruthGuardrail, calculate_morphology

from app.rag.retriever import RAGMetadataRetriever
from app.rag.generator import RAGGeneratorService

from app.schemas.responses import (
    IdentifyResponse,
    DesignMatch,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["Identification"]
)


# ================================================================
# PATHS
# ================================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Strictly use the trained ResNet50 checkpoint
CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "indexes",
    "classifier_resnet50.pth"
)

INDEX_DIR = os.path.join(
    BASE_DIR,
    "indexes",
    "image_index"
)

CHROMA_DIR = os.path.join(
    BASE_DIR,
    "indexes",
    "rag_index"
)


# ================================================================
# LAZY SINGLETON SERVICES
# ================================================================

classifier_service = None
embedding_service = None
faiss_index = None
search_service = None
rejection_service = None
rag_retriever = None
rag_generator = None


def get_services():
    global classifier_service
    global embedding_service
    global faiss_index
    global search_service
    global rejection_service
    global rag_retriever
    global rag_generator

    # ------------------------------------------------------------
    # Classifier — ResNet50
    # ------------------------------------------------------------
    if classifier_service is None:
        classifier_service = ClassifierService(
            checkpoint_path=CHECKPOINT_PATH
        )

    # ------------------------------------------------------------
    # DINOv2 visual embedding
    # ------------------------------------------------------------
    if embedding_service is None:
        embedding_service = DINOv2EmbeddingService(
            model_name="dinov2_vits14"
        )

    # ------------------------------------------------------------
    # FAISS gallery index
    # ------------------------------------------------------------
    if faiss_index is None:
        faiss_file = os.path.join(
            INDEX_DIR,
            "faiss_gallery.index"
        )

        if os.path.exists(faiss_file):
            faiss_index = FAISSGalleryIndex.load(INDEX_DIR)
        else:
            faiss_index = FAISSGalleryIndex(
                dimension=embedding_service.dimension
            )

        search_service = VisualSearchService(faiss_index)

    # ------------------------------------------------------------
    # Rejection / decision service
    # ------------------------------------------------------------
    if rejection_service is None:
        rejection_service = RejectionDecisionService()

    # ------------------------------------------------------------
    # RAG retriever + generator
    # ------------------------------------------------------------
    if rag_retriever is None:
        rag_retriever = RAGMetadataRetriever(chroma_dir=CHROMA_DIR)

    if rag_generator is None:
        rag_generator = RAGGeneratorService()

    return (
        classifier_service,
        embedding_service,
        search_service,
        rejection_service,
        rag_retriever,
        rag_generator,
    )


# ================================================================
# IMAGE CROPPING HELPERS
# ================================================================

def _safe_crop(
    image: Image.Image,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> Image.Image:
    """Crop with boundary clamping so no crop ever exceeds image dimensions."""
    width, height = image.size
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    return image.crop((left, top, right, bottom))


def crop_center_object(image: Image.Image) -> Image.Image:
    """
    Crop the middle 80% of the image to remove finger/hand/background edges.

    Most jewellery photographs (worn or held) have the item centred with
    surrounding skin or background that confuses the classifier.
    Removing the outer 10% on each side helps the model see more chain/ring
    and less hand, reducing chain→bracelet and ring→bracelet errors.
    """
    w, h = image.size
    left   = int(w * 0.10)
    top    = int(h * 0.10)
    right  = int(w * 0.90)
    bottom = int(h * 0.90)
    return image.crop((left, top, right, bottom))


def generate_classification_crops(
    image: Image.Image,
) -> List[Tuple[str, Image.Image]]:
    """
    Generate several spatial views of a user photograph.

    Lightweight — requires no external detector.

    Goal: give the classifier a chance to see jewellery when the
    item occupies only a fraction of the original photo.

    The confidence-weighted aggregation decides which crops carry
    useful signal; not every crop is guaranteed to contain jewellery.
    """
    image = image.convert("RGB")
    width, height = image.size
    crops: List[Tuple[str, Image.Image]] = []

    # Full image
    crops.append(("full", image))

    # Central 75 % crop
    cx, cy = width // 2, height // 2
    cw, ch = int(width * 0.75), int(height * 0.75)
    crops.append((
        "center",
        _safe_crop(image, cx - cw // 2, cy - ch // 2, cx + cw // 2, cy + ch // 2)
    ))

    # Lower-center — useful when jewellery is held or worn in the lower half
    crops.append((
        "lower_center",
        _safe_crop(image, int(width * 0.10), int(height * 0.25), int(width * 0.90), height)
    ))

    # Four overlapping 70 % corner regions
    crop_w = int(width * 0.70)
    crop_h = int(height * 0.70)

    for name, x_ratio, y_ratio in [
        ("top_left",     0.00, 0.00),
        ("top_right",    0.30, 0.00),
        ("bottom_left",  0.00, 0.30),
        ("bottom_right", 0.30, 0.30),
    ]:
        left = int(width * x_ratio)
        top = int(height * y_ratio)
        crops.append((
            name,
            _safe_crop(image, left, top, left + crop_w, top + crop_h)
        ))

    return crops


def classify_with_multiple_views(
    classifier: ClassifierService,
    image: Image.Image,
) -> dict:
    """
    Classify the image using multiple spatial views and aggregate
    the probability distributions with per-crop weights.

    Returns a dict compatible with the old single-view cat_result
    plus two extra keys:
        best_crop_name  — name of the highest-evidence crop
        best_crop_image — PIL image of that crop (used for DINOv2)
        crop_predictions — raw per-crop results (for debugging)
    """
    crops = generate_classification_crops(image)
    predictions = []

    for name, crop in crops:
        try:
            result = classifier.predict(crop)
            predictions.append({"name": name, "result": result})
        except Exception as exc:
            print(f"[MultiViewClassifier] Failed crop '{name}': {exc}")

    # Fall back to plain single-view if all crops fail
    if not predictions:
        return classifier.predict(image)

    categories = classifier.categories

    # Per-crop weights — full image down-weighted (lots of background)
    weights = {
        "full":         0.50,
        "center":       1.00,
        "lower_center": 1.15,
        "top_left":     0.80,
        "top_right":    0.80,
        "bottom_left":  0.90,
        "bottom_right": 0.90,
    }

    aggregated = {cat: 0.0 for cat in categories}
    total_weight = 0.0

    for item in predictions:
        weight = weights.get(item["name"], 0.75)
        probs = item["result"].get("probabilities", {})
        for cat in categories:
            aggregated[cat] += float(probs.get(cat, 0.0)) * weight
        total_weight += weight

    if total_weight > 0:
        for cat in aggregated:
            aggregated[cat] /= total_weight

    top_category = max(aggregated, key=aggregated.get)
    confidence = float(aggregated[top_category])

    # Find the crop that gave the strongest single evidence for
    # the winning category — used as the DINOv2 embedding source
    best_crop_name = "full"
    best_crop_confidence = -1.0
    best_crop_image = image

    for item in predictions:
        crop_conf = float(
            item["result"].get("probabilities", {}).get(top_category, 0.0)
        )
        if crop_conf > best_crop_confidence:
            best_crop_confidence = crop_conf
            best_crop_name = item["name"]
            for name, crop in crops:
                if name == best_crop_name:
                    best_crop_image = crop
                    break

    return {
        "category": top_category,
        "confidence": confidence,
        "probabilities": aggregated,
        "best_crop_name": best_crop_name,
        "best_crop_image": best_crop_image,
        "crop_predictions": predictions,
    }


# ================================================================
# IDENTIFICATION ENDPOINT
# ================================================================

@router.post(
    "/identify",
    response_model=IdentifyResponse,
    summary="Identify Gold Jewellery Image",
)
async def identify_jewellery(file: UploadFile = File(...)):
    """
    Jewellery identification pipeline:

    1.  Validate uploaded file (MIME type, size, PIL integrity).
    2.  Classify using multiple spatial crops — weighted probability aggregation.
    3.  Select the highest-evidence crop for DINOv2 embedding.
    4.  Search FAISS **only within the predicted category**.
    5.  Apply confidence + similarity decision rules (LOW_CONFIDENCE
        is distinct from UNSUPPORTED_OBJECT).
    6.  Retrieve RAG metadata with a final defensive category check.
    7.  Return grounded explanation.
    """

    # ============================================================
    # 1. FILE VALIDATION
    # ============================================================
    if (
        file.content_type
        and file.content_type.lower() not in ALLOWED_MIME_TYPES
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid file type '{file.content_type}'. "
                f"Allowed MIME types: {', '.join(ALLOWED_MIME_TYPES)}"
            ),
        )

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Uploaded file exceeds max limit of "
                f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
            ),
        )

    # ============================================================
    # 2. SAFE IMAGE DECODING
    # ============================================================
    try:
        image = load_and_verify_image(file_bytes, filename=file.filename or "")
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode image.",
        )

    # ============================================================
    # 3. LOAD SERVICES
    # ============================================================
    (
        cls_svc,
        emb_svc,
        search_svc,
        rej_svc,
        rag_ret,
        rag_gen,
    ) = get_services()

    # Initialize Guardrail
    guardrail = PhysicalTruthGuardrail()


    # ============================================================
    # 4. MULTI-VIEW CLASSIFICATION
    # ============================================================
    # Pre-crop the centre 80% to reduce fingers/skin/background
    # before running multi-view classification. This is the main
    # fix for chain-on-hand being classified as Bracelet.
    cropped_image = crop_center_object(image)
    cat_result = classify_with_multiple_views(cls_svc, cropped_image)

    # ============================================================
    # 4b. PHYSICAL TRUTH OVERRIDE
    # ============================================================
    # Use geometric guardrails to catch 'Confidently Wrong' predictions
    # (e.g. AI says 'Earring' but object is physically a 'Necklace')
    morph_data = calculate_morphology(cropped_image)
    category, status_msg = guardrail.override(cat_result["category"], morph_data)

    # Update the result dictionary so subsequent logic uses the corrected category
    cat_result["category"] = category

    confidence = float(cat_result["confidence"])


    print(
        f"[Classification] category={category}, "
        f"confidence={confidence:.4f}, "
        f"best_crop={cat_result.get('best_crop_name')}"
    )

    # ============================================================
    # 5. SELECT IMAGE FOR VISUAL EMBEDDING
    # ============================================================
    # Use the crop with the strongest classification signal.
    # This helps when jewellery is small in the original photo
    # (e.g. a ring worn on a hand in a wide-angle shot).
    embedding_image = cat_result.get("best_crop_image", image)

    # ============================================================
    # 6. DINOv2 EMBEDDING
    # ============================================================
    query_vec = emb_svc.embed_image(embedding_image)

    # ============================================================
    # 7. CATEGORY-FILTERED FAISS SEARCH
    # ============================================================
    # Only catalogue entries that belong to the same category
    # as the classifier prediction are considered.
    # A Ring will NEVER be matched to a Necklace design.
    search_results = search_svc.search_design(
        query_embedding=query_vec,
        category=category,
        top_k_designs=5,
        raw_top_k=200,
    )

    print(
        f"[VisualSearch] category={category}, "
        f"results={len(search_results)}"
    )

    # ============================================================
    # 8. DECISION
    # ============================================================
    # No unrestricted FAISS consensus override — FAISS is already
    # restricted to the classifier's own category.
    eval_result = rej_svc.evaluate(cat_result, search_results)

    eval_status = eval_result["status"]
    top_design_id = eval_result.get("best_match_design_id")

    # ============================================================
    # 9. RAG METADATA
    # ============================================================
    design_meta = None
    design_details = None

    if eval_status == "KNOWN_DESIGN" and top_design_id:
        design_meta = rag_ret.get_by_design_id(top_design_id)

        if design_meta and "metadata" in design_meta:
            metadata = design_meta["metadata"]
            top_match = eval_result.get("best_match", {})
            best_view = top_match.get("best_view", {})

            # ----------------------------------------------------
            # Defensive category validation against RAG metadata
            # ----------------------------------------------------
            rag_category = str(metadata.get("category", "")).strip().lower()
            classifier_category = str(category).strip().lower()

            if rag_category and rag_category != classifier_category:
                print(
                    f"[Safety] Prevented category mismatch: "
                    f"classifier={category}, RAG={metadata.get('category')}"
                )
                eval_status = "KNOWN_CATEGORY_UNKNOWN_DESIGN"
                top_design_id = None
                design_meta = None
                design_details = None

            else:
                try:
                    karat_value = int(metadata.get("karat", 18))
                except (TypeError, ValueError):
                    karat_value = 18

                design_details = DesignMatch(
                    design_id=top_design_id,
                    title=metadata.get("title"),
                    category=metadata.get("category"),
                    similarity_score=float(
                        eval_result.get("similarity_score", 0.0)
                    ),
                    material=metadata.get("material"),
                    karat=karat_value,
                    stone=metadata.get("stone"),
                    collection=metadata.get("collection"),
                    description=metadata.get("description"),
                    matched_view_type=best_view.get("view_type"),
                )

    # ============================================================
    # 10. LLM SUMMARY
    # ============================================================
    explanation = rag_gen.generate_identification_summary(
        eval_status, category, design_meta
    )

    # ============================================================
    # 11. RESPONSE
    # ============================================================
    return IdentifyResponse(
        status=eval_status,
        category=category,
        category_confidence=confidence,
        category_probabilities=cat_result["probabilities"],
        best_match_design_id=top_design_id,
        similarity_score=float(eval_result.get("similarity_score", 0.0)),
        design_details=design_details,
        explanation=explanation,
    )
