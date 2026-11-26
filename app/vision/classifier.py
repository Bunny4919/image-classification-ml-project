import os
from typing import Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from PIL import Image

from app.vision.preprocessing import preprocess_for_classification, extract_morphological_features

# Category labels order (matches PyTorch ImageFolder alphabetical folder order)
CATEGORIES = ["Bracelet", "Earring", "Necklace", "Ring"]
NUM_CLASSES = len(CATEGORIES)


class JewelleryClassifier(nn.Module):
    """Hybrid ResNet50 + Morphological feature classifier for 4 core jewellery categories."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True, backbone_name: str = "resnet50"):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = resnet50(weights=weights)
        else:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)

        # Remove the original FC layer to get the features vector
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Late Fusion Head: [ResNet Features (2048) + Morph Features (4)]
        self.fusion_head = nn.Sequential(
            nn.Linear(in_features + 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor, morph: torch.Tensor) -> torch.Tensor:
        # Extract visual features
        visual_feats = self.backbone(x) # (batch, 2048)

        # Concatenate with morphological features
        combined = torch.cat([visual_feats, morph], dim=1) # (batch, 2052)

        return self.fusion_head(combined)


class ClassifierService:
    """Inference service for jewellery category classification."""

    def __init__(self, checkpoint_path: str = None, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.categories = CATEGORIES

        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if checkpoint_path is None:
            checkpoint_path = os.path.join(BASE_DIR, "indexes", "classifier_resnet50.pth")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Required ResNet50 classifier checkpoint not found at: {checkpoint_path}. "
                f"Please ensure classifier_resnet50.pth exists and has been trained."
            )

        self.checkpoint_path = checkpoint_path
        self.backbone_name = "resnet50"

        try:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            # Remap keys if necessary
            remapped_sd = {}
            for k, v in state_dict.items():
                if k == "backbone.fc.weight":
                    remapped_sd["backbone.fc.1.weight"] = v
                elif k == "backbone.fc.bias":
                    remapped_sd["backbone.fc.1.bias"] = v
                else:
                    remapped_sd[k] = v

            model = JewelleryClassifier(num_classes=len(self.categories), pretrained=False, backbone_name="resnet50")
            try:
                model.load_state_dict(remapped_sd)
            except Exception:
                model.load_state_dict(state_dict)

            self.model = model
            print(f"Successfully loaded ResNet50 classifier checkpoint from: {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ResNet50 classifier checkpoint from {checkpoint_path}: {e}"
            )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predicts category for a PIL Image.
        Returns:
            {
                "category": str,
                "confidence": float,
                "probabilities": Dict[str, float]
            }
        """
        tensor = preprocess_for_classification(image).to(self.device)
        morph_feats = extract_morphological_features(image)
        morph_tensor = torch.from_numpy(morph_feats).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor, morph_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_idx = int(probs.argmax())
        confidence = float(probs[top_idx])
        top_category = self.categories[top_idx]

        prob_dict = {cat: float(prob) for cat, prob in zip(self.categories, probs)}

        return {
            "category": top_category,
            "confidence": confidence,
            "probabilities": prob_dict
        }
