import os
from typing import List, Union
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

from app.vision.preprocessing import preprocess_for_embedding


class DINOv2EmbeddingService:
    """
    Extracts L2-normalized visual feature embeddings using DINOv2 ViT backbone.
    Default backbone: 'dinov2_vits14' (384-dimensional embeddings).
    """

    def __init__(self, model_name: str = "dinov2_vits14", device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = model_name

        print(f"Initializing DINOv2 model ({model_name}) on {self.device}...")
        try:
            # Load pretrained DINOv2 from PyTorch Hub
            self.model = torch.hub.load("facebookresearch/dinov2", model_name)
            self.dimension = 384 if "vits14" in model_name else 768
            print(f"Successfully loaded DINOv2 {model_name} (Dim: {self.dimension})")
        except Exception as e:
            print(f"Failed to load DINOv2 via torch.hub: {e}. Falling back to default PyTorch vision backbone.")
            # Fallback lightweight vision backbone if offline / network restriction
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            self.model.classifier = torch.nn.Identity()
            self.dimension = 576

        self.model.to(self.device)
        self.model.eval()

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Extracts L2-normalized 1D feature vector for a single PIL image.
        Returns:
            np.ndarray of shape (dim,) and dtype float32
        """
        tensor = preprocess_for_embedding(image).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
            # Normalize vector L2
            norm_features = F.normalize(features, p=2, dim=-1).squeeze(0).cpu().numpy()

        return norm_features.astype(np.float32)

    def embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Extracts L2-normalized 2D feature matrix for a list of PIL images.
        Returns:
            np.ndarray of shape (N, dim) and dtype float32
        """
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)

        tensors = torch.cat([preprocess_for_embedding(img) for img in images], dim=0).to(self.device)
        with torch.no_grad():
            features = self.model(tensors)
            norm_features = F.normalize(features, p=2, dim=-1).cpu().numpy()

        return norm_features.astype(np.float32)
