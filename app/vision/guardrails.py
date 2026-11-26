import logging
import numpy as np
import cv2
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PhysicalTruth")

class PhysicalTruthGuardrail:
    def __init__(self):
        # Veto Rules: If these are met, the AI is simply WRONG.
        self.vetos = {
            'anti_necklace': {
                'min_circularity': 0.5,    # Anything this round is not a necklace
                'max_aspect_ratio': 0.6,    # Anything this compact is not a necklace
                'override_to': 'Earring'
            },
            'anti_earring': {
                'max_circularity': 0.3,    # Anything this elongated is not an earring
                'min_aspect_ratio': 0.8,    # Anything this long is not an earring
                'override_to': 'Necklace'
            }
        }

    def override(self, prediction: str, morph_data: dict):
        if not prediction:
            return prediction, "NO_PREDICTION"
            
        pred = prediction.lower()
        aspect = morph_data.get('aspect_ratio', 1.0)
        circ = morph_data.get('circularity', 1.0)
        
        logger.info(f"VETO CHECK: Pred={prediction}, Aspect={aspect:.3f}, Circ={circ:.3f}")

        # VETO 1: Is the AI claiming this is a Necklace, but it's actually round/compact?
        if pred == 'necklace':
            v = self.vetos['anti_necklace']
            if circ > v['min_circularity'] or aspect < v['max_aspect_ratio']:
                logger.info(f"VETO: Necklace predicted but object is compact/round. Overriding to {v['override_to']}")
                return v['override_to'], "VETO_PHYSICAL_TRUTH"

        # VETO 2: Is the AI claiming this is an Earring, but it's actually long/thin?
        if pred == 'earring':
            v = self.vetos['anti_earring']
            if circ < v['max_circularity'] or aspect > v['min_aspect_ratio']:
                logger.info(f"VETO: Earring predicted but object is long/thin. Overriding to {v['override_to']}")
                return v['override_to'], "VETO_PHYSICAL_TRUTH"

        return prediction, "AI_CONFIRMED"

def calculate_morphology(pil_image):
    try:
        open_cv_image = np.array(pil_image)
        if open_cv_image.ndim == 3:
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = open_cv_image
        
        # Stronger contrast enhancement for morphology
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"aspect_ratio": 1.0, "circularity": 1.0}
        
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 100:
            return {"aspect_ratio": 1.0, "circularity": 1.0}
        
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = min(w, h) / max(w, h)
        
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        
        return {"aspect_ratio": float(aspect_ratio), "circularity": float(circularity)}
    except Exception as e:
        return {"aspect_ratio": 1.0, "circularity": 1.0}
