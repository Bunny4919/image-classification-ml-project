import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PhysicalTruth")

class PhysicalTruthGuardrail:
    def __init__(self):
        # These 'magic numbers' were derived from the failure analysis of confidently wrong samples
        self.thresholds = {
            'necklace': {
                'min_aspect_ratio': 0.7,
                'max_circularity': 0.4,
                'target_classes': ['earring', 'bracelet', 'ring']
            },
            'ring': {
                'min_circularity': 0.8,
                'max_aspect_ratio': 0.5,
                'target_classes': ['necklace', 'bracelet']
            }
        }

    def override(self, prediction, morph_data):
        """
        Evaluates if the AI prediction contradicts physical geometric truth.
        
        :param prediction: The class predicted by the AI model (string)
        :param morph_data: Dictionary containing {'aspect_ratio': float, 'circularity': float}
        :return: (final_prediction, status_message)
        """
        pred = prediction.lower()
        aspect = morph_data.get('aspect_ratio', 1.0)
        circ = morph_data.get('circularity', 1.0)
        
        # --- RULE 1: The Necklace Guardrail ---
        # If AI says it's a small item (earring/ring/bracelet) but it's actually long and non-circular
        necklace_cfg = self.thresholds['necklace']
        if pred in necklace_cfg['target_classes']:
            if aspect > necklace_cfg['min_aspect_ratio'] and circ < necklace_cfg['max_circularity']:
                logger.info(f"Guardrail Triggered: {prediction} -> Necklace (Aspect: {aspect}, Circ: {circ})")
                return "Necklace", "OVERRIDE_GEOMETRIC_TRUTH"

        # --- RULE 2: The Ring/Small Item Guardrail ---
        # If AI says it's a necklace but it's actually very circular and compact
        ring_cfg = self.thresholds['ring']
        if pred in ring_cfg['target_classes']:
            if circ > ring_cfg['min_circularity'] and aspect < ring_cfg['max_aspect_ratio']:
                logger.info(f"Guardrail Triggered: {prediction} -> Ring (Aspect: {aspect}, Circ: {circ})")
                return "Ring", "OVERRIDE_GEOMETRIC_TRUTH"

        return prediction, "AI_CONFIRMED"

if __name__ == "__main__":
    # Validation Suite
    guard = PhysicalTruthGuardrail()
    
    test_cases = [
        {"pred": "Earring", "morph": {"aspect_ratio": 0.85, "circularity": 0.3}, "expected": "Necklace"},
        {"pred": "Bracelet", "morph": {"aspect_ratio": 0.9, "circularity": 0.25}, "expected": "Necklace"},
        {"pred": "Necklace", "morph": {"aspect_ratio": 0.3, "circularity": 0.9}, "expected": "Ring"},
        {"pred": "Earring", "morph": {"aspect_ratio": 0.3, "circularity": 0.85}, "expected": "Earring"},
    ]
    
    print("\n--- Guardrail Validation ---")
    for i, case in enumerate(test_cases):
        res, status = guard.override(case['pred'], case['morph'])
        passed = res == case['expected']
        print(f"Case {i}: {case['pred']} (Morph: {case['morph']}) -> {res} [{status}] | {'PASS' if passed else 'FAIL'}")
