def apply_geometric_guardrail(prediction, morph_data):
    """
    Overrides AI predictions based on physical jewelry characteristics.
    
    morph_data expects: {'aspect_ratio': float, 'circularity': float, 'solidity': float}
    """
    pred = prediction.lower()
    aspect = morph_data.get('aspect_ratio', 1.0)
    circ = morph_data.get('circularity', 1.0)
    solidity = morph_data.get('solidity', 1.0)
    
    # RULE 1: Necklace Guardrail
    # Necklaces are typically long (aspect ratio closer to 1 or > 0.7) 
    # and NOT circular (low circularity).
    if pred in ['earring', 'bracelet', 'ring']:
        if aspect > 0.7 and circ < 0.4:
            return 'Necklace', "OVERRIDE: Physical dimensions match Necklace (long/non-circular)."

    # RULE 2: Ring/Earring distinction
    # Rings are typically very circular and have high solidity.
    if pred == 'necklace' and circ > 0.8 and aspect < 0.5:
        return 'Ring', "OVERRIDE: Physical dimensions match Ring (circular/small)."

    return prediction, "AI_CONFIRMED"

# Test the logic
test_cases = [
    {'pred': 'Earring', 'morph': {'aspect_ratio': 0.95, 'circularity': 0.32}}, # Should be Necklace
    {'pred': 'Bracelet', 'morph': {'aspect_ratio': 0.82, 'circularity': 0.28}}, # Should be Necklace
    {'pred': 'Necklace', 'morph': {'aspect_ratio': 0.2, 'circularity': 0.9}}, # Should be Ring
    {'pred': 'Earring', 'morph': {'aspect_ratio': 0.3, 'circularity': 0.9}}, # Correct Earring
]

print("--- Guardrail Test Results ---")
for case in test_cases:
    res, reason = apply_geometric_guardrail(case['pred'], case['morph'])
    print(f"Input: {case['pred']} | Result: {res} | Reason: {reason}")
