import json
import os
from PIL import Image
from app.vision.classifier import ClassifierService

def main():
    audit_path = "scratch_audit_results.json"
    if not os.path.exists(audit_path):
        print(f"Audit file not found: {audit_path}")
        return

    with open(audit_path, "r") as f:
        audit_data = json.load(f)

    service = ClassifierService()

    hard_cases = [item for item in audit_data.get("top_bracelets", []) if item.get("pred") == "Earring"]

    print(f"Verifying {len(hard_cases)} hard cases (Bracelets misclassified as Earrings)...")
    print("-" * 60)

    correct = 0
    for case in hard_cases:
        path = case["path"]
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        img = Image.open(path).convert("RGB")
        res = service.predict(img)

        pred = res["category"]
        conf = res["confidence"]

        if pred == "Bracelet":
            correct += 1
            status = "FIXED"
        else:
            status = f"STILL FAIL ({pred})"

        print(f"Path: {os.path.basename(path)}")
        print(f"  Result: {status} | Conf: {conf:.4f}")
        print("-" * 30)

    print(f"\nSummary: {correct}/{len(hard_cases)} hard cases fixed!")
    print(f"Accuracy on hard cases: {correct/len(hard_cases)*100:.2f}%" if hard_cases else "No hard cases found.")

if __name__ == "__main__":
    main()
