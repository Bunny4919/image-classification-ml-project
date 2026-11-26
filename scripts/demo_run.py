import requests
import json
import os

BASE_URL = "http://localhost:8000"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main():
    print("==================================================")
    print("   Gold Jewellery AI System - Live Execution Test")
    print("==================================================")

    # 1. Health Check
    print("\n[1] Checking System Health (/api/v1/health)...")
    r = requests.get(f"{BASE_URL}/api/v1/health")
    print(f"Status Code: {r.status_code}")
    print(f"Health Response: {json.dumps(r.json(), indent=2)}")

    # 2. Multimodal Identification & Vector Search
    sample_img_path = os.path.join(BASE_DIR, "data", "processed", "classification", "test", "Bracelet", "test_bracelet_025_jpg.rf.ccd694b5a550b4169f0a94556abd2383.jpg")
    if not os.path.exists(sample_img_path):
        import glob
        imgs = glob.glob("data/**/Bracelet/*.jpg", recursive=True) + glob.glob("dataset/**/*.jpg", recursive=True)
        if imgs:
            sample_img_path = imgs[0]

    print(f"\n[2] Submitting Image for Identification: {os.path.basename(sample_img_path)}...")
    with open(sample_img_path, "rb") as f:
        files = {"file": (os.path.basename(sample_img_path), f, "image/jpeg")}
        r = requests.post(f"{BASE_URL}/api/v1/identify", files=files)
    
    print(f"Status Code: {r.status_code}")
    res = r.json()
    print(f"Decision Status: {res.get('status')}")
    print(f"Predicted Category: {res.get('category')} (Confidence: {res.get('category_confidence'):.2f})")
    print(f"Best Match Design ID: {res.get('best_match_design_id')}")
    print(f"Vector Similarity Score: {res.get('similarity_score'):.4f}")
    
    if res.get('design_details'):
        details = res.get('design_details')
        print(f"\nMatched Design Details:")
        print(f"  Title: {details.get('title')}")
        print(f"  Material: {details.get('material')} ({details.get('karat')}K)")
        print(f"  Stone: {details.get('stone')}")
        print(f"  Collection: {details.get('collection')}")
        print(f"  Description: {details.get('description')}")
    
    print(f"\nAI Explanation:\n{res.get('explanation')}")

    target_design_id = res.get('best_match_design_id') or "GOLD_RING_100"

    # 3. Grounded RAG Query
    print(f"\n[3] Performing Grounded RAG Query for Design '{target_design_id}'...")
    payload = {
        "design_id": target_design_id,
        "question": "What is the material purity, stone type, and design description of this item?"
    }
    r = requests.post(f"{BASE_URL}/api/v1/query", json=payload)
    print(f"Status Code: {r.status_code}")
    rag_res = r.json()
    print(f"Design ID: {rag_res.get('design_id')}")
    print(f"Grounded Context Found: {rag_res.get('grounded_context_found')}")
    print(f"\n--- AI Answer ---")
    print(rag_res.get('answer'))

    print("\n==================================================")
    print("   Live Execution Completed Successfully!")
    print("==================================================")

if __name__ == "__main__":
    main()
