import pandas as pd
import shutil
import os

# Load the suspect list we generated previously
suspects_file = 'experiments/data_audit_suspects.csv'
df = pd.read_csv(suspects_file)

# Define the review directory
review_base_dir = 'experiments/visual_audit'
os.makedirs(review_base_dir, exist_ok=True)

count = 0
for idx, row in df.iterrows():
    img_path = row['image_path']
    gt_class = row['ground_truth_class']
    pred_class = row['predicted_class']
    conf = row['confidence']
    suspect_type = row['suspect_type']
    
    # Create subfolders by suspect type and ground truth for easier review
    target_dir = os.path.join(review_base_dir, suspect_type, gt_class)
    os.makedirs(target_dir, exist_ok=True)
    
    # Define a clear filename for the auditor: [Pred]_[Conf]_[OriginalName]
    filename = os.path.basename(img_path)
    new_filename = f"PRED_{pred_class}_CONF_{conf:.2f}_{filename}"
    target_path = os.path.join(target_dir, new_filename)
    
    try:
        if os.path.exists(img_path):
            shutil.copy2(img_path, target_path)
            count += 1
        else:
            print(f"Warning: File not found {img_path}")
    except Exception as e:
        print(f"Error copying {img_path}: {e}")

print(f"\nSuccessfully organized {count} images into {review_base_dir}")
print("You can now open this folder and verify if the labels are correct.")
