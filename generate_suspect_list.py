import pandas as pd

file_path = 'experiments/B1_evaluation/b1_test_predictions.csv'
df = pd.read_csv(file_path)

# Category 1: Confidently Wrong (High confidence, but incorrect)
# These are the "Poison" samples that mislead the model.
confidently_wrong = df[(df['confidence'] > 0.7) & (df['correct'] == False)]

# Category 2: Unstable Corrects (Correct, but low confidence or high margin for the wrong class)
# These are "Borderline" samples that indicate the model is confused about the boundary.
unstable_correct = df[(df['correct'] == True) & (df['confidence'] < 0.6)]

# Combine and tag
confidently_wrong['suspect_type'] = 'CONFIDENTLY_WRONG'
unstable_correct['suspect_type'] = 'UNSTABLE_CORRECT'

suspects = pd.concat([confidently_wrong, unstable_correct])
suspects = suspects.sort_values(by='confidence', ascending=False)

# Select relevant columns for a human auditor
audit_cols = ['image_path', 'ground_truth_class', 'predicted_class', 'confidence', 'suspect_type']
result = suspects[audit_cols]

print(f"Found {len(confidently_wrong)} Confidently Wrong samples.")
print(f"Found {len(unstable_correct)} Unstable Correct samples.")
print("\n--- TOP SUSPECTS FOR MANUAL AUDIT ---")
print(result.head(50).to_string(index=False))

result.to_csv('experiments/data_audit_suspects.csv', index=False)
