import pandas as pd

file_path = 'experiments/B1_evaluation/b1_test_predictions.csv'
df = pd.read_csv(file_path)

# Confidently wrong: confidence > 0.9 and correct == False
poisoned = df[(df['confidence'] > 0.9) & (df['correct'] == False)]

# Sort by confidence descending to find the most "arrogant" failures
poisoned = poisoned.sort_values(by='confidence', ascending=False)

print(f"Total images analyzed: {len(df)}")
print(f"Confidently wrong images found: {len(poisoned)}")
print("\n--- TOP 20 POISONED SAMPLES ---")
print(poisoned[['image_path', 'ground_truth_class', 'predicted_class', 'confidence']].head(20).to_string(index=False))

# Save to a CSV for later auditing
poisoned.to_csv('experiments/poisoned_samples.csv', index=False)
