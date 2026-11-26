import pandas as pd

# Load predictions
df = pd.read_csv('experiments/B1_evaluation/b1_test_predictions.csv')

# We need the morph data which is in the failure analysis files
# Let's read one of the failure analysis files to see the morph columns
fail_df = pd.read_csv('experiments/failure_analysis/necklace_to_bracelet.csv')

# Let's compare the 'morph_aspect_ratio' and 'morph_circularity' 
# for things the model thinks are Bracelets vs what they actually are.

print("--- Morphology of Confidently Wrong (Necklace predicted as Bracelet) ---")
wrong = fail_df[fail_df['predicted_class'] == 'Bracelet']
print(wrong[['morph_aspect_ratio', 'morph_circularity', 'morph_solidity']].describe())

print("\n--- Morphology of Correct Bracelets (from B1 test set) ---")
# We'll need to find the morph data for correct ones. 
# Since the B1 predictions file doesn't have morph, 
# I'll look for another file that has both.
