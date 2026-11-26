import pandas as pd

# We use the failure analysis files because they contain the morphology data
# that the main predictions file lacks.
files = {
    'necklace_to_bracelet': 'experiments/failure_analysis/necklace_to_bracelet.csv',
    'necklace_to_earring': 'experiments/failure_analysis/necklace_to_earring.csv'
}

results = []

for label, path in files.items():
    df = pd.read_csv(path)
    # Focus on the "Wrong" ones (where the model predicted wrong)
    # In these files, predicted_class is the WRONG one.
    wrong = df.copy()
    
    # Calculate means for the "Wrong" samples
    stats = {
        'class': label,
        'avg_aspect': wrong['morph_aspect_ratio'].mean(),
        'avg_circ': wrong['morph_circularity'].mean(),
        'avg_solidity': wrong['morph_solidity'].mean(),
        'count': len(wrong)
    }
    results.append(stats)

print("--- FAILURE MORPHOLOGY (The 'Wrong' Patterns) ---")
print(pd.DataFrame(results).to_string(index=False))
