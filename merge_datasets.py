import pandas as pd
import os

# List of TSV files to merge
tsv_files = [
    'clinical_dataframe.tsv',
    'miRNA_dataframe.tsv',
    'proteomics_dataframe.tsv',
    'phosphoproteomics_dataframe.tsv',
    'transcriptomics_dataframe.tsv',
    'somatic_mutation_dataframe.tsv'
]

# Function to get patient IDs from each file
def get_patient_ids(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')
        # Assuming the first column contains patient IDs
        return set(df.iloc[:, 0])
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return set()

# Get patient IDs from each file
print("Reading patient IDs from each file...")
patient_sets = []
for file in tsv_files:
    if os.path.exists(file):
        patients = get_patient_ids(file)
        print(f"{file}: {len(patients)} patients")
        patient_sets.append(patients)
    else:
        print(f"Warning: {file} not found")

# Find common patients across all files
if patient_sets:
    common_patients = set.intersection(*patient_sets)
    print(f"\nNumber of patients common to all datasets: {len(common_patients)}")

    # Create merged dataset
    print("\nCreating merged dataset...")
    merged_dfs = []
    
    for file in tsv_files:
        if os.path.exists(file):
            df = pd.read_csv(file, sep='\t')
            # Filter for common patients
            df = df[df.iloc[:, 0].isin(common_patients)]
            merged_dfs.append(df)
            print(f"Added {file} to merged dataset")

    # Save list of common patients
    with open('common_patients.txt', 'w') as f:
        f.write('\n'.join(sorted(common_patients)))
    print(f"\nList of common patients saved to 'common_patients.txt'")

    # Save individual filtered datasets
    for i, file in enumerate(tsv_files):
        if os.path.exists(file):
            output_file = f"filtered_{file}"
            merged_dfs[i].to_csv(output_file, sep='\t', index=False)
            print(f"Saved filtered dataset to {output_file}")

else:
    print("No files were successfully processed") 