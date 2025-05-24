import pandas as pd
import os
from typing import Dict, Set
from functools import reduce

def read_first_column(file_path: str) -> Set[str]:
    """
    Read only the first column (identifier column) of a TSV file to save memory.
    Returns a set of unique identifiers.
    """
    # Read just the first chunk to get the column name
    first_chunk = pd.read_csv(file_path, sep='\t', nrows=1)
    id_column = None
    
    # Determine the ID column name
    if 'Patient_ID' in first_chunk.columns:
        id_column = 'Patient_ID'
    elif 'Name' in first_chunk.columns:
        id_column = 'Name'
    else:
        raise ValueError(f"Could not find identifier column in {file_path}")
    
    # Read only the identifier column
    ids = pd.read_csv(file_path, sep='\t', usecols=[id_column])
    return set(ids[id_column].unique())

def analyze_common_ids():
    """
    Analyze common identifiers across all TSV files in the current directory.
    """
    # Dictionary to store file names and their respective IDs
    file_ids: Dict[str, Set[str]] = {}
    
    # Process each TSV file
    for filename in os.listdir('.'):
        if filename.endswith('_dataframe.tsv'):
            try:
                print(f"\nProcessing {filename}...")
                ids = read_first_column(filename)
                file_ids[filename] = ids
                print(f"Found {len(ids)} unique identifiers")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Find common IDs across all files
    if file_ids:
        all_id_sets = list(file_ids.values())
        common_ids = reduce(lambda x, y: x & y, all_id_sets)
        
        print("\n=== Summary ===")
        print(f"\nNumber of common identifiers across all files: {len(common_ids)}")
        
        print("\nNumber of identifiers in each file:")
        for filename, ids in file_ids.items():
            print(f"{filename}: {len(ids)} identifiers")
        
        print("\nNumber of missing identifiers compared to common set:")
        for filename, ids in file_ids.items():
            missing = len(ids - common_ids)
            print(f"{filename}: {missing} missing")
        
        # Save common IDs to a file
        common_ids_list = sorted(list(common_ids))
        with open('common_identifiers.txt', 'w') as f:
            f.write('\n'.join(common_ids_list))
        print("\nCommon identifiers have been saved to 'common_identifiers.txt'")
    else:
        print("No TSV files were processed successfully.")

if __name__ == "__main__":
    analyze_common_ids() 