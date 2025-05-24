import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_common_identifiers(file_path='common_identifiers.txt'):
    """Load the list of common patient identifiers."""
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f)

def process_large_file(file_path, common_ids, chunk_size=10000, id_column=None):
    """
    Process large TSV files in chunks and filter for common patients.
    
    Args:
        file_path: Path to the TSV file
        common_ids: Set of common patient identifiers
        chunk_size: Number of rows to process at a time
        id_column: Name of the identifier column ('Patient_ID' or 'Name')
    """
    output_path = file_path.replace('.tsv', '_filtered.tsv')
    first_chunk = True
    
    # Determine the identifier column if not provided
    if id_column is None:
        sample_df = pd.read_csv(file_path, sep='\t', nrows=5)
        if 'Patient_ID' in sample_df.columns:
            id_column = 'Patient_ID'
        elif 'Name' in sample_df.columns:
            id_column = 'Name'
        else:
            raise ValueError(f"Could not find identifier column in {file_path}")
    
    logging.info(f"Processing {file_path} using {id_column} as identifier")
    
    for chunk in pd.read_csv(file_path, sep='\t', chunksize=chunk_size):
        # Filter rows where the identifier is in common_ids
        filtered_chunk = chunk[chunk[id_column].isin(common_ids)]
        
        if not filtered_chunk.empty:
            # Write to output file
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            filtered_chunk.to_csv(output_path, sep='\t', index=False, 
                                mode=mode, header=header)
            first_chunk = False
    
    logging.info(f"Filtered data saved to {output_path}")
    return output_path

def main():
    # Load common identifiers
    common_ids = load_common_identifiers()
    logging.info(f"Loaded {len(common_ids)} common patient identifiers")
    
    # Define file configurations
    files_config = [
        {'path': 'clinical_dataframe.tsv', 'id_column': 'Patient_ID'},
        {'path': 'miRNA_dataframe.tsv', 'id_column': 'Patient_ID'},
        {'path': 'somatic_mutation_dataframe.tsv', 'id_column': 'Patient_ID'},
        {'path': 'phosphoproteomics_dataframe.tsv', 'id_column': 'Name'},
        {'path': 'proteomics_dataframe.tsv', 'id_column': 'Name'},
        {'path': 'transcriptomics_dataframe.tsv', 'id_column': 'Name'}
    ]
    
    # Process each file
    for file_config in files_config:
        file_path = file_config['path']
        if not Path(file_path).exists():
            logging.warning(f"File not found: {file_path}")
            continue
            
        try:
            output_path = process_large_file(
                file_path,
                common_ids,
                id_column=file_config['id_column']
            )
            logging.info(f"Successfully processed {file_path}")
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    main() 