import pandas as pd
import numpy as np

# Read all data files
clinical_df = pd.read_csv('cleaned_clinical_dataframe_filtered.tsv', sep='\t')
transcriptomics_df = pd.read_csv('cleaned_transcriptomics_dataframe_filtered.tsv', sep='\t')
phosphoproteomics_df = pd.read_csv('cleaned_phosphoproteomics_dataframe_filtered.tsv', sep='\t')
proteomics_df = pd.read_csv('cleaned_proteomics_dataframe_filtered.tsv', sep='\t')
mirna_df = pd.read_csv('cleaned_miRNA_dataframe_filtered.tsv', sep='\t')

# Set Patient_ID as index for all dataframes
clinical_df.set_index('Patient_ID', inplace=True)
transcriptomics_df.set_index('Patient_ID', inplace=True)
phosphoproteomics_df.set_index('Patient_ID', inplace=True)
proteomics_df.set_index('Patient_ID', inplace=True)
mirna_df.set_index('Patient_ID', inplace=True)

# Define clinical features for survival and recurrence
survival_clinical_features = [
    'pathologic_staging_distant_metastasis_pm',
    'tumor_stage_pathological',
    'histologic_type',
    'pathologic_staging_regional_lymph_nodes_pn',
    'number_of_lymph_nodes_positive_for_tumor_by_he_staining',
    'perineural_invasion',
    'adjuvant_post-operative_pharmaceutical_therapy',
    'survival_target'
]

recurrence_clinical_features = [
    'pathologic_staging_distant_metastasis_pm',
    'tumor_stage_pathological',
    'new_tumor_after_initial_treatment',
    'histologic_type',
    'pathologic_staging_regional_lymph_nodes_pn',
    'number_of_lymph_nodes_positive_for_tumor_by_he_staining',
    'perineural_invasion',
    'adjuvant_post-operative_pharmaceutical_therapy',
    'recurrence_target'
]

# Create survival dataframe
survival_df = pd.concat([
    clinical_df[survival_clinical_features],
    transcriptomics_df,
    phosphoproteomics_df,
    proteomics_df,
    mirna_df
], axis=1)

# Create recurrence dataframe
recurrence_df = pd.concat([
    clinical_df[recurrence_clinical_features],
    transcriptomics_df,
    phosphoproteomics_df,
    proteomics_df,
    mirna_df
], axis=1)

# Save the joined dataframes
survival_df.to_csv('joined_survival_dataframe.tsv', sep='\t')
recurrence_df.to_csv('joined_recurrence_dataframe.tsv', sep='\t')

# Print some basic information about the joined dataframes
print("\nSurvival Dataframe Info:")
print(f"Shape: {survival_df.shape}")
print("\nRecurrence Dataframe Info:")
print(f"Shape: {recurrence_df.shape}")

# Print the number of survival and recurrence events
print(f"\nNumber of survival events: {survival_df['survival_target'].sum()}")
print(f"Number of recurrence events: {recurrence_df['recurrence_target'].sum()}") 