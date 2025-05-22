# Install TCGAbiolinks if not already installed
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}
BiocManager::install("TCGAbiolinks")

library(TCGAbiolinks)
library(SummarizedExperiment)

# Query COAD primary tumor samples
query_coad <- GDCquery(
  project = "TCGA-COAD",
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  workflow.type = "STAR - Counts",   # ✅ updated line
  sample.type = "Primary Tumor"
)

# Query READ primary tumor samples
query_read <- GDCquery(
  project = "TCGA-READ",
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  workflow.type = "STAR - Counts",   # ✅ updated line
  sample.type = "Primary Tumor"
)

# Download and prepare both
GDCdownload(query_coad)
GDCdownload(query_read)

data_coad <- GDCprepare(query_coad)
data_read <- GDCprepare(query_read)

# Combine gene expression matrices
combined_counts <- cbind(assay(data_coad), assay(data_read))

# Number of tumor samples
ncol(combined_counts)