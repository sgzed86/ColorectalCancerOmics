library(TCGAbiolinks)

# Query COAD for normal tissue RNA-seq (STAR counts)
query_coad_norm <- GDCquery(
  project = "TCGA-COAD",
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  workflow.type = "STAR - Counts",
  sample.type = "Solid Tissue Normal"
)

GDCdownload(query_coad_norm)
data_coad_norm <- GDCprepare(query_coad_norm)

# Make sure the rownames (genes) match first:
common_genes <- intersect(rownames(data_read), rownames(data_coad_norm))

data_read_filtered <- data_read[common_genes, ]
data_coad_norm_filtered <- data_coad_norm[common_genes, ]

# Combine
merged_data <- cbind(data_read_filtered, data_coad_norm_filtered)

# Get common column names
common_cols <- intersect(colnames(colData(data_read)), colnames(colData(data_coad_norm)))

# Align and coerce to DataFrame
colData(data_read) <- as(colData(data_read)[, common_cols], "DataFrame")
colData(data_coad_norm) <- as(colData(data_coad_norm)[, common_cols], "DataFrame")

# Ensure same column order
colData(data_coad_norm) <- colData(data_coad_norm)[, common_cols]

common_genes <- intersect(rownames(data_read), rownames(data_coad_norm))

data_read_filtered <- data_read[common_genes, ]
data_coad_filtered <- data_coad_norm[common_genes, ]

merged_data <- cbind(data_read_filtered, data_coad_filtered)

colData(merged_data)$sample_type <- factor(c(
  rep("Tumor", ncol(data_read_filtered)),
  rep("Normal", ncol(data_coad_filtered))
))

library(DESeq2)

dds <- DESeqDataSet(merged_data, design = ~ sample_type)
dds <- dds[ rowSums(counts(dds)) > 10, ]  # filter low-expression genes
vsd <- vst(dds, blind = TRUE)
vsd_mat <- assay(vsd)  # log2-normalized expression matrix

library(matrixStats)
top_genes <- head(order(rowVars(vsd_mat), decreasing = TRUE), 50)
vsd_subset <- vsd_mat[top_genes, ]

annotation_col <- data.frame(
  Group = colData(merged_data)$sample_type
)
rownames(annotation_col) <- colnames(vsd_subset)

library(pheatmap)

pheatmap(vsd_subset,
         annotation_col = annotation_col,
         cluster_cols = TRUE,
         cluster_rows = TRUE,
         scale = "row",  # z-score normalization by gene
         show_rownames = TRUE,
         show_colnames = FALSE,
         fontsize_row = 8,
         main = "Top 50 Variable Genes - Tumor vs Normal")







