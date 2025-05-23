library(DESeq2)

dds <- DESeqDataSet(data_read, design = ~1)
dds <- dds[ rowSums(counts(dds)) > 10, ]
vsd <- vst(dds, blind = TRUE)
vsd_mat <- assay(vsd)

# Compute variance for each gene
gene_var <- apply(vsd_mat, 1, var)

# Select top 50
top_genes <- names(sort(gene_var, decreasing = TRUE))[1:50]
heatmap_data <- vsd_mat[top_genes, ]

if (!requireNamespace("pheatmap", quietly = TRUE)) {
  install.packages("pheatmap")
}
library(pheatmap)

sample_anno <- as.data.frame(colData(data_read)[, c("sample_type", "definition")])
rownames(sample_anno) <- colnames(heatmap_data)  # make sure rownames match columns of expression matrix


pheatmap(
  heatmap_data,
  cluster_rows = TRUE,
  cluster_cols = TRUE,
  show_rownames = TRUE,
  annotation_col = sample_anno,
  scale = "row",  # standardize each gene across samples
  fontsize_row = 6,
  fontsize_col = 6
)

