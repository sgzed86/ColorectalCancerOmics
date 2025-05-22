library(SummarizedExperiment)
gene_counts <- assay(data_read)
gene_counts_df <- as.data.frame(t(gene_counts))
sample_metadata <- colData(data_read)
gene_counts_df$SampleID <- rownames(gene_counts_df)
sample_metadata_df <- as.data.frame(sample_metadata)
gene_counts_df <- cbind(sample_metadata_df[, c("patient", "sample_type", "definition")], gene_counts_df)

gene_counts_df