library(DESeq2)


# Use a simple design (~1) if no group info available
dds <- DESeqDataSet(merged_data, design = ~ 1)

Perform variance stabilizing transformation (VST) to normalize counts
vst_data <- vst(dds, blind = TRUE)

Extract the normalized expression matrix (genes x samples)
expr_matrix <- assay(vst_data)

Transpose matrix to samples x genes for clustering
expr_matrix_t <- t(expr_matrix)

#Inspect and choose number of clusters (k) via elbow plot
wss <- numeric(10)
for (k in 1:10) {
  km <- kmeans(expr_matrix_t, centers = k, nstart = 25)
  wss[k] <- km$tot.withinss
}
plot(1:10, wss, type = "b", xlab = "Number of clusters K", ylab = "Within-cluster sum of squares")

# run k-means clustering
set.seed(123)  
k <- 2
km_res <- kmeans(expr_matrix_t, centers = k, nstart = 25)

#Extract cluster assignments
clusters <- km_res$cluster

# Add cluster assignments back to sample metadata
colData(dds)$kmeans_cluster <- factor(clusters[match(colnames(dds), names(clusters))])

# View cluster assignments
table(colData(dds)$kmeans_cluster)

# Plot PCA colored by k-means clusters
library(ggplot2)
pca <- prcomp(expr_matrix_t)
pca_df <- data.frame(PC1 = pca$x[,1], PC2 = pca$x[,2], cluster = colData(dds)$kmeans_cluster)
ggplot(pca_df, aes(PC1, PC2, color = cluster)) +
  geom_point(size = 3) + theme_minimal() + labs(title = "PCA colored by k-means clusters")


