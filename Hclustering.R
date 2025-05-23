heat <- pheatmap(vsd_subset,
                 annotation_col = annotation_col,
                 cluster_cols = TRUE,
                 cluster_rows = TRUE,
                 scale = "row",
                 cutree_cols = 2,  # cut sample dendrogram into 2 clusters
                 cutree_rows = 3)  # cut gene dendrogram into 3 clusters


heat$tree_col  # sample clustering tree (dendrogram)
heat$tree_row  # gene clustering tree

sample_clusters <- cutree(heat$tree_col, k = 2)
table(sample_clusters, colData(merged_data)$sample_type)
