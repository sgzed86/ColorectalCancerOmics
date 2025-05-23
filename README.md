# ColorectalCancerOmics
This analysis is written in R using the data from TCGA, analyzing based on the available cancer markers.

5/22/2025 The underlying data is from 481 files with a size 2.03 gb. It took a little while to download directly into RStudio.

The Cancer Genome Atlas (TCGA), specifically the colorectal cancer cohort, which includes colon adenocarcinoma (COAD) and rectal adenocarcinoma (READ) cases. I ended up only using the READ cases. 166 cases with over 60k variables for gene expression associated with READ, it was the smaller dataset. 

5/23/2025 I created a heat map that shows each column as a tumor and each row as a gene expression for normal and READ patients. 

I did some hierarchal clustering on the normal and READ cases. And found some overlap in the 2 clusters.

### Sample Clustering Results

| Cluster | Normal Samples | Tumor Samples |
|---------|----------------|----------------|
| 1       | 0              | 158            |
| 2       | 41             | 8              |


I am very interested in the 8 cases that are overlapping...Tumors with lower malignancy or stromal contamination, Misclassified samples (rare in TCGA, but possible), Candidates for deeper clinical or molecular subtype review

Further analysis of the 8 cases in cluster 2 reveal 50/50 of early and late stage cancers:
### Tumor Stage Distribution – "Normal-like" Tumors

| Tumor Stage | Count |
|-------------|-------|
| Stage I     | 1     |
| Stage II    | 1     |
| Stage IIA   | 2     |
| Stage IIIB  | 2     |
| Stage IIIC  | 1     |

This suggests that clustering with normal samples may partly reflect stage, but other factors are likely influencing expression, such as:Tumor purity (high stromal/immune content may skew expression), Tumor location (right vs left colon), MSI status or other molecular subtypes
