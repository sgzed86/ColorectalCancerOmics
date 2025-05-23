# ColorectalCancerOmics
This analysis is written in R using the data from TCGA, analyzing based on the available cancer markers.

5/22/2025 The underlying data is from 481 files with a size 2.03 gb. It took a little while to download directly into RStudio.

The Cancer Genome Atlas (TCGA), specifically the colorectal cancer cohort, which includes colon adenocarcinoma (COAD) and rectal adenocarcinoma (READ) cases. I ended up only using the READ cases. 166 cases with over 60k variables for gene expression associated with READ, it was the smaller dataset. 

5/23/2025 I created a heat map that shows each column as a tumor and each row as a gene expression for normal and READ patients. 

I did some hierarchal clustering on the normal and READ cases. And found a very interesting overlap in the 2 clusters.

sample_clusters Normal Tumor<br/>
              1      0   158<br/>
              2     41     8<br/>

I am very interested in the 8 cases that are overlapping...
