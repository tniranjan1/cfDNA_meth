#if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
#BiocManager::install(c("recountmethylation"))

library(recountmethylation)
library(HDF5Array)

cache_dir <- "/results/ep/study/hg38s/study250-cfDNA_prelim/cfDNA-MS/other_reference/cache"
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

# 1. Download/Access the pre-merged 450K and EPIC compilation
# This contains the ~453k probes shared between both platforms
h5se_path <- getdb_h5se_gm(platform = c("epic"), dfp = cache_dir) # Fetches the merged 'gm' database

# 2. Extract and filter metadata for "Whole Blood" samples
metadata <- as.data.frame(colData(h5se_path))
relevant = c("blood", "cells;blood;white_blood_cell", "blood;whole_blood",
             "blood;peripheral;peripheral_blood",
             "blood;peripheral;peripheral_blood;peripheral_blood_leukocytes",
             "blood;buffy_coat;peripheral;peripheral_blood")
healthy_terms = c("NA", "control", "normal", "case;normal", "healthy", "healthy;control",
                  "normal;healthy", "normal;control")   
blood_samples <- subset(metadata,
                        tissue %in% relevant &
                        disease %in% healthy_terms) # Filter for healthy blood-type controls

# 3. Subset the HDF5-SummarizedExperiment to your selected samples
gm_blood <- h5se_path[, rownames(blood_samples)]

# 4. Extract Beta-values (Methylation fractions 0 to 1)
# Note: This may take time depending on the number of samples selected
beta_matrix <- getBeta(gm_blood)

# 5. Convert the probe names to hg38 genomic coordinates
probe_info <- rownames(beta_matrix)
probe_coords <- read.table("/home/tejasvi/Infinium/fromZhou/EPIC.hg38.bed", sep="\t", header=FALSE, stringsAsFactors=FALSE)
# probe id in 4th column is not in the same order as in rownames(beta_matrix), so need to match to get the row id for each probe
probe_map <- match(probe_info, probe_coords[,4])
beta_matrix <- as.data.frame(as.matrix(beta_matrix))
# re-order beta_matrix rows to match probe_coords by sort-ordering probe_map
beta_matrix <- beta_matrix[order(probe_map), ]
probe_map <- probe_map[order(probe_map)]

new_names <- probe_coords[probe_map, 1:3]
new_names <- paste(new_names[,1], new_names[,2], new_names[,3], sep="_")
duplicate_names <- table(new_names)
duplicate_names <- duplicate_names[duplicate_names > 1]
for(n in names(duplicate_names)) new_names[which(new_names == n)[-c(1)]] <- NA
beta_matrix <- beta_matrix[!is.na(new_names), ]
new_names <- new_names[!is.na(new_names)]
rownames(beta_matrix) <- new_names

# 6. Get rownames from file "other_beta_values.txt" to ensure only common CpGs are retained
other_beta_vals <- read.table("/results/ep/study/hg38s/study250-cfDNA_prelim/cfDNA-MS/other_reference/other_beta_values.txt", sep="\t", header=TRUE, row.names=1, stringsAsFactors=FALSE)
common_cpgs <- intersect(rownames(beta_matrix), rownames(other_beta_vals))
# assign position number to common_cpgs based on order in other_beta_vals
common_cpgs_order <- data.frame(pos=1:nrow(other_beta_vals), row.names=rownames(other_beta_vals))[common_cpgs,]
common_cpgs <- common_cpgs[order(common_cpgs_order)]
beta_matrix <- beta_matrix[common_cpgs, ]

# 7. Print beta_matrix as tab-delimited file
output_file <- "/results/ep/study/hg38s/study250-cfDNA_prelim/cfDNA-MS/other_reference/blood_beta_values.txt"
write.table(beta_matrix, file=output_file, sep="\t", row.names=T, col.names=T, quote=F)
# Also print phenotype file
phenotype <- data.frame(Pheno1=rep("leukocyte", ncol(beta_matrix)), Pheno2=rep("Ctrl", ncol(beta_matrix)))
rownames(phenotype) <- colnames(beta_matrix)
output_file <- "/results/ep/study/hg38s/study250-cfDNA_prelim/cfDNA-MS/other_reference/blood_pheno_label.txt"
write.table(phenotype, file=output_file, sep="\t", col.names=T, row.names=T, quote=F)