main_path = '/home/tejasvi/epiclock/450/methylvals'
files = list.files(path=main_path, pattern='.txt', full.names=T, recursive=T)
setwd("/home/tejasvi/cfDNA-MS/other_reference")
output_dir <- "/home/tejasvi/cfDNA-MS/other_reference"


to_remove = 'lung|muscle|skin|saliva|umbilical|cervix|whole|buccal|prostate|bone|adipose|spleen|cartilage|liver|kidney|diaph|villus|breast|mucosa|fibroblast|thyroid|adrenal|aorta|blood|colon|esoph|heart|lymph|epithel|pancreas|cardi|stomach|CD4'
files = grep(to_remove, files, invert=T, perl=T, ignore.case=T, value=T)
phenotype <- rep('leukocyte', times=length(files))
phenotype[grep('leukocyte', files, invert=T, ignore.case=T)] <- 'Control-NCx'
phenotype[grep('cerebellum', files, ignore.case=T)] <- 'Control-Cerebellum'

phenotype <- cbind(phenotype, 'Ctrl')
colnames(phenotype) <- c('Pheno1', 'Pheno2')

sample_names <- as.vector(sapply(files, function(f) { strsplit(f, '/')[[1]][8] }))
rownames(phenotype) <- sample_names

beta_vals <- data.frame()
i = 0
for(f in files)
{
  i <- i + 1
  df <- scan(f, what=character(), sep="\n")
  header <- strsplit(df[1], "\t")[[1]]
  vals <- strsplit(df[2], "\t")[[1]]
  header <- header[3:length(header)]
  vals <- vals[3:length(vals)]
  beta_vals[sample_names[i],header] <- vals
  cat(i)
}

phenotype <- phenotype[!is.na(beta_vals[,1]),]
beta_vals <- beta_vals[!is.na(beta_vals[,1]),]

cpg_list <- read.table("/home/tejasvi/Infinium/fromZhou/EPIC.hg38.bed", sep="\t", header=FALSE, stringsAsFactors=FALSE)
rownames(cpg_list) <- cpg_list[,4]
new_names <- cpg_list[colnames(beta_vals),1:3]
new_names <- paste(new_names[,1], new_names[,2], new_names[,3], sep="_")
new_names_dup <- table(new_names)
new_names_dup <- new_names_dup[new_names_dup > 1]
for(n in names(new_names_dup)) new_names[which(new_names == n)[-c(1)]] <- NA

beta_vals <- beta_vals[,!is.na(new_names)]
new_names <- new_names[!is.na(new_names)]
colnames(beta_vals) <- new_names
beta_vals <- t(beta_vals)
beta_vals <- beta_vals[rownames(beta_vals) != "NA_NA_NA",]

write.table(beta_vals, file="other_beta_values.txt", sep="\t", row.names=T, col.names=T, quote=F)
write.table(phenotype, file="other_pheno_label.txt", sep="\t", col.names=T, row.names=T, quote=F)
