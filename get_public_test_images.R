#GSEs <- c("GSE185090", "GSE156374"
GSEs <- c("GSE101658", "GSE40360")
#setwd("/home/tejasvi/cfDNA-MS/MCD_reference")
setwd("/home/tejasvi/cfDNA-MS/MS_reference")
arrays <- c(a27k="IlluminaHumanMethylation27k", a450k="IlluminaHumanMethylation450k", aEPIC="IlluminaHumanMethylationEPIC")
#aname <- 'aEPIC'
aname <- 'a450k'

#output_dir <- "/home/tejasvi/cfDNA-MS/MCD_reference"
output_dir <- "/home/tejasvi/cfDNA-MS/MS_reference"
if(!dir.exists(output_dir)) dir.create(output_dir, recursive=TRUE)

library(parallel)

cpg_list <- read.table("/home/tejasvi/Infinium/fromZhou/EPIC.hg38.bed", sep="\t", header=FALSE, stringsAsFactors=FALSE)
rownames(cpg_list) <- cpg_list[,4]

options(timeout=6000)
Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 20)

if(!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
packages <- c('minfi', 'GEOquery', 'stringdist', 'IlluminaHumanMethylationEPICanno.ilm10b4.hg19')
#packages <- c('minfi', 'GEOquery', 'IlluminaHumanMethylation450kmanifest', 'IlluminaHumanMethylation27kanno.ilmn12.hg19', 'IlluminaHumanMethylationEPICanno.ilm10b2.hg19')
for(pack in packages) if(!requireNamespace(pack, quietly = TRUE)) { BiocManager::install(pack) } else { library(pack, character.only=TRUE) }

results <- list()

for(i in GSEs)
{
  gse_accession <- i
  i <- 1
  index <- i
  path <- paste(output_dir, gse_accession, sep="/")
  array_name <- arrays[aname]
  getGEOSuppFiles(gse_accession)
  untar(paste(gse_accession, "/", gse_accession, "_RAW.tar", sep=""), exdir = paste(gse_accession, "/idat", sep=""))
  if(gse_accession != "GSE40360")
  {
    idatFiles <- list.files(paste(gse_accession, "/idat", sep=""), pattern = "idat.gz$", full = TRUE)
    sapply(idatFiles, gunzip, overwrite = TRUE)
    rgSet <- read.metharray.exp(paste(gse_accession, "/idat", sep=""), force=T)
  }

  bvSet <- getGEO(gse_accession, GSEMatrix =TRUE, getGPL=FALSE)[[index]]
  pD <- pData(bvSet)
  sample.names <- rownames(pD) <- pD[,'geo_accession']
  if(gse_accession == "GSE185090")
  {
    column_path <- "GSE185090/GSE185090_processed.txt.gz"
    header <- read.table(column_path, nrow=1, sep="\t", stringsAsFactors=F, header=F)
    pheno <- pD[,'supplementary_file']
    pheno <- as.vector(sapply(pheno, function(f) { rev(strsplit(f, "/")[[1]])[1] }))
    pheno <- sapply(unlist(header[1,amatch(pheno, header[1,], maxDist=Inf)]), function(f) { strsplit(f, "\\.")[[1]][1] })
    pheno <- as.vector(gsub(" ", "", pheno))
    pD$disease_group <- pheno
  } else if(gse_accession == "GSE156374") {
    column_path <- "GSE156374/GSE156374_normalized_values.txt.gz"
    header <- read.table(column_path, nrow=2, sep="\t", stringsAsFactors=F, header=F)[2,]
    pheno <- pD[,'supplementary_file']
    pheno <- as.vector(sapply(pheno, function(f) { rev(strsplit(f, "/")[[1]])[1] }))
    pheno <- sapply(unlist(header[1,amatch(pheno, header[1,], maxDist=Inf)]), function(f) { strsplit(f, "\\.")[[1]][1] })
    pheno <- as.vector(gsub(" ", "", pheno))
    pheno <- gsub("TLE/CTRL", "Control-TLE", pheno)
    pD$disease_group <-	pheno
  } else if(gse_accession == "GSE101658") {
    pheno <- pD[,'source_name_ch1']
    pheno <- gsub('hippo', 'Hippo', pheno)
    pheno <- gsub('yelinated MS Hippocampus', 'y_MS_Hipp', pheno)
    pD$disease_group <- pheno
  } else if(gse_accession == "GSE40360") {
    pheno <- pD[,'source_name_ch1']
    pheno[pheno == "Multiple sclerosis"] <- "MS"
    pheno[pheno == "Control"] <- "Ctrl"
    pD$disease_group <- pheno
    file = paste(output_dir, gse_accession, 'GSE40360_GenomeStudio_norm_bg.txt.gz', sep="/")
    command = paste("zcat", file, "| head -n 4 | tail -n 1 | sed 's/\t/\n/g' | grep -n 'AVG_Beta' | cut -f1 -d:")
    command = paste("| cut -f 1,$(echo $(", command, ") | sed 's/ /,/g')")
    command = paste("zcat", file, command, "| tail -n +4")
    command = paste(command, "| sed 's/\t/,/g' >", paste(file, "out", sep="."))
    system(command)
    BetaVals <- read.table(file=paste(file, 'out', sep="."), header=T, stringsAsFactors=F, sep=",")
    row.names(BetaVals) <- BetaVals[,1]
    BetaVals <- BetaVals[,2:ncol(BetaVals)]
    colnames(BetaVals) <- gsub(".AVG_Beta", "", colnames(BetaVals))
    library(preprocessCore)
    beta_norm <- as.data.frame(normalize.quantiles(as.matrix(BetaVals)))
    colnames(beta_norm) <- colnames(BetaVals)
    rownames(beta_norm) <- rownames(BetaVals)
    BetaVals <- beta_norm
    colnames(BetaVals) <- sapply(colnames(BetaVals), function(B) { rownames(pD)[(pD[,'title'] == B)] })
    results[[which(gse_accession == GSEs)]] <- list(pD, BetaVals)
  }
  if(gse_accession != "GSE40360")
  {
    sampleNames(rgSet) <- as.vector(sapply(sampleNames(rgSet), function(s) { strsplit(s, "_")[[1]][1] }))
    rownames(pD) <- pD$geo_accession
    pD <- pD[sampleNames(rgSet),]
    for(column_name in colnames(pD)) pData(rgSet)[,column_name] <- pD[,column_name]
    grSet <- preprocessQuantile(rgSet)
    BetaVals <- getBeta(grSet)
    tosave <- list(grSet, BetaVals)
    results[[which(gse_accession == GSEs)]] <- tosave
  }
}

if(GSEs[2] != "GSE40360")
{
  combined_data <- combineArrays(results[[1]][[1]], results[[2]][[1]], outType="IlluminaHumanMethylationEPIC")
  combined_beta <- getBeta(combined_data)
  rm(results)
} else {
  combined_data <- getBeta(results[[1]][[1]])
  combined_beta <- cbind(combined_data, results[[2]][[2]][rownames(combined_data),])
}
beta_loci <- cpg_list[rownames(combined_beta),1:3]
beta_loci <- paste(beta_loci[,1], beta_loci[,2], beta_loci[,3], sep="_")
beta_loci_dup <- table(beta_loci)
for(b in names(beta_loci_dup[beta_loci_dup > 1])) beta_loci[which(beta_loci == b)[-c(1)]] <- NA
combined_beta <- combined_beta[!is.na(beta_loci),]
beta_loci <- beta_loci[!is.na(beta_loci)]

rownames(combined_beta) <- beta_loci
if(GSEs[1] == "GSE185090")
{
  write.table(combined_beta, file="beta_values.txt", sep="\t", row.names=T, col.names=T, quote=F)
} else {
  write.table(combined_beta, file="MS_beta_values.txt", sep="\t", row.names=T, col.names=T, quote=F)
}

if(GSEs[2] != "GSE40360")
{
  pheno_table <- pData(combined_data)$disease_group
  pheno_table <- cbind(pheno_table, "Ctrl")
  pheno_table[grep('Control', pheno_table[,1], invert=T),2] <- "Disease"
  rownames(pheno_table) <- rownames(pData(combined_data))
  colnames(pheno_table) <- c("Pheno1", "Pheno2")
  pheno_table <- as.data.frame(pheno_table, stringsAsFactors=F)
  pheno_table[,"Pheno3"] <- "non-TLE"
  pheno_table[grep('TLE', pheno_table[,"Pheno1"]),"Pheno3"] <- "TLE"
  pheno_table[,"Pheno4"] <- "non-FCD"
  pheno_table[grep('FCD1', pheno_table[,"Pheno1"]),"Pheno4"] <- "FCD1"
  pheno_table[grep('FCD2', pheno_table[,"Pheno1"]),"Pheno4"] <- "FCD2"
  pheno_table[grep('FCD3', pheno_table[,"Pheno1"]),"Pheno4"] <- "FCD3"
  pheno_table[,"Pheno5"] <- "non-FCD"
  pheno_table[grep('FCD', pheno_table[,"Pheno1"]),"Pheno5"] <- "FCD"
  pheno_table[,"Pheno6"] <- "non-MCD"
  pheno_table[grep('mMCD|MOGHE|FCD1|FCD2|HME', pheno_table[,"Pheno1"], perl=T),"Pheno6"] <- "MCD1"
  pheno_table[grep('PMG|FCD3', pheno_table[,"Pheno1"], perl=T),"Pheno6"] <- "MCD3"
  pheno_out <- "pheno_label.txt"
} else {
  pheno_table <- pData(results[[1]][[1]])$disease_group
  names(pheno_table) <- rownames(pData(results[[1]][[1]]))
  pheno_table[rownames(results[[2]][[1]])] <- results[[2]][[1]]$disease_group
  pheno_table <- as.vector(pheno_table[colnames(combined_beta)])
  pheno_table <- cbind(pheno_table, "Ctrl")
  rownames(pheno_table) <- colnames(combined_beta)
  pheno_table[grep('Ctrl', pheno_table[,1], invert=T),2] <- "Disease"
  pheno_table[pheno_table[,1] == 'My_MS_Hipp',2] <- "Ctrl"
  pheno_out <- "MS_pheno_label.txt"
}

write.table(pheno_table, file=pheno_out, sep="\t", col.names=T, row.names=T, quote=F)
