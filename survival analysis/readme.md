# Survival analysis


Import R package
```R
library(SummarizedExperiment)
library(TCGAbiolinks)
library(survival)
library(survminer)
library(maftools)
```
Download miRNA expression data and clinical data. (we utilize TCGA-KIRC dataset.)
```
TCGAbiolinks:::getProjectSummary("TCGA-KIRC") 
clinical <- GDCquery_clinic(project = "TCGA-KIRC", type = "clinical")
query <- GDCquery(project = "TCGA-KIRC", 
                  experimental.strategy = "miRNA-Seq",
                  data.category = "Transcriptome Profiling", 
                  data.type = "miRNA Expression Quantification"
                  )
GDCdownload(query)
KIRC_miseq <- GDCprepare(query)
rownames(KIRC_miseq)<-KIRC_miseq$miRNA_ID
rpm_miR<-colnames(KIRC_miseq)[grep('reads_per_million',colnames(KIRC_miseq))]
miR<-KIRC_miseq[,rpm_miR]
colnames(miR) <- gsub("reads_per_million_miRNA_mapped_","", colnames(miR))
miR_matrix<-as.matrix(miR)
```


Input miRNAs for the analysis and integrate required files.
```
samplesTP <- TCGAquery_SampleTypes(barcode = colnames(miR_matrix),typesample = c("TP"))
gene_exp <- miR_matrix[c("hsa-mir-155"),samplesTP]#
names(gene_exp) <-  sapply(strsplit(names(gene_exp),'-'),function(x) paste(x[1:3],collapse="-"))
clinical$GENE <- gene_exp[clinical$submitter_id]
```

Determine the high and low expression levels of miRNA.
```
df<-subset(clinical,select=c(submitter_id,vital_status,days_to_death,days_to_last_follow_up,GENE))
df$os<-ifelse(df$vital_status=='Alive',df$days_to_last_follow_up,df$days_to_death)
df<-subset(df,select=c(submitter_id,vital_status,GENE,os))
df <- df[!is.na(df$GENE),]
df <- df[!is.na(df$os),]
df <- df[complete.cases(df),]
df$exp=''
df[df$GENE >= mean(df$GENE),]$exp <- "High"
df[df$GENE <  mean(df$GENE),]$exp <- "Low"
df[df$vital_status=='Dead',]$vital_status <- 2
df[df$vital_status=='Alive',]$vital_status <- 1
df$vital_status <- as.numeric(df$vital_status)
```


Construct survival analysis model and draw curve.

```
fit <- survfit(Surv(os, vital_status)~exp, data=df) 
ggsurvplot(fit, conf.int=TRUE, pval=TRUE, 
           legend.labs=c("high", "low"), legend.title="Expression",  
           palette=c("dodgerblue2", "orchid2"), 
           surv.median.line = "hv",
           legend = c(0.9,0.9),
           title="hsa-mir-155")
```
