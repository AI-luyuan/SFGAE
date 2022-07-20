# SFGAE

SFGAE is a self-feature-based graph autoencoder model for predicting miRNA-disease associations. The novelty of SFGAE is to construct miRNA-self embeddings and disease-self embeddings, and let them be independent of graph interactions between two types of nodes. SFGAE can effectively overcome the over-smoothing issue of GNN-based models.


## Environment Requirement
The code has been tested under Python 3.7.5. The required packages are:
- dgl-cu102 == 0.4.3
- mxnet-cu102 == 1.6.0.post0
- numpy == 1.20.3
- pandas == 1.3.4
- pytorch == 1.8.0
- scikit-learn == 1.0.1


## Example to Run the Code

```python mainauto.py```


## List of Folders
### data
We provide two processed datasets, HMDD v2.0 and HMDD v3.2. 
The HMDD v2.0 dataset contains 495 miRNAs and 383 diseases, with 5430 associations between them.
The HMDD v3.2 dataset further complements the associations between the miRNA and disease nodes in v2.0 dataset. 
We also provide datasets for carrying out case studies on three diseases, *colon neoplasms*, *esophageal neoplasms*, and *kidney neoplasms*.

File descriptions:

all_mirna_disease_pairs.csv: miRNA-disease associations of HMDD v2.0; \
all_mirna_disease_pairs_3.csv: miRNA-disease associations of HMDD v3.2; \
all_mirna_disease_pairs_Colon: miRNA-disease associations used in case study of colon neoplasms;\
all_mirna_disease_pairs_Esophageal: miRNA-disease associations used in case study of esophageal neoplasms;\
all_mirna_disease_pairs_Kidney: miRNA-disease associations used in case study of kidney neoplasms;\
D_GSM.txt: Gaussian interaction profile kernel similarity;\
D_SSM1.txt: disease semantic similarity;\
D_SSM2.txt: disease semantic similarity;\
M_FSM.txt: miRNA functional similarity;\
M_GSM.txt: Gaussian interaction profile kernel similarity.


### weight
We share the weights of the trained SFGAE on the datasets HMDD v2.0 and HMDD v3.2. For each dataset, we provide 5 weights corresponding to 5-fold cross-validation. The AUC values are reported.

### survival analysis
We utilize TCGA-KIRC dataset, and perform a survival analysis on the top 40 miRNAs predicted by SFGAE for kidney neoplasms. 

## List of Files
utilsauto.py: data preparation;\
layers.py: construct GNN layers;\
model.py: construct SFGAE model;\
trainauto.py: train SFGAE model;\
mainauto.py: main file.

## License
MIT LICENSE
