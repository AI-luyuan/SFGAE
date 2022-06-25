# SFGAE

SFGAE is a self-feature-based graph autoencoder model for predicting miRNA-disease associations. The novelty of SFGAE is to construct miRNA-self embeddings and disease-self embeddings, and let them be independent of graph interactions between two types of nodes. SFGAE can effectively overcome the over-smoothing issue of GNN-based models.


## Environment Requirement
The code has been tested under Python 3.7.5. The required packages are:
- dgl == 0.4.3
- mxnet == 1.6.0.post0
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
The HMDD v3.2 dataset contains 11930 associations between the miRNA and disease nodes in HMDD v2.0 dataset. 
We also provide datasets for carrying out case studies on three diseases, *colon neoplasms*, *esophageal neoplasms*, and *kidney neoplasms*.


all_mirna_disease_pairs.csv: miRNA-disease associations of HMDD v2.0 
all_mirna_disease_pairs_3.csv: miRNA-disease associations of HMDD v3.2 
D_GSM.txt: Gaussian interaction profile kernel similarity
D_SSM1.txt: Disease semantic similarity
D_SSM2.txt:Disease semantic similarity
M_FSM.txt: miRNA functional similarity
M_GSM.txt: Gaussian interaction profile kernel similarity


## Weights of the trained SFGAE
We share the weights of the trained SFGAE. The weights are the result of 5-fold cross validation.


## Survival analysis
We performed a survival analysis of the top 40 miRNAs predicted by SFGAE for kidney neoplasms. 
We provide all the analysis results in the folder "Survival_analysis".
