# D4: Deep Drug-drug interaction Discovery and Demystification

D4 is a novel method for predicting drug-drug interactions along with their mechnsims of interaction. It uses 11 different mechanisms  at pharmacokinetic, pharmacodynamic, multi-pathway, and pharmacogenetic levels. D4 utlizes a neuro-symbolic deep learning strategies to encode for background knowledge about basic biological processes and phenomena about drugs. 

This repository contains scripts which were used to build supervised artificial neural network model, an optimizion of the artificial neural network model using hypers, along with the script for drewing AUCs for evaluting the model's pereformance.

# Dependencies
To install python dependencies run: pip install -r requirements.txt

# Scripts
The scripts require embedding files, which can be found: https://bio2vec.cbrc.kaust.edu.sa/data/D4/embeddings/

- Hypers.py: This scripts requires 2 inputs file: 1) the embedding; and 2) the DDIs file represting each drug as a vector. 
- ANNmodel.py: This scripts requires 2 inputs file: 1) the embedding; and 2) the DDIs file represting each drug as a vector. The scripts comes after Hypers.py to use the best hyperparameters.
- AUC.py: This scripts requires 1 input file that is the.pckl file to drew the AUC curve. 

# Data
- D4 predcitions of novel DDIs can be found at: https://bio2vec.cbrc.kaust.edu.sa/data/D4/predcitions/
- D4 embeddings of all drugs can be found at: https://bio2vec.cbrc.kaust.edu.sa/data/D4/embeddings/

# Citation
In case using our scroipts, datasets, or predictions, please cite our work: 
