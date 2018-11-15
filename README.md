# CMD

The code for the model described in the paper "Conditional molecular design with deep generative models"
https://arxiv.org/abs/1805.00108

Modified by Gonzalo Colmenarejo

## Components

- **run.py** : main script
- **SSVAE.py** - model architecture
- **preprocessing.py** - functions for preprocessing
- **exp0** : default run of CMD with original dataset and analysis of diversity, correctness and novelty of output
- **exp1** : analysis of output (cor, div, nov) after training the CMD with increasing subsets of original training dataset
- **exp3** : modified CMD for natural products; analyis of cor, div, nov of output
- **exp4** : analysis of output (cor, div, nov) after training with orthogonal set vs clustered set
- **cix** : folder with myfuncs.py, a module created for performing all cheminformatic analyses

## Dependencies

- **TensorFlow**
- **NumPy**
- **RDKit**
- **chemfp**
- **scikit-learn**
- **pandas**
- **matplotlib**

## Environments

Two Anaconda environments are used to run the code: tf35 to run the CMD, and cix to run the analysis Jupyter notebooks
