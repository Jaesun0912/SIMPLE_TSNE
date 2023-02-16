# SIMPLE_TSNE
Accelerated t-SNE for SIMPLE-NN

## 1. Installation
First, please install anaconda: https://docs.anaconda.com/anaconda/install/index.html
Then, clone this repository by
```
(base) $ git clone git@github.com:Jaesun0912/SIMPLE_TSNE.git
(base) $ cd SIMPLE_TSNE
```
Then, change prefix of `env.yaml` by `$ (base) vi env.yaml`  
The prefix at the bottom line of `env.yaml` should be `/path/to/your/anaconda3/envs/tsne`  
For example, `prefix: /home/sunny990912/anaconda3/envs/tsne`  
  
Then, create conda environment with following script.
```
(base) $ conda env create -f env.yaml
(base) $ conda activate tsne
```

## 2. Running
