# SIMPLE_TSNE
Accelerated t-SNE for SIMPLE-NN

## 1. Installation
First, please install anaconda: https://docs.anaconda.com/anaconda/install/index.html  
Then, clone this repository by
```
(base) $ git clone git@github.com:Jaesun0912/SIMPLE_TSNE.git
(base) $ cd SIMPLE_TSNE
```
Then, change prefix of `env.yaml` by `(base) $ vi env.yaml`  
The prefix at the bottom line of `env.yaml` should be `/path/to/your/anaconda3/envs/tsne`  
For example, `prefix: /home/sunny990912/anaconda3/envs/tsne`  
  
Then, create conda environment with following script.
```
(base) $ conda env create -f env.yaml
(base) $ conda activate tsne
```

## 2. Running

### 1) Configuration
Configuration required for running this code is in `input.yaml`  
Please change `input.yaml` based on your purpose.  

Meaning of {key: value} is...
`pca_parameter`: name, value(s) of parameter used in PCA. (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)  
`tsne_parameter`: name, value(s) of parameter used in t-SNE. (https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)  
* Warning: Some parameters available in sklearn are deprecated in multicore, tsnecuda.  
`element`: name of element.

The script conducts kind of 'grid search' for given above parameters.  
For example,  
```
element:
  - Ti
  - N

pca_parameter:
  n_components:
    - 20
    - 30

tsne_parameter:
  perplexity:
    - 20
    - 50
```
would conduct (n_components, perplexity) = (20, 20), (20, 50), (30, 20), (20, 50) for Titanium and Nitrogen (total: 8 t-SNE).  

For `tsne` key, please write one of 'sklearn' (to use sklearn.manifold.TSNE), 'opentsne' (to use openTSNE), 'multicore' (to use MulticoreTSNE), 'gpu' (to use tsnecuda).  

Note that for sklearn and opentsne, multi-thread is also used (but still, usually slower than MulticoreTSNE and tsnecuda).  
  
For `*_name` keys, you can set name of files. (Usually do not need to be changed)  

### Input and output
First, make `data` folder by
```
(tsne) $ mkdir data
```
The data points used for SIMPLE-NN (dataXXXX.pt) should be under `data` directory.  
Code will automatically read every data*.pt files under `./data/{raw_data_path}` which can be set in `raw_data_path` in `input.yaml`
For example, if you set `raw_data_path` in `input.yaml` as 'raw_data', the directory should be look like
```
- data/
| - raw_data/
  | - data1.pt
  | - data2.pt
  | ...
- script/
- utils/
...
```
or
```
- data/
| - raw_data/
  | - sub_dir_1/
    | - data1.pt
    | - data2.pt
    | ...
  | - sub_dir_2/
    | - data1.pt
    | - data2.pt
    | ...
- script/
- utils/
...
```
