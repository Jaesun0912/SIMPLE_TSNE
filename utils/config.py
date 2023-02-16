from sklearn.decomposition import PCA
import os
import torch


def check_config(config):

    data_root = f'./data/{config["raw_data_path"]}/'
    if not os.path.isdir(data_root):
        raise Exception(f'Check your raw_data_path.')

    flag=False
    for (root, dirs, files) in os.walk(data_root):
        for file in files:
            extension = file.split('.')[-1]
            if 'data' in file and extension == 'pt':
                file_path = os.path.join(root, file)
                flag = True
                break
        if flag:
            break
    if not flag:
        raise Exception('No available dataXXXX.pt file in raw_data_path.')
    data = torch.load(file_path)

    available_elements = data['x'].keys()
    for element_type in config['element']:
        if element_type not in available_elements:
            raise Exception(f'element: {element_type} is not valid. Please use elements in {available_elements}.')

    pca_params = config['pca_params']

    try:
        model = PCA(**pca_params)
    except:
        raise Exception('Some pca parameters are not valid.')

    lib_name = config['tsne'].lower()

    if lib_name == 'sklearn':
        from sklearn.manifold import TSNE
    elif lib_name == 'multicore':
        from MulticoreTSNE import MulticoreTSNE as TSNE
    elif lib_name == 'gpu':
        from tsnecuda import TSNE
    elif lib_name =='opentsne':
        from openTSNE import TSNE
    else:
        raise Exception('"tsne" in input.yaml must be one of "sklearn", "multicore", "gpu", "opentsne".')
    
    tsne_params = config['tsne_params']

    try:
        model = TSNE(**tsne_params)
    except:
        raise Exception('Some tsne parameters are not valid.')