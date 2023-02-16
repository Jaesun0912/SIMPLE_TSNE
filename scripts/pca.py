from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import os
import pickle
from utils.utils import time_trace, flush_print, dict_key_val_as_filename, dict_key_val_as_string


@time_trace
def pca(config: dict):
    '''
    Run PCA (Principal Component Analysis)
    '''
    matrix_name = config['matrix_name']
    element_type = config['element_type']
    pca_params = config['pca_params']
    mat_reduced_name = config['pca_matrix_name']
    param_name = dict_key_val_as_filename(pca_params)
    with open(f'./data/preprocess/{element_type}_{matrix_name}', 'rb') as f:
        data = pickle.load(f)

    filename = f'./data/preprocess/{element_type}{param_name}_{mat_reduced_name}'

    if os.path.isfile(filename):
        flush_print(f'Same file name is detected!: {filename}')
        flush_print('Please try with another file name for new calculation...')

        return

    # Scaling
    data_s = scale(data)

    # PCA
    flush_print(f'PCA starts: {dict_key_val_as_string(pca_params)}')
    pca = PCA(**pca_params)
    data_low = pca.fit_transform(data_s)
    flush_print('PCA done.')
    with open(filename, 'wb') as f:
        pickle.dump(data_low, f)

    flush_print('--- PCA normally terminated ---')

    return