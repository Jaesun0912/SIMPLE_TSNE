import yaml
import os
from itertools import product
from scripts.data import process_data
from scripts.pca import pca
from scripts.tsne import run_tSNE
from utils.config import check_config
from utils.utils import flush_print
import argparse

def get_grid_from_dict(dict_):
    list_params = []
    keys = list(dict_.keys())
    for key in keys:
        list_params.append(dict_[key])
    
    grid = product(*list_params)
    
    grid_list = []
    for point in grid:
        temp_dict = {}
        for idx, key in enumerate(keys):
            temp_dict[key] = point[idx]
        grid_list.append(temp_dict)
    
    return grid_list


def main():

    parser = argparse.ArgumentParser(description='Grid implementation for data conversion, pca, tsne')
    parser.add_argument('--data', action='store_true')
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--tsne', action='store_true')
    args = parser.parse_args()

    py_file_path = os.path.dirname(os.path.abspath(__file__))
    path_input = f'{py_file_path}/input.yaml'

    with open(path_input, 'r') as f:
        config = yaml.full_load(f)

    pca_params_grid = get_grid_from_dict(config['pca_parameters'])
    tsne_params_grid = get_grid_from_dict(config['tsne_parameters'])
    config['pca_params'] = pca_params_grid[0]
    config['tsne_params'] = tsne_params_grid[0]
    check_config(config)

    rank = 0

    for element in config['element']:
        config['element_type'] = element
        if args.data:  # TODO: change to call torch.load 1 time.
            rank = process_data(config)
        else:
            for pca_params in pca_params_grid:
                config['pca_params'] = pca_params
                if args.pca:
                    pca(config)
                else:
                    for tsne_params in tsne_params_grid:
                        config['tsne_params'] = tsne_params
                        if args.tsne:
                            run_tSNE(config)
    
    if rank == 0:
        flush_print('=== Normal termination of all process ===')

if __name__=='__main__':
    main()
