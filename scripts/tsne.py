import matplotlib.pyplot as plt
import pickle
from joblib import parallel_backend
from utils.utils import dict_key_val_as_string, dict_key_val_as_filename, time_trace, flush_print
import os


@time_trace
def run_tSNE(config):
    '''
    Run t-SNE (t-Stochastic Neighbor Embedding)
    '''

    element_type = config['element_type']
    mat_reduced_name = config['pca_matrix_name']
    pca_param_name = dict_key_val_as_filename(config['pca_params'])
    pca_filename = f'./data/preprocess/{element_type}{pca_param_name}_{mat_reduced_name}'

    tsne_params = config['tsne_params']
    tsne_param_name = dict_key_val_as_filename(tsne_params)

    label_name = config['label_name']
    path_plot = config['plot_name']
    path_tsne = config['tsne_matrix_name']
    lib_name = config['tsne'].lower()

    to_plot = config['is_plot']

    tsne_plot_name = f'./data/{lib_name}/{element_type}{pca_param_name}{tsne_param_name}_{path_plot}'
    tsne_res_name = f'./data/{lib_name}/{element_type}{pca_param_name}{tsne_param_name}_{path_tsne}'

    with open(f'./data/preprocess/{element_type}_{label_name}', 'rb') as f:
        struct_type_dict = pickle.load(f)
    
    if os.path.isfile(tsne_plot_name) and os.path.isfile(tsne_res_name):
        flush_print(f'Same file name is detected!: {tsne_plot_name} or {tsne_res_name}')
        flush_print('Please try with another file name for new calculation...')

        return
    else:
        os.makedirs(f'./data/{lib_name}', exist_ok=True)

        with open(pca_filename, 'rb') as f:
            data = pickle.load(f)

        if lib_name == 'sklearn':
            from sklearn.manifold import TSNE
        elif lib_name == 'multicore':
            from MulticoreTSNE import MulticoreTSNE as TSNE
        elif lib_name == 'gpu':
            from tsnecuda import TSNE
        elif lib_name =='opentsne':
            from openTSNE import TSNE
        else:
            raise NotImplementedError()

        flush_print(f't-SNE starts: {dict_key_val_as_string(tsne_params)}')
        """
        if os.path.isfile(path_tSNE):
            with open(path_tSNE, 'rb') as f:
                transformed = pickle.load(f)
        """

        if lib_name.lower() != 'gpu':
            model = TSNE(verbose=True, n_jobs=-1, **tsne_params)  # TODO: make n_jobs selectable
            
        else:
            model = TSNE(verbose=True, **tsne_params)
        
        #if lib_name.lower() in ['opentsne','sklearn']:
        if lib_name.lower() != 'gpu':
            with parallel_backend('threading', n_jobs=-1):  # TODO: make n_jobs selectable
                if lib_name.lower() == 'opentsne':
                    transformed = model.fit(data)
                else:
                    transformed = model.fit_transform(data)

        else:
            transformed = model.fit_transform(data)

        flush_print(f't-SNE done!')
        with open(tsne_res_name, 'wb') as f:
            pickle.dump(transformed, f)

        if to_plot:
            xs = transformed[:, 0]
            ys = transformed[:, 1]

            fig, ax = plt.subplots(figsize=(10, 6))  # TODO: make figure size selectable

            for struct_type, idx_pair_list in struct_type_dict.items():
                xs_part = xs[idx_pair_list]
                ys_part = ys[idx_pair_list]

                ax.scatter(
                        xs_part, ys_part,
                        label=struct_type,
                        marker='+',
                        s=5,
                        )
            ax.legend(bbox_to_anchor=(1, 1))
            fig.tight_layout()

            # plt.show()
            plt.savefig(tsne_plot_name, bbox_inches='tight')
            plt.cla()
            flush_print('Saving plot done!')
            flush_print('--- tsne normally terminated ---')

        return