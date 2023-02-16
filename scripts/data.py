import torch
from mpi4py import MPI
from collections import defaultdict
import pickle
import os
import math
import time
import numpy as np

from utils.utils import time_trace, flush_print


def adjust_idx_and_collate_dict(ls_dict: list[defaultdict]) -> defaultdict:
    number = 0
    max_num = 0

    return_dict = defaultdict(list)
    for dict_ in ls_dict:
        for (key, value) in dict_.items():
            for idx in range(len(value)):
                dict_[key][idx] += number
                if dict_[key][idx] > max_num:
                    max_num = dict_[key][idx]
            return_dict[key] += dict_[key]
        number = max_num + 1

    return return_dict

@time_trace
def process_data(config: dict):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    matrix_name = config['matrix_name']
    label_name = config['label_name']
    element_type = config['element_type']
    data_root = f'./data/{config["raw_data_path"]}/'

    if (os.path.isfile(f'./data/{matrix_name}') and
        os.path.isfile(f'./data/{label_name}')):
        if rank == 0:
            flush_print(f'Same file name is detected!: {matrix_name} or {label_name}')
            flush_print('Please try with another file name for new calculation...')

        return rank

    else:
        if rank == 0:
            flush_print(f'Starting processing data for element: {element_type}.')
            os.makedirs('./data/preprocess/', exist_ok=True)
        file_dirs = []
        for (root, dirs, files) in os.walk(data_root):
            for file in files:
                extension = file.split('.')[-1]
                if 'data' in file and extension == 'pt':
                    file_path = os.path.join(root, file)
                    file_dirs.append(file_path)
        
        length_data = len(file_dirs)
        part_size = length_data//world_size
        i_start = rank*part_size
        i_end = i_start + part_size

        if rank == 0:
            length = part_size
            div = min(20, length)

        if (rank == (world_size-1)):
            i_end = length_data

        tensor_list = []
        struct_type_dict = defaultdict(list)

        idx_count = 0
        t = time.time()
        for idx in range(i_start, i_end):
            if rank == 0:
                if idx % int((length)/div) == 0:
                    flush_print(f'{math.ceil(idx/(length)*100)}% done | Wall time: {round(time.time() - t, 3)} s')

            path_data = file_dirs[idx]
            data = torch.load(path_data)
            tensor_list.append(
                    data['x'][element_type]
                )

            struct_type = data['struct_type']
            idx_start = idx_count
            idx_count += data['N'][element_type]
            idx_end = idx_count

            struct_type_dict[struct_type] += [
                i for i in range(idx_start, idx_end)
            ]
        
        try:
            tensor_list = comm.gather(tensor_list, root=0)
            struct_type_dict = comm.gather(struct_type_dict, root=0)

            if rank == 0:
                tmp = []
                for tensor in tensor_list:
                    tmp.extend(tensor)

                data = torch.Tensor(np.concatenate(tmp))
                struct_type_dict = adjust_idx_and_collate_dict(struct_type_dict)

        except:
            flush_print('Data size is too big for using comm.gather.')
            flush_print('Save and load data.')
            os.makedirs('./data/tmp/', exist_ok=True)
            with open(f'./data/tmp/{rank}_{matrix_name}', 'wb') as f:
                pickle.dump(tensor_list, f)

            with open(f'./data/tmp/{rank}_{label_name}', 'wb') as f:
                pickle.dump(struct_type_dict, f)

            comm.Barrier()

            if rank == 0:
                flush_print('Too large data for using comm.gather.')
                flush_print('Saving and loading data.')
                tensor_list = []
                struct_type_dict = []
                for i in range(world_size):
                    with open(f'./data/tmp/{i}_{matrix_name}', 'rb') as f:
                        tensor_list.extend(pickle.load(f))

                    with open(f'./data/tmp/{i}_{label_name}', 'rb') as f:
                        struct_type_dict.append(pickle.load(f))
                
                data = torch.Tensor(np.concatenate(tensor_list))
                struct_type_dict = adjust_idx_and_collate_dict(struct_type_dict)

        if rank == 0:
            with open(f'./data/preprocess/{element_type}_{matrix_name}', 'wb') as f:
                pickle.dump(data, f)

            with open(f'./data/preprocess/{element_type}_{label_name}', 'wb') as f:
                pickle.dump(struct_type_dict, f)
            flush_print('--- Data generation normally terminated ---')

    return rank
