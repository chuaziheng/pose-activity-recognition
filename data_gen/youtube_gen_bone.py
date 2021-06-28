import os
import numpy as np
from numpy.lib.format import open_memmap
import argparse
from tqdm import tqdm

paris = {
    'xview': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
        (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
        (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
        (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
        (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
    ),
    'xsub': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
        (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
        (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
        (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
        (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
    ),

    'kinetics': (
        (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1),
        (6, 5), (7, 6), (8, 2), (9, 8), (10, 9), (11, 5),
        (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)
    )
}


def gen_bone_data(arg):
    """Generate bone data from joint data for NTU skeleton dataset"""
    if arg.data_path:
        data = np.load(arg.data_path)
    else:
        data = np.load(r'C:\Users\chuaz\file-upload\dgnn\data\test_data_joint.npy')  # TODO: not hardcode
    N, C, T, V, M = data.shape
    if arg.data_path:
        fp_sp = open_memmap(
            arg.data_path,
            dtype='float32',
            mode='w+',
            shape=(N, 3, T, V, M))
    else:
        fp_sp = open_memmap(
            r'C:\Users\chuaz\file-upload\dgnn\data\test_data_bone.npy',
            dtype='float32',
            mode='w+',
            shape=(N, 3, T, V, M))

    # Copy the joints data to bone placeholder tensor
    fp_sp[:, :C, :, :, :] = data
    for v1, v2 in tqdm(paris['xview']):
        # Reduce class index for NTU datasets
        v1 -= 1
        v2 -= 1
        # Assign bones to be joint1 - joint2, the pairs are pre-determined and hardcoded
        # There also happens to be 25 bones
        fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]


if __name__ == '__main__':
    print('In bone gen')
    parser = argparse.ArgumentParser(description='Generate bone data from joint data for youtube datasets.')
    parser.add_argument('--data_path')
    arg = parser.parse_args()
    gen_bone_data(arg)

