import os
import numpy as np
from numpy.lib.format import open_memmap
import random

sets = {
    'train', 'val'
}

# 'ntu/xview', 'ntu/xsub'
datasets = {
    'ntu/xview', 'ntu/xsub',
}

from tqdm import tqdm

for dataset in datasets:
    for set1 in sets:
        print(dataset, set1)
        data = np.load('/data/xcl_data/{}/{}_data_joint.npy'.format(dataset, set1))
        N, C, T, V, M = data.shape
        T1 = T // 2                     # T1 = 150
        # frames_per_group = T // T1    # Frames per group=5



        # reverse = open_memmap(
        #     '../data/{}/{}_data_joint_cut_150.npy'.format(dataset, set1),
        #     dtype='float32',
        #     mode='w+',
        #     shape=(N, 3, T1, V, M))
        # reverse[:, :, :T1, :, :] = data[:, :, ::2, :, :]


        reverse = open_memmap(
            '/data/xcl_data/{}/{}_data_joint_FR.npy'.format(dataset, set1),  # 修改保存的文件名
            dtype='float32',
            mode='w+',
            shape=(N, 3, T, V, M))
        for i in range(T1):
            frame_indices = random.sample(range(i * 2, (i + 1) * 2), 1)  # 从每五帧中随机选择一帧
            reverse[:, :, i, :, :] = data[:, :, frame_indices[0], :, :]
        reverse[:, :, (i+1):, :, :] = reverse[:, :, T:(T1-1):-1, :, :]