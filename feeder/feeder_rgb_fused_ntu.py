# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from feeder import segment_rgbbody_ntu as rgb_roi

# visualization
import time
import math
# operation
from . import tools

# rgb --B
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                data_path,
                label_path,
                random_choose=False,
                random_move=False,
                random_flip=False,
                random_interval=False,
                random_rot=False,
                bone=False, 
                vel=False,
                random_roi_move=False,
                centralization=False,
                split='train',
                temporal_rgb_frames=5,
                window_size=-1,
                p_interval=1,
                debug=False,
                evaluation=False,
                mmap=True):
        self.debug = debug
        self.evaluation = evaluation
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.random_flip = random_flip
        self.p_interval = p_interval
        self.bone = bone
        self.vel = vel
        self.random_rot = random_rot
        self.random_interval = random_interval
        self.random_roi_move = random_roi_move
        self.centralization = centralization
        self.split = split
        self.temporal_rgb_frames = temporal_rgb_frames

        #self.rgb_path = '/mnt/nas/ntu-rgbd/NTU/RGB_videos/rgb_frames/'
        #self.rgb_path = '/mnt/nas/ntu-rgbd/NTU/RGB_videos/rgb_frames_all_rmbg/'
        self.rgb_path_ntu120 = '/media/bruce/2Tssd/data/ntu120/ntu_rgb_frames_crop/fivefs/'
        self.rgb_path_ntu60 = '/data/xcl_data/MMNets/ntu60/ntu_rgb_frames_crop/NTU60_twofs/'

        self.load_data(mmap)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize(size=224),
            transforms.Resize(size=(225,45*self.temporal_rgb_frames)),
            #transforms.ColorJitter(hue=.05, saturation=.05),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(20, resample=Image.BILINEAR),
            #transforms.RandomErasing(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_evaluation = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize(size=224),
            transforms.Resize(size=(225,45*self.temporal_rgb_frames)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_weight = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=225),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    # def load_data(self):
    #     # data: N C V T M
    #     npz_data = np.load(self.data_path)
    #     if self.split == 'train':
    #         self.data = npz_data['x_train']
    #         self.label = np.where(npz_data['y_train'] > 0)[1]
    #         self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
    #     elif self.split == 'test':
    #         self.data = npz_data['x_test']
    #         self.label = np.where(npz_data['y_test'] > 0)[1]
    #         self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
    #     else:
    #         raise NotImplementedError('data split only supports train/test')
    #     N, T, _ = self.data.shape
    #     self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
    #     self.N, self.C, self.T, self.V, self.M = self.data.shape

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        label = self.label[index]

        # add RGB features based on self.sample_name  -- B
        # print('self.sample_name',self.sample_name)
        sample_name_length = len(self.sample_name[index])
        filename = self.sample_name[index][sample_name_length-29:sample_name_length-9]
        # print(filename)
        action_id = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        # if True or self.random_interval:
        if not self.evaluation:
            rgb = rgb_roi.construct_st_roi(filename, self.evaluation, self.random_interval,self.random_roi_move,self.random_flip, self.temporal_rgb_frames)
        else:
            rgb = filename + '.png'

            if action_id < 61:
                rgb = Image.open(self.rgb_path_ntu60 + rgb)
            else:
                rgb = Image.open(self.rgb_path_ntu120 + rgb)
        width, height = rgb.size

        rgb = np.array(rgb.getdata())

        rgb = torch.from_numpy(rgb).float()
        T, C = rgb.size()

        rgb = rgb.permute(1, 0).contiguous()
        rgb = rgb.view(C, height, width)

        if self.evaluation:
            rgb = self.transform_evaluation(rgb)
        else:
            rgb = self.transform(rgb) # resize to 225x225

        # get data
        data_numpy = np.array(self.data[index])
        if self.centralization:
            data_numpy = tools.centralization(data_numpy)
        # processing
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)


        return data_numpy, rgb, label
