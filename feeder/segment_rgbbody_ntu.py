#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:23:37 2019

@author: bruce
"""
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import time

frame_path = '/data/xcl_data/nturgb+d_rgb/nturgb+d_60_rgb/nturgb+d_frames/'     # RGB帧的位置
frame_path_120 = '/media/bruce/2Tssd2/ntu120/ntu_rgb_frames/'                   # 
openpose_path = '/data/xcl_data/MMNets/openpose/openpose/'                      # 由openpose获得的骨关节点的位置坐标
openpose_path_120 = '/media/bruce/2T/data/openpose120/'                         # 
save_path = '/data/xcl_data/MMNets/ntu60/ntu_rgb_frames_crop/NTU60_twofs/'            # 裁剪后的图片的保存位置
save_path_120 = '/data/xcl_data/MMNets/ntu120/ntu_rgb_frames_crop/fivefsz'      # 
depth_path = '/data/xcl_data/nturgb+d_depth_masked/nturgb+d_depth_masked/nturgb+d_depth_masked/'      # 深度图的存放位置
depth_path_120 = '/mnt/nas/ntu-rgbd/NTU120/Masked depth maps/nturgb+d_depth_masked/'
debug = False

def filename_construct(setup_id, camera_id, subject_id, duplicate_id, action_id):   # 文件名构建
    skeleton_file_name = ''
    if setup_id/10 >= 1:
        skeleton_file_name = skeleton_file_name +'S0' + str(setup_id)
    else:
        skeleton_file_name = skeleton_file_name + 'S00' +  str(setup_id)

    if camera_id/10 >= 1:
        skeleton_file_name = skeleton_file_name + 'C0' +  str(camera_id)
    else:
        skeleton_file_name = skeleton_file_name + 'C00' +  str(camera_id)

    if subject_id/100 >= 1:
        skeleton_file_name = skeleton_file_name + 'P' +  str(subject_id)
    elif subject_id/10 >= 1:
        skeleton_file_name = skeleton_file_name + 'P0' +  str(subject_id)
    else:
        skeleton_file_name = skeleton_file_name + 'P00' +  str(subject_id)

    if duplicate_id/10 >= 1:
        skeleton_file_name = skeleton_file_name + 'R0' +  str(duplicate_id)
    else:
        skeleton_file_name = skeleton_file_name + 'R00' +  str(duplicate_id)

    if action_id/100 >= 1:
        skeleton_file_name = skeleton_file_name + 'A' +  str(action_id)
    elif action_id/10 >= 1:
        skeleton_file_name = skeleton_file_name + 'A0' +  str(action_id)
    else:
        skeleton_file_name = skeleton_file_name + 'A00' +  str(action_id)

    return skeleton_file_name

def openposeFile(frame_file, frame, skeleton_file_name, openpose_path):
    frame_file_ = frame_file + '/' + str(frame) + '.jpg'
    frame_ = '';
    if frame/100 >= 1:
        frame_ = str(frame)
    elif frame/10 >= 1:
        frame_ = '0' + str(frame)
    else:
        frame_ ='00' + str(frame)
    openpose_file_ = openpose_path + skeleton_file_name + '/' + skeleton_file_name + '_rgb_000000000'+ frame_ + '_keypoints.json'

    return openpose_file_, frame_file_                                  # openpose_file_指openpose处理后的.json文件，frame_file_指的是matlab处理后的.jpg图片

def cropBody(openpose_file, frame_file, action_id, flip):
    #upper=Image.new( 'RGB' , (224,112) , (0,0,0) )
    #lower=Image.new( 'RGB' , (224,112) , (0,0,0) )
    #whole=Image.new( 'RGB' , (224,448) , (0,0,0) )

    frame = Image.open(frame_file)                                                   # 读取原始图片
    frame_width, frame_height = frame.size                                           # frame_width，frame_height分别是图片的宽和高

    if openpose_file:   # 如果存在文件
        with open(openpose_file, 'r') as f:   # 读取文件
            skeleton = json.load(f)           # 加载.json文件


    # calculate which people?
    if len(skeleton['people']) == 1 or action_id < 50: # or action_id > 49: 如果只有一个人
        people_index = 0                                                # 初始化people_index=0
        if len(skeleton['people']) > 1:
            #print('frame_file: ', frame_file)
            #print('number of prople:', len(skeleton['people']))
            frame_file_split = frame_file.split('/')              # 将frame_file按/分隔开来
            frame_num = int(frame_file_split[7].split('.')[0])
            if frame_num/1000 >= 1:
                depth_frame_file = depth_path + frame_file_split[6] +'/MDepth-0000'+str(frame_num)+'.png'
            elif frame_num/100 >= 1:
                depth_frame_file = depth_path + frame_file_split[6] +'/MDepth-00000'+str(frame_num)+'.png'
            elif frame_num/10 >= 1:
                depth_frame_file = depth_path + frame_file_split[6] +'/MDepth-000000'+str(frame_num)+'.png'
            else:
                depth_frame_file = depth_path + frame_file_split[6] +'/MDepth-0000000'+str(frame_num)+'.png'
            # print(depth_frame_file)

            depth_frame = Image.open(depth_frame_file)
            depth_frame = depth_frame.resize((1338,1080))
            depth_frame_arr = np.fromiter(iter(depth_frame.getdata()), np.uint16)
            depth_frame_arr.resize(1080, 1338)

            people_dist_min = 4500
            joint = 1
            for p in range(len(skeleton['people'])):
                #for i in []: # openpose joint 2, 5, 8, 11
                x = int(skeleton['people'][p]['pose_keypoints_2d'][(joint+1)*3-3])
                y = int(skeleton['people'][p]['pose_keypoints_2d'][(joint+1)*3-2])
                k = 0
                if x >= 1338:
                    x = 1336
                    #print('frame_file: x ', frame_file)
                if y >= 1080:
                    y = 1078
                    #print('frame_file: y ', frame_file)
                #print('(x, y): ', x, y)
                people_dist = 0
                for i in [-3, -2,-1, 0, 1, 2, 3]:
                    for j in [-3, -2,-1, 0, 1, 2, 3]:
                        if depth_frame_arr[y+i][x+j-291] > 0:
                            people_dist = people_dist + depth_frame_arr[y+i][x+j-291]
                            k = k + 1
                #print(p, people_dist)
                if people_dist > 0:
                    people_dist = people_dist/k
                    #print('people_dist, k: ',people_dist, k)
                    if people_dist <= people_dist_min:
                        people_dist_min = people_dist
                        people_index = p


        # if len(skeleton['people']) > 1:
        #     for p in range(len(skeleton['people'])):
        #         people_index = p


        if len(skeleton['people']) < 1:
            return ''

        # head_x = skeleton['people'][people_index]['pose_keypoints_2d'][0]
        # head_y = skeleton['people'][people_index]['pose_keypoints_2d'][1]
        L_hand_x = skeleton['people'][people_index]['pose_keypoints_2d'][12]
        L_hand_y = skeleton['people'][people_index]['pose_keypoints_2d'][13]
        R_hand_x = skeleton['people'][people_index]['pose_keypoints_2d'][21]
        R_hand_y = skeleton['people'][people_index]['pose_keypoints_2d'][22]
        # L_leg_x = skeleton['people'][people_index]['pose_keypoints_2d'][30]
        # L_leg_y = skeleton['people'][people_index]['pose_keypoints_2d'][31]
        # R_leg_x = skeleton['people'][people_index]['pose_keypoints_2d'][39]
        # R_leg_y = skeleton['people'][people_index]['pose_keypoints_2d'][40]
        #print(',head_x: ', head_x)
        # head = frame.crop((head_x-48, head_y - 48, head_x + 48, head_y + 48)) #2*3+1
        L_hand = frame.crop((L_hand_x-64, L_hand_y - 64, L_hand_x + 64, L_hand_y + 64))
        R_hand = frame.crop((R_hand_x-64, R_hand_y - 64, R_hand_x + 64, R_hand_y + 64))
        # L_leg = frame.crop((L_leg_x-48, L_leg_y - 48, L_leg_x + 48, L_leg_y + 48))
        # R_leg = frame.crop((R_leg_x-48, R_leg_y - 48, R_leg_x + 48, R_leg_y + 48))

        frame_concat=Image.new( 'RGB' , (128,256) , (0,0,0) )
        if flip:
            # frame_concat.paste(head, (0,0))
            frame_concat.paste(L_hand, (0,0))
            frame_concat.paste(R_hand, (0,128))
            # frame_concat.paste(L_leg, (0,288))
            # frame_concat.paste(R_leg, (0,384))
            frame_concat=frame_concat.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            # frame_concat.paste(head, (0,0))
            frame_concat.paste(L_hand, (0,0))
            frame_concat.paste(R_hand, (0,128))
            # frame_concat.paste(L_leg, (0,288))
            # frame_concat.paste(R_leg, (0,384))     # 至此完成图片补丁空间上的paste
        #print('frame_concat   if')
        return frame_concat

    elif len(skeleton['people']) > 1:  # 两个人的情况
        # filter the non-subjects
        #print('frame_file: ', frame_file)
        #print('number of prople:', len(skeleton['people']))

        # cropping the area
        # head_x = skeleton['people'][0]['pose_keypoints_2d'][0]
        # head_y = skeleton['people'][0]['pose_keypoints_2d'][1]
        L_hand_x = skeleton['people'][0]['pose_keypoints_2d'][12]
        L_hand_y = skeleton['people'][0]['pose_keypoints_2d'][13]
        R_hand_x = skeleton['people'][0]['pose_keypoints_2d'][21]
        R_hand_y = skeleton['people'][0]['pose_keypoints_2d'][22]
        # L_leg_x = skeleton['people'][0]['pose_keypoints_2d'][30]
        # L_leg_y = skeleton['people'][0]['pose_keypoints_2d'][31]
        # R_leg_x = skeleton['people'][0]['pose_keypoints_2d'][39]
        # R_leg_y = skeleton['people'][0]['pose_keypoints_2d'][40]

        # head = frame.crop((head_x-24, head_y - 48, head_x + 24, head_y + 48)) #2*3+1
        L_hand = frame.crop((L_hand_x-32, L_hand_y - 64, L_hand_x + 32, L_hand_y + 64))
        R_hand = frame.crop((R_hand_x-32, R_hand_y - 64, R_hand_x + 32, R_hand_y + 64))
        # L_leg = frame.crop((L_leg_x-24, L_leg_y - 48, L_leg_x + 24, L_leg_y + 48))
        # R_leg = frame.crop((R_leg_x-24, R_leg_y - 48, R_leg_x + 24, R_leg_y + 48))

        # head_x_1 = skeleton['people'][1]['pose_keypoints_2d'][0]
        # head_y_1 = skeleton['people'][1]['pose_keypoints_2d'][1]
        L_hand_x_1 = skeleton['people'][1]['pose_keypoints_2d'][12]
        L_hand_y_1 = skeleton['people'][1]['pose_keypoints_2d'][13]
        R_hand_x_1 = skeleton['people'][1]['pose_keypoints_2d'][21]
        R_hand_y_1 = skeleton['people'][1]['pose_keypoints_2d'][22]
        # L_leg_x_1 = skeleton['people'][1]['pose_keypoints_2d'][30]
        # L_leg_y_1 = skeleton['people'][1]['pose_keypoints_2d'][31]
        # R_leg_x_1 = skeleton['people'][1]['pose_keypoints_2d'][39]
        # R_leg_y_1 = skeleton['people'][1]['pose_keypoints_2d'][40]

        # head_1 = frame.crop((head_x_1-24, head_y_1 - 48, head_x_1 + 24, head_y_1 + 48)) #2*3+1
        L_hand_1 = frame.crop((L_hand_x_1-36, L_hand_y_1 - 64, L_hand_x_1 + 36, L_hand_y_1 + 64))
        R_hand_1 = frame.crop((R_hand_x_1-36, R_hand_y_1 - 64, R_hand_x_1 + 36, R_hand_y_1 + 64))
        # L_leg_1 = frame.crop((L_leg_x_1-24, L_leg_y_1 - 48, L_leg_x_1 + 24, L_leg_y_1 + 48))
        # R_leg_1 = frame.crop((R_leg_x_1-24, R_leg_y_1 - 48, R_leg_x_1 + 24, R_leg_y_1 + 48))

        frame_concat=Image.new( 'RGB' , (128,256) , (0,0,0) )
        if flip:
            # frame_concat.paste(head, (0,0))
            # frame_concat.paste(head_1, (48,0))
            frame_concat.paste(R_hand, (0,0))
            frame_concat.paste(R_hand_1, (64,0))
            frame_concat.paste(L_hand, (0,128))
            frame_concat.paste(L_hand_1, (64,128))
            # frame_concat.paste(R_leg, (0,288))
            # frame_concat.paste(R_leg_1, (48,288))
            # frame_concat.paste(L_leg, (0,384))
            # frame_concat.paste(L_leg_1, (48,384))
            frame_concat=frame_concat.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            # frame_concat.paste(head, (0,0))
            # frame_concat.paste(head_1, (48,0))
            frame_concat.paste(L_hand, (0,0))
            frame_concat.paste(L_hand_1, (64,0))
            frame_concat.paste(R_hand, (0,128))
            frame_concat.paste(R_hand_1, (64,128))
            # frame_concat.paste(L_leg, (0,288))
            # frame_concat.paste(L_leg_1, (48,288))
            # frame_concat.paste(R_leg, (0,384))
            # frame_concat.paste(R_leg_1, (48,384))
        #print('frame_concat   elif')
        return frame_concat
    else:
        #print(len(skeleton['people']))
        return ''

# to set the continue point S006C002P019R002A039



done = False
'''
for setup_id in range(1,21):     # 1:20 Diferernt height and distance
    if setup_id < sss:
        continue
    for camera_id in range(1,4):     # 1:3 camera views
        if setup_id < sss + 1 and camera_id < ccc:
            continue
        for subject_id in range(1,41):   # 1:40 distinct subjects aged between 10 to 35
            if setup_id < sss + 1 and camera_id < ccc + 1 and subject_id < ppp:
                continue
            for duplicate_id in range(1,3):  # 1:2 Performance action twice, one to left camera, one to right camera
                if setup_id < sss + 1 and camera_id < ccc + 1 and subject_id < ppp + 1 and duplicate_id < rrr:
                    continue
                for action_id in range(1,61):    # 1:60 Action class [11,12,30,31,53]
                    if setup_id < sss + 1 and camera_id < ccc + 1 and subject_id < ppp + 1 and duplicate_id < rrr +1 and action_id < aaa:
                        continue
'''
def construct_st_roi(filename, evaluation=False, random_interval=False,random_roi_move=False,random_flip=False, temporal_rgb_frames=5):
    sequence_length = temporal_rgb_frames + 1
    setup_id = int(
        filename[filename.find('S') + 1:filename.find('S') + 4])
    camera_id = int(
        filename[filename.find('C') + 1:filename.find('C') + 4])
    subject_id = int(
        filename[filename.find('P') + 1:filename.find('P') + 4])
    duplicate_id = int(
        filename[filename.find('R') + 1:filename.find('R') + 4])
    action_id = int(
        filename[filename.find('A') + 1:filename.find('A') + 4])

    #if action_id > 60:
    #    return ''

    skeleton_file_name = filename_construct(setup_id, camera_id, subject_id, duplicate_id, action_id)
    if action_id < 61:
        frame_file = frame_path + skeleton_file_name
    else:
        frame_file = frame_path_120 + skeleton_file_name
    #print(frame_file)
    fivefs_concat=Image.new( 'RGB' , (128*temporal_rgb_frames,256) , (0,0,0) )
    if os.path.isdir(frame_file):# and action_id == 50:

        # load the frames' file name from folder
        frames = os.listdir(frame_file)

        start_i = 0
        # checked all len(frames) are  > 6
        sample_interval = len(frames) // sequence_length
        flip = False
        if sample_interval == 0:
            f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
            frame_range = f(temporal_rgb_frames, len(frames))
        else:
            if not evaluation:
                # Randomly choose sample interval and start frame
                start_i=0
                if random_interval:
                    #print('random_interval:::::::::::::',random_interval)
                    sample_interval = np.random.randint(1, len(frames) // sequence_length + 1)
                    start_i = np.random.randint(0, len(frames) - sample_interval * sequence_length + 1)
                #if random_roi_move:

                if random_flip:
                    flip = np.random.random() < 0.5

                # aline selection to the two sides
                frame_range = range(start_i, len(frames), sample_interval)

                #print(flip)
                #print(start_i, sample_interval)
            else:
                # Start at first frame and sample uniformly over sequence
                start_i = 0
                flip = False
                frame_range = range(start_i, len(frames), sample_interval)


        i=0
        for frame in frame_range:

            if frame != 0 and frame != (sequence_length*sample_interval):

                #print(frame)
                if not debug:
                    #openpose_file_, frame_file_ = openposeFile(frame_file, frame, skeleton_file_name, openpose_path)
                    frame_croped = ''
                    frame_ = frame
                    # find the closest non'' frame
                    while frame_croped == '':
                        if action_id < 61:
                            openpose_file_, frame_file_ = openposeFile(frame_file, frame_, skeleton_file_name, openpose_path)
                        else:
                            openpose_file_, frame_file_ = openposeFile(frame_file, frame_, skeleton_file_name, openpose_path_120)
                        # both openpose and RGB frame should exist
                        if os.path.isfile(openpose_file_) and os.path.isfile(frame_file_):
                            frame_croped = cropBody(openpose_file_, frame_file_, action_id, flip)
                            
                            #print('file consistent: ',openpose_file_)
                        else:
                            #print('file_unconsistent: ',openpose_file_, os.path.isfile(openpose_file_))
                            string = str(frame_file_ + " {}\n" + openpose_file_ + ' {}\n').format(os.path.isfile(frame_file_), os.path.isfile(openpose_file_))
                            with open('file_unconsistent_crop.txt', 'a') as fd:
                                fd.write(f'\n{string}')
                            

                        frame_ = frame_ + 1
                        if frame_ > len(frames):
                            frame_croped = Image.new( 'RGB' , (128,256) , (0,0,0) )
                            break

                    fivefs_concat.paste(frame_croped, (i*128+1,0))
                    i+=1
            '''
            plt.imshow(fivefs_concat)
            plt.suptitle('Corpped Body Parts')
            plt.show()
            '''
        # for generating st-roi

        if action_id < 61:
            frames_save = save_path + skeleton_file_name +'.png'
        else:
            frames_save = save_path_120 + skeleton_file_name +'.png'
        fivefs_concat.save(frames_save,"PNG")
        time.sleep(0.01)

    return fivefs_concat

if __name__ == '__main__':
    #'''
    file_list = []
    ignored_samples = []
    # folder = '/mnt/nas/ntu-rgbd/NTU120/3d_skeletons/'
    folder = '/data/xcl_data/nturgb+d_skeletons'
    ignored_sample_path = '/home/xcl/MMNet-main/data/ntu/samples_with_missing_skeletons.txt'

    for path in os.listdir(folder):
        file_list.append((folder, path))
    if ignored_sample_path is not None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]

    for folder, filename in sorted(file_list):
        if filename in ignored_samples:
            continue

        construct_st_roi(filename, evaluation=True, random_interval=False, random_flip=False)
    #'''
    #print(filename)
    #fivefs_concat = np.einsum('kli->lik',fivefs_concat.numpy())
    '''
    fivefs_concat = construct_st_roi('S003C002P019R001A001', evaluation=True,random_interval=False, random_flip=False)
    plt.imshow(fivefs_concat)
    plt.suptitle('Corpped Body Parts')
    plt.show()
    #'''
