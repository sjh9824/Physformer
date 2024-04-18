from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import numpy as np
import random
import torch
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
from VIPLHR_DataLoader import VIPLHR_DataLoader



clip_frames = 160   



def adjust_shape(arr1, arr2):
    # 각 튜플에서 각 차원의 크기를 가져옵니다.
    shape1 = arr1
    shape2 = arr2
    
    if shape1 is None:
        shape1_len = 0
    else :
        shape1_len = len(shape1)
        
    if shape2 is None:
        shape2_len = 0
    else :
        shape2_len = len(shape2)

    # 각 차원의 크기를 비교하여 더 큰 차원의 크기로 작은 튜플의 해당 차원을 확장합니다.
    for i in range(max(shape1_len, shape2_len)):
        if len(shape1) <= i:
            shape1 = shape1 + (1,)  # 작은 튜플에 차원 추가
        if len(shape2) <= i:
            shape2 = shape2 + (1,)  # 작은 튜플에 차원 추가
            
    return shape1, shape2



class Normaliztion (object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        video_x, clip_average_HR, ecg_label, frame_rate = sample['video_x'],sample['clip_average_HR'], sample['ecg'],sample['frame_rate']
        new_video_x = (video_x - 127.5)/128
        return {'video_x': new_video_x, 'clip_average_HR':clip_average_HR, 'ecg':ecg_label, 'frame_rate':frame_rate}




class RandomHorizontalFlip (object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        video_x, clip_average_HR, ecg_label, frame_rate = sample['video_x'],sample['clip_average_HR'], sample['ecg'],sample['frame_rate']

        h, w = video_x.shape[1], video_x.shape[2]
        new_video_x = np.zeros((clip_frames, h, w, 3))


        p = random.random()
        if p < 0.5:
            #print('Flip')
            for i in range(clip_frames):
                # video 
                image = video_x[i, :, :, :]
                image = cv2.flip(image, 1)
                new_video_x[i, :, :, :] = image
                
                
                
            return {'video_x': new_video_x, 'clip_average_HR':clip_average_HR, 'ecg':ecg_label, 'frame_rate':frame_rate}
        else:
            #print('no Flip')
            return {'video_x': video_x, 'clip_average_HR':clip_average_HR, 'ecg':ecg_label, 'frame_rate':frame_rate}



class ToTensor (object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        video_x, clip_average_HR, ecg_label, frame_rate = sample['video_x'],sample['clip_average_HR'], sample['ecg'],sample['frame_rate']

        # swap color axis because
        # numpy image: (batch_size) x depth x H x W x C
        # torch image: (batch_size) x C x depth X H X W
        video_x = video_x.transpose((3, 0, 1, 2))
        video_x = np.array(video_x)
        
        clip_average_HR = np.array(clip_average_HR)
        
        frame_rate = np.array(frame_rate)
        
        
        return {'video_x': torch.from_numpy(video_x.astype(float)).float(), 'clip_average_HR': torch.from_numpy(clip_average_HR.astype(float)).float(), 'ecg': torch.from_numpy(ecg_label.astype(float)).float(), 'frame_rate': torch.from_numpy(frame_rate.astype(float)).float()}



# train
class VIPL_train (Dataset):

    def __init__(self, info_list, root_dir, transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):
        print('루트' + self.landmarks_frame.iloc[idx, 0])
        video_path = os.path.join(self.root_dir, str(self.landmarks_frame.iloc[idx, 0]),"frames/")
        
        start_frame = self.landmarks_frame.iloc[idx, 1]
        
        frame_rate  = self.landmarks_frame.iloc[idx, 2]
        clip_average_HR  = self.landmarks_frame.iloc[idx, 3]
        
        print(video_path)
        
        p = random.random()
        if p < 0.5 and start_frame==61:    # sampling aug     p < 0.5
            ecg_label = self.landmarks_frame.iloc[idx, 5:5+160].values 
            video_x, clip_average_HR, ecg_label_new = self.get_single_video_x_aug(video_path, start_frame, clip_average_HR, ecg_label)
            
        else: 
            video_x = self.get_single_video_x(video_path, start_frame)
            ecg_label_new = self.landmarks_frame.iloc[idx, 5:5+160].values 
 
        
        sample = {'video_x': video_x, 'frame_rate':frame_rate, 'ecg':ecg_label_new, 'clip_average_HR':clip_average_HR}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_video_x(self, video_path, start_frame):
        video_jpgs_path = video_path

        video_x = np.zeros((clip_frames, 128, 128, 3))
        
        size_crop = np.random.randint(16)
        
        log_file = open('temp.txt', 'w')
        
        image_id = start_frame

        for i in range(clip_frames):
            s = "%04d" % (image_id - 1)
            image_name = 'frame_' + s + '.png'

            # face video 
            image_path = os.path.join(video_jpgs_path, image_name)


        for i in range(clip_frames):
            s = "%04d" % (image_id - 1)
            image_name = 'frame_' + s + '.png'

            # face video 
            image_path = os.path.join(video_jpgs_path, image_name)
            
            
            log_file.write(image_path)
            log_file.write("\n")
            log_file.flush()
                
            
            tmp_image = cv2.imread(image_path)
            #cv2.imwrite('test111.jpg', tmp_image)
            
            if tmp_image is None:    # It seems some frames missing 
                tmp_image = cv2.imread("/media/neuroai/T7 Shield/VIPL_HR/PhysFormer-main/p10/v1/source1/frames/frame_0722.png")
            
            
            tmp_image = cv2.resize(tmp_image, (128+size_crop, 128+size_crop), interpolation=cv2.INTER_CUBIC)[(size_crop//2):(128+size_crop//2), (size_crop//2):(128+size_crop//2), :]
            
            video_x[i, :, :, :] = tmp_image  

                        
            image_id += 1
   
        return video_x
    
   
    def get_single_video_x_aug(self, video_path, start_frame, clip_average_HR, ecg_label):
        video_jpgs_path = video_path

        video_x = np.zeros((clip_frames, 128, 128, 3))
        ecg_label_new = np.zeros(clip_frames)
        
        
        size_crop = np.random.randint(16)
        
        if clip_average_HR>88:  #  halve
            clip_average_HR = clip_average_HR/2
            for tt in range(clip_frames):
                if tt%2 == 0:
                    image_id = start_frame + tt//2
                    s = "%04d" % (image_id - 1)
                    image_name = 'frame_' + s + '.png'
                    image_path = os.path.join(video_jpgs_path, image_name)
                    tmp_image = cv2.imread(image_path)
                    if tmp_image is None:    # It seems some frames missing 
                        tmp_image = cv2.imread("/media/neuroai/T7 Shield/VIPL_HR/PhysFormer-main/p10/v1/source1/frames/frame_0722.png")
                    ecg_label_new[tt] = ecg_label[tt//2]
                    
                
                else:
                    try:
                        image_id1 = start_frame + tt//2
                        image_id2 = image_id1+1
                        s = "%04d" % (image_id - 1)
                        image_name = 'frame_' + s + '.png'
                        image_path = os.path.join(video_jpgs_path, image_name)
                        tmp_image1 = cv2.imread(image_path)
                        s = "%04d" % (image_id - 1)
                        image_name = 'frame_' + s + '.png'
                        image_path = os.path.join(video_jpgs_path, image_name)
                        tmp_image2 = cv2.imread(image_path)
                        if tmp_image1 is None:    # It seems some frames missing 
                            tmp_image1 = cv2.imread("/media/neuroai/T7 Shield/VIPL_HR/PhysFormer-main/p10/v1/source1/frames/frame_0722.png")
                        if tmp_image2 is None:    # It seems some frames missing 
                            tmp_image2 = cv2.imread("/media/neuroai/T7 Shield/VIPL_HR/PhysFormer-main/p10/v1/source1/frames/frame_0722.png")

                        tmp_image1,tmp_image2 = adjust_shape(tmp_image1,tmp_image2)
                        
                        tmp_image = tmp_image1//2+tmp_image2//2    # mean linear interpolation
                        ecg_label_new[tt] = ecg_label[tt//2]/2+ecg_label[tt//2+1]/2
                        
                    except:
                        image_id = start_frame + tt//2
                        s = "%04d" % (image_id - 1)
                        image_name = 'frame_' + s + '.png'
                        image_path = os.path.join(video_jpgs_path, image_name)
                        tmp_image = cv2.imread(image_path)
                        if tmp_image is None:    # It seems some frames missing 
                            tmp_image = cv2.imread("/media/neuroai/T7 Shield/VIPL_HR/PhysFormer-main/p10/v1/source1/frames/frame_0722.png")
                        ecg_label_new[tt] = ecg_label[tt//2]
                    
                
                
                tmp_image = cv2.resize(tmp_image, (128+size_crop, 128+size_crop), interpolation=cv2.INTER_CUBIC)[(size_crop//2):(128+size_crop//2), (size_crop//2):(128+size_crop//2), :]
                
                image_x_aug = tmp_image
                video_x[tt, :, :, :] = image_x_aug  
            
                    
        else:     # double
            clip_average_HR = clip_average_HR*2
            for tt in range(clip_frames):
                image_id = start_frame + tt*2
                s = "%04d" % (image_id - 1)
                image_name = 'frame_' + s + '.png'
                image_path = os.path.join(video_jpgs_path, image_name)
                tmp_image = cv2.imread(image_path)
                if tmp_image is None:    # It seems some frames missing 
                    tmp_image = cv2.imread("/media/neuroai/T7 Shield/VIPL_HR/PhysFormer-main/p10/v1/source1/frames/frame_0722.png")
                
                tmp_image = cv2.resize(tmp_image, (128+size_crop, 128+size_crop), interpolation=cv2.INTER_CUBIC)[(size_crop//2):(128+size_crop//2), (size_crop//2):(128+size_crop//2), :]
                
                image_x_aug = tmp_image
                video_x[tt, :, :, :] = image_x_aug  
                
                # approximation
                if tt<80:
                    ecg_label_new[tt] = ecg_label[tt*2]
                else:
                    ecg_label_new[tt] = ecg_label_new[tt-80]
            
        
   
        return video_x, clip_average_HR, ecg_label_new
    
    


