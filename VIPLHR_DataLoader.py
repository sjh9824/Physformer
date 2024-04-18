#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import torch
import cv2

class VIPLHR_DataLoader():
    
    def __init__(self, 
                 video_path, 
                 wave_path=None,
                 gtHR_path=None,
                 SpO2_path=None,
                 timestamp_path=None):
        
        self.video_path = video_path        
        self.wave_path = wave_path
        self.gtHR_path = gtHR_path
        self.SpO2_path = SpO2_path        
        self.timestamp_path = timestamp_path

    
    #% 'video_path' 에서 얼굴 영상을 로드 %#
    def read_video(self):
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print('fps', fps)
        
        frames = []  # 로드한 비디오 프레임들이 저장되는 빈 리스트 정의
        
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
  
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 프레임을 RGB color로 변환
            frame = cv2.resize(frame, (128, 128))  # 프레임의 크기를 (128, 128) 로 조정
            frames.append(frame)
            
        cap.release()
        
        video_seq =  np.array(frames, dtype=np.float32)  
        
        return video_seq
    
    
    #% 'wave_path' 에서 ground truth PPG 로드 %#    
    def read_timeseries(self, path):
        
        if path is not None:
            x = pd.read_csv(path, header=None)
            x = x.iloc[1:]
            x = x.to_numpy().squeeze(1)
        
        return x
    
    
    def record_video_timestamp(self):
        
        video_data = self.read_video()
        video_timestamp = pd.read_csv(self.timestamp_path)
        
        video_timestamp = video_timestamp.to_numpy().squeeze(1)
        
        video_df = pd.DataFrame({'timestamp': video_timestamp, 'data': video_data})
        
        return video_df
    
    
    def record_wave_timestamp(self):
        
        wave_data = self.read_timeseries(self.wave_path)
        
        wave_len = len(wave_data)
        HR_len = len(self.read_timeseries(self.gtHR_path))
        
        wave_period = HR_len / wave_len
        
        wave_timestamp = np.arange(0, wave_period * wave_len, wave_period)
        
        wave_df = pd.DataFrame({'timestamp': wave_timestamp, 'data': wave_data})
        
        return wave_df
             
    
    #% 얼굴 영상 & ground truth PPG 동기화 %#    
    def Data_Synchronization(self):
        
        video_time_df = pd.read_csv(self.timestamp_path, names=['timestamp'])
        video_time_df['timestamp'] = video_time_df['timestamp'] / 1000.0
        
        wave_df = self.record_wave_timestamp()
        
        wave_sync_df = pd.merge_asof(video_time_df, wave_df, on='timestamp', direction='nearest')
        
        return wave_sync_df

