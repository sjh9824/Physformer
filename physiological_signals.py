"""
'Predicted PPG 와 ground truth PPG 출력'

########################################################################
1. Pretrained weight load : 사전에 학습된 모델 가중치를 로드
2. Predicted PPG : 얼굴 영상을 입력으로 하여 모델 (MTTS-CAN) 로 출력
3. Ground truth PPG : predicted PPG 와 비교를 위한 reference PPG
4. Filterling: predicted PPG와 ground truth PPG의 노이즈를 제거 
"""

import tensorflow as tf
import pickle
import torch
import numpy as np
import scipy.io
import os
import sys
sys.path.append('../')
from mt_model import Attention_mask, MTTS_CAN
from model import ViT_ST_ST_Compact3_TDC_gra_sharp
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter
from inference_preprocess import preprocess_raw_video, preprocess_wave, detrend

def predicted_and_gt(video_path, bvp_path):
    img_rows = 36       # 이미지 (프레임)의 width
    img_cols = 36       # 이미지 (프레임)의 height
    frame_depth = 10    # 연속적인 프레임의 개수 
    model_checkpoint = './Weights/Physformer.pkl' # 사전 학습된 모델의 가중치 경로S
    batch_size = 100    # 배치 사이즈 
    fs = 30             # 샘플링 속도
    sample_data_path = video_path # 얼굴 영상의 경로
    
    # 얼굴 영상 로드 
    dXsub = preprocess_raw_video(sample_data_path, dim=36)
    print('dXsub shape: ', dXsub.shape)

    dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]
    
    model = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(160,128,128), patches=(4,4,4), dim=96, ff_dim=144, num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
    
    # 모델 설정 및 가중치 로드
    #model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    #model = torch.load(model_checkpoint)
    with open('Physformer.pkl','rb') as f:
        weights = pickle.load(f)
    
    model.load_state_dict(weights)

    # 모델로부터 PPG 추정 (Input 은 얼굴 영상)
    input = torch.tensor((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)
    
    model.eval()
    with torch.no_grad():
        output= model(input)
    
    
    pred_ppg = output.item()
    # Ground truth PPG 로드
    gt_ppg = preprocess_wave(bvp_path)
    gt_ppg_len = (gt_ppg.shape[0] // frame_depth) * frame_depth
    gt_ppg = gt_ppg[:gt_ppg_len]
    gt_ppg = gt_ppg[:,np.newaxis]
    
    # Filter predicted PPG
    pulse_pred = pred_ppg[0]
    pulse_pred = detrend(np.cumsum(pulse_pred), 100)
    [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))
    print('Predicte PPG shape: ', pulse_pred.shape)
    
    # Filter ground truth PPG
    gt_ppg = detrend(np.cumsum(gt_ppg), 100)
    [b_gt_pulse, a_gt_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    gt_ppg = scipy.signal.filtfilt(b_gt_pulse, a_gt_pulse, np.double(gt_ppg))
    print('Ground truth PPG shape: ', gt_ppg.shape)
    
    return pulse_pred, gt_ppg
