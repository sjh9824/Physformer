"""
- Heart rate (HR) 를 추정 : calculate_HR
- Predicted PPG와 Ground truth PPG의 heart rate 오차 측정 : calculate_HR_metrics
 * MAE (Mean absolute error) : 평균 절대값 오차
 * RMSE (Root mean square error) : 평균 제곱근 오차
"""


import numpy as np
import scipy
from scipy.sparse import spdiags

def _next_power_of_2(x):
    """2의 가장 가까운 거듭제곱을 계산"""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _calculate_fft_hr(ppg_signal, fs, low_pass=0.75, high_pass=2.5):
    """Fast Fourier transform (FFT) 을 사용하여 PPG 로 부터 HR 계산"""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


def calculate_metric_per_video(predictions, labels, fs=30):
    """Predicted PPG & ground truth PPG 의 HR 계산"""  
  
    hr_pred = _calculate_fft_hr(predictions, fs=fs)
    hr_label = _calculate_fft_hr(labels, fs=fs)
    
    return hr_label, hr_pred


def calculate_HR(prediction, label):
    """Predicted PPG & ground truth PPG 의 HR 계산"""
    
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()   
    
    # '100개' 의 연속적인 비디오 프레임으로 부터 heart rate 를 계산
    window_frame_size = 100

    for i in range(0, len(prediction) // window_frame_size):
        pred_window = prediction[i * window_frame_size: (i+1) * window_frame_size]
        label_window = label[i * window_frame_size: (i+1) * window_frame_size]        
           
        gt_hr_fft, pred_hr_fft= calculate_metric_per_video(
            pred_window, label_window, fs=30)
        gt_hr_fft_all.append(gt_hr_fft)
        predict_hr_fft_all.append(pred_hr_fft)       
    
    return gt_hr_fft_all, predict_hr_fft_all


def calculate_HR_metrics(pred_hr_np, gt_hr_np):
    """ HR 오차를 계산하는 지표들 
     *Parameters*
      pred_hr_np : predicted PPG 로 부터 추정된 HR (numpy)
      gt_hr_np : ground truth PPG 로 부터 추정된 HR (numpy) 
    """
    
    gt_hr_fft_all = gt_hr_np
    predict_hr_fft_all = pred_hr_np 

    num_test_samples = len(predict_hr_fft_all)
    
    # Heart rate MAE 
    HR_MAE = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
    standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
    print("HR MAE : Mean {0} +/- Std {1}".format(HR_MAE, standard_error))
    
    # Heart rate RMSE 
    HR_RMSE = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
    standard_error = np.std(np.square(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
    print("HR RMSE : Mean {0} +/- Std {1}".format(HR_RMSE, standard_error))                