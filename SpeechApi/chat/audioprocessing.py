import pandas as pd 
from joblib import Parallel, delayed
import mutagen
from mutagen.wave import WAVE
import shutil
import os
from glob import glob
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, './inference_code/amplitude-modulation-analysis-module1/amplitude-modulation-analysis-module/')
from am_analysis import am_analysis as ama


import librosa
import numpy as np
import scipy
from scipy.signal import lfilter, hamming,resample
from scipy.signal.windows import hann
from scipy.io import wavfile
import numpy as np
import math
import librosa
import numpy as np
from scipy.io import wavfile
import numpy as np

def LPres(speech ,fs, framesize, frameshift,lporder, preemp):
    if (framesize>50):
        print("Warning!")
    else:
        # Converting unit of variuos lengths from 'time' to 'sample number'
        Nframesize	= round(framesize * fs / 1000)
        Nframeshift	= round(frameshift * fs / 1000)
        Nspeech 	= len(speech)

        #Transpose the 'speech' signal if it is not of the form 'N x 1'

        speech=speech.reshape(Nspeech,1)
        #speech = speech(:); % Make it a column vector
        #PREEMPHASIZING SPEECH SIGNAL
    if (preemp != 0):
        speech = preemphasize(speech)
        #COMPUTING RESIDUAL
    res = np.asarray(np.zeros((Nspeech,1)))[:,0]
    #NUMBER OF FRAMES
    lporder=int(lporder)
    nframes=math.floor((Nspeech-Nframesize)/Nframeshift)+1
    j = 1
    for i in range(0,Nspeech-Nframesize,Nframeshift):
        SpFrm	= speech[i:i+Nframesize]
        winHann =  np.asmatrix(hann(Nframesize))
        y_frame = np.asarray(np.multiply(winHann,SpFrm.T))
        #print(y_frame[0,:], lporder)
        lpcoef = librosa.lpc(y_frame[0,:], lporder)
        
        if(i <= lporder):
            PrevFrm=np.zeros((1,lporder))
        else:
            # print('i: ', i)
            PrevFrm=speech[(i-lporder):(i)]
        ResFrm = ResFilter_v2(np.real(PrevFrm),np.real(SpFrm),np.real(lpcoef),lporder,Nframesize,0)
        res[i:i+Nframeshift]	= ResFrm[:Nframeshift]
        j = j+1
    
    res[i+Nframeshift:i+Nframesize]	= ResFrm[Nframeshift:Nframesize]
    #PROCESSING LASTFRAME SAMPLES, 
    if(i < Nspeech):
        SpFrm= speech[i:Nspeech]
        winHann =  np.asmatrix(hamming(len(SpFrm)))
        y_frame = np.asarray(np.multiply(winHann,SpFrm.T))
        lpcoef	= librosa.lpc(y_frame[0,:],lporder)
        # print(lpcoef)
        PrevFrm	= speech[(i-lporder):(i)]
        ResFrm	= ResFilter_v2(np.real(PrevFrm),np.real(SpFrm),np.real(lpcoef),lporder,Nframesize,1)
        # print(ResFrm)
        res[i:i+len(ResFrm)]	= ResFrm[:len(ResFrm)]
        j = j+1
    hm = hamming(2*lporder)
    for i in range(1,round(len(hm)/2)):
        res[i]	= res[i] * hm[i]      #attenuating first lporder samples
    return res



def preemphasize(sig):
    # bcoefs=[-1,factor]
    # acoefs=1
    # y=lfilter(bcoefs,acoefs,sig)
    dspeech=np.diff(sig)
    dspeech[len(dspeech)+1]=dspeech(len(dspeech))
    return dspeech


def ResFilter_v2(PrevSpFrm,SpFrm,FrmLPC,LPorder,FrmSize,plotflag):
    # print('This is getting executed')
    ResFrm=np.asarray(np.zeros((1,FrmSize)))
    ResFrm=ResFrm[0,:]
    # print('b: ', (ResFrm))
    tempfrm=np.zeros((1,FrmSize+LPorder))
    # print(np.shape(tempfrm))
    # tempfrm[1:LPorder]=PrevSpFrm
    # #tempfrm(1:FrmSize)=PrevSpFrm(1:FrmSize);
    # tempfrm[LPorder+1:LPorder+FrmSize]=SpFrm[1:FrmSize]


    temp_PrevSpFrm=np.asmatrix(PrevSpFrm)
    temp_SpFrm=np.asmatrix(SpFrm[:FrmSize])
    if (np.shape(temp_PrevSpFrm)[0]==1):
        temp_PrevSpFrm=temp_PrevSpFrm.T
    if (np.shape(temp_SpFrm)[0]==1):
        temp_SpFrm=temp_SpFrm.T

    # print(np.shape(temp_PrevSpFrm))
    # print(np.shape(temp_SpFrm))
    tempfrm=np.concatenate((temp_PrevSpFrm, temp_SpFrm))
    tempfrm=np.asarray(tempfrm)[:,0]
    # print((tempfrm))
    # print(np.shape(tempfrm))


    for i in range(FrmSize):
        t=0
        for j in range(LPorder):
            # print(FrmLPC[j+1], tempfrm[-j+i+LPorder-1])
            # print(FrmLPC[j+1])
            t=t+FrmLPC[j+1]*tempfrm[-j+i+LPorder-1]

        ResFrm[i]=SpFrm[i]-(-t)

    return ResFrm

def excitation(sample,fs):
    # if fs!=8000:
    #   sample=resample(sample,8000)+0.00001
    #   fs=8000

    lporder=10
    residual=LPres(sample,fs,20,10,lporder,0)
    henv = np.abs(scipy.signal.hilbert(residual))
    resPhase=np.divide(residual, henv) 
    return residual, henv, resPhase


def load_wav(audio_filepath, sr, min_dur_sec=5):
    audio_data, fs = librosa.load(audio_filepath, sr=8000)
    len_file = len(audio_data)
    if len_file <= int(min_dur_sec * sr):
        temp = np.zeros((1, int(min_dur_sec * sr) - len_file))
        joined_wav = np.concatenate((audio_data, temp[0]))
    else:
        joined_wav = audio_data
        
    return audio_data,fs#joined_wav,fs


def modulation_spectogram_from_wav(audio_data,fs):
    x=audio_data
    x = x / np.max(x)
    residual, _, _ = excitation(x, fs)
    win_size_sec = 0.04 
    win_shft_sec = 0.01  
    stft_modulation_spectrogram = ama.strfft_modulation_spectrogram(residual, fs, win_size = round(win_size_sec*fs), win_shift = round(win_shft_sec*fs), channel_names = ['Modulation Spectrogram'])
    X_plot=ama.plot_modulation_spectrogram_data(stft_modulation_spectrogram, 0 , modf_range = np.array([0,20]), c_range =  np.array([-90, -50]))
    return X_plot

def load_data(filepath, sr=8000, min_dur_sec=5, win_length=160, hop_length=80, n_mels=40, spec_len=504):
    import wave
    audio_data,fs = load_wav(filepath, sr=sr, min_dur_sec=min_dur_sec)
#     raw = wave.open(filepath)
    linear_spect = modulation_spectogram_from_wav(audio_data,fs)
    # mag, _ = librosa.magphase(linear_spect)
    # mag = np.log1p(mag)
    mag_T = linear_spect
    shape = np.shape(mag_T)
    padded_array = np.zeros((161, 505))
    padded_array[:shape[0],:shape[1]] = mag_T[:, 0:505] 
    mag_T=padded_array
    randtime = np.random.randint(0, mag_T.shape[1] - spec_len)
    spec_mag = mag_T[:, randtime:randtime + spec_len]
  
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)


from torchvision.models import resnet34
import torch
import torch.nn as nn

def get_model(device, num_classes, pretrained=False):
    model = resnet34(pretrained=pretrained)
    model.fc = nn.Linear(512, num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2,2), padding=(3,3), bias=False)
    model.to(device, dtype=torch.float)
    return model

import os
import sys
import numpy as np
import torch
import yaml
import warnings
import os
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device ="cpu"

num_classes = 2
model = get_model(device, num_classes, pretrained=False)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'inference_code', '1.libri_lj', 'Checkpoints_libri', 'best_model', 'best_checkpoint.pt')
#print("***********",model_path)

checkpoints = torch.load(model_path,map_location=torch.device('cpu'))
model.load_state_dict(checkpoints['state_dict'])

model.eval()


def getPrediction(filepath):

    spec=load_data(filepath)[np.newaxis, ...]
    feats = np.asarray(spec)
    feats = torch.from_numpy(feats)
    feats = feats.unsqueeze(0)
    feats = feats.to(device)
    label = model(feats.float())
    _, pred=label.max(1)
    return pred.item()

#     filepath = "/home/rishi/workspace/codes/open_smile_2/DataSets/LibriTTS/combined/Test/fake_slt/84_121550_000268_000000_gen.wav"
# # spec = load_data(filepath)
# # print(spec.shape)
# result = getPrediction(model, filepath)
# print(result)

# 1 