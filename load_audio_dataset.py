import numpy as np
import os
import python_speech_features as psf 
import soundfile as sf 
import sounddevice as sd 

datafilepath = 'data'
featurs_dir = 'features'

def getdatafiles(path):
    list_dir = os.listdir(path)
    list_dir_path = []

    for f in list_dir:
        filedir = os.path.join(path,f)
        list_dir_path.append(filedir)

    if(len(list_dir) == 0):
        print('No data to train the model')
    return list_dir_path

def setfilepath(filename):
    cwd = os.getcwd()
    filepath = os.path.join(cwd,filename)
    return filepath

def readaudiofile(filepath):
    signal,_= sf.read(filepath)
    if len(signal.shape)>1:
        signal = signal[:,0].reshape(signal.shape[0],1)
    frame_len = int(0.6*16000) #9600 600ms parts 
    frame_step = int(0.1*16000) # 1600 100ms overlapping 
    
    if len(signal)<frame_len:
        zeros = np.zeros(frame_len - len(signal))
        signal = np.append(signal,zeros)

    # if len(signal) >= frame_len:
    num_aud_frames = 1 + int(np.ceil((len(signal)-frame_len)/frame_step))
    features = np.zeros([num_aud_frames,58,40,1])

    startval = 0
    endval = (startval+frame_len)
    for i in range(num_aud_frames):
        aud = signal[startval:endval]
        if len(aud)<frame_len:
            zeros = np.zeros(frame_len - len(aud))
            aud = np.append(aud,zeros)

        f =psf.logfbank(aud,samplerate=16000,winlen=0.03,winstep=0.01,preemph=0.97,nfilt=40)
        f = np.expand_dims(f,2)
        features[i]= f
        startval += frame_len - frame_step
        endval = (startval+frame_len)

    features = np.array(features)
    return features 

def saveFeatures(savepath):
    datafilepath = 'data\\news'
    # path = 'E:\\GAYANI\\hotWorddetection_gayani\\dataset\\Sam'
    files = getdatafiles(datafilepath)
    featureMap = []
    for e in files:
        filepath = setfilepath(e)
        feature = readaudiofile(filepath)
        for f in range(len(feature)):
            featureMap.append(feature[f])
    featureMap = np.array(featureMap)
    np.save(savepath,featureMap)

    print(featureMap.shape)

featurs_dir = 'features\\news.npy'
saveFeatures(featurs_dir)



