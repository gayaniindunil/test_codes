import numpy as np
import sounddevice as sd
import soundfile as sf
import python_speech_features as psf
import keras
import queue


q = queue.Queue()

duration = 3.0
myrecording = sd.rec(int(duration * 16000), samplerate=16000, channels=1)
sd.play(myrecording)

#change default sd settings
sd.default.samplerate = 16000
sd.default.channels = 1


#read audio file
file_p = 'E:\\Neural Networks\\Audios\\News\\news.wav'
signal,rate = sf.read(file_p)

# get spech features
input_frame = signal[0:9600]
f =psf.logfbank(input_frame,samplerate=16000,winlen=0.03,winstep=0.01,preemph=0.97,nfilt=40)

f = np.expand_dims(f,2) #set the format to conv2D input
features = np.array(f)

