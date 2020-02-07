# sudo pip3 install pysoundfile
# sudo pip3 install python_speech_features --upgrade
# sudo apt-get install python3-pyaudio

import os
import numpy as np
import threading
import pyaudio
import wave
import soundfile as sf
import time
import librosa
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import keras
from keras import backend as K
from keras.models import Sequential,model_from_json
from keras.layers import Conv2D,Conv1D,MaxPooling1D,GlobalAveragePooling1D,GlobalMaxPooling1D
from keras.layers import MaxPooling2D
from keras.layers import Flatten,Dropout
from keras import optimizers, callbacks
import numpy as np
from keras.layers import Dense,Activation
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator,img_to_array
from python_speech_features import logfbank
from scipy.signal import butter,lfilter,freqz

fs = 44100


with open('ncnn1.json', 'r') as f:
		mymodel=model_from_json(f.read())

mymodel.load_weights("ncnn.h5")
print("loading success")

def butter_lowpass(cutoff,fs,order=5):
    nyq=0.5*fs
    normal_cutoff=cutoff/nyq
    b,a=butter(order,normal_cutoff,btype='low',analog=False)
    return b,a
def butter_lowpass_filter(data,cutoff,fs,order=5):
    b,a=butter_lowpass(cutoff,fs,order=order)
    y=lfilter(b,a,data)
    return y
def feature(soundfile):
    s,r=sf.read(soundfile)
    s=butter_lowpass_filter(s,11025,44100,order=3)
    x=np.array_split(s,32)
    
    logg=[]
    for i in x:
             
        xx=np.mean(logfbank(i,r,nfilt=40,nfft=1103),axis=0)
        logg.append(xx)
    print(logg)    
    return  logg  

def doafter5():
   
    feats = feature(r'C:\Users\alimi\Downloads\baby\2.ogg')
    d=np.zeros((64,40))
    for i in range(len(feats)):
        d[i:,]=feats[i]
    x=np.expand_dims(d,axis=0)
    soundclass = int(mymodel.predict_classes(x))

    print("Detecting....")
    print(soundclass)
    if soundclass==1:
       print("yes")


if __name__ == '__main__':
    print('Detecting......')
    doafter5()
