#Usage: python app.py
import os
 
from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
import argparse
import time
import uuid
import base64
import os
import numpy as np
import threading
import pyaudio
import requests
import wave
import soundfile as sf
import time

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


UPLOAD_FOLDER = 'uploads'

fs = 44100
url = "https://www.fast2sms.com/dev/bulk"
headers = {
 'authorization': "vijZsnUAxX1NmTElCFfu0QWwa8zd3pbSR6I2VOHPrtyqG9koL7LZoiYp01D2EWc8vklwOPIzUqfmgN6A",
 'Content-Type': "application/x-www-form-urlencoded",
 'Cache-Control': "no-cache",
 }




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
    print(soundfile)
    s,r=sf.read(soundfile)
    s=butter_lowpass_filter(s,11025,44100,order=3)
    x=np.array_split(s,32)
    
    logg=[]
    for i in x:
             
        xx=np.mean(logfbank(i,r,nfilt=40,nfft=1103),axis=0)
        logg.append(xx)
    print(logg)    
    return  logg  

def predict1(file):
    print("predict1")
    with open('ncnn1.json', 'r') as f:
        mymodel=model_from_json(f.read())

    mymodel.load_weights("ncnn.h5")
    print("loading success")
    feats = feature(file)
    d=np.zeros((64,40))
    for i in range(len(feats)):
        d[i:,]=feats[i]
    x=np.expand_dims(d,axis=0)
    soundclass = int(mymodel.predict_classes(x))

    print("Detecting....")
    print(soundclass)
    if soundclass==1:
       print("yes")

    return soundclass
		



def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../uploads/template.jpg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("Upload")
        import time
        start_time = time.time()
        file = request.files['file']
        print(file.filename)
     
        filename = secure_filename(file.filename)

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(file_path)
        result = predict1(file_path)
        if result==1:
            msg="Baby needs immediate attention"		
        if result==0:
            msg="NO"    
        print(file_path)
        filename = my_random_string(6) + filename

        os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print("--- %s seconds ---" % str (time.time() - start_time))
        payload = "sender_id=FSTSMS&message="+msg+"&language=english&route=p&numbers=9963611235"
        response = requests.request("POST", url, data=payload, headers=headers)
 
        print(response.text)
        return render_template('template.html',label=msg )
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=False
    app.run(host='0.0.0.0', port=3000)
