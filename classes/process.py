
import cv2
import pickle
import numpy as np 
from os import stat
import librosa as lb
import tensorflow as tf 
from scipy import signal
from scipy.stats import skew
from scipy.fftpack import fft
from tensorflow.keras.models import load_model

# ['Adnane_Driouche', 'Benjamin_Netanyau', 'Jens_Stoltenberg', 'Julia_Gillard', 'Magaret_Tarcher', 'Nelson_Mandela']


SPEAKERS_DICT = {0: 'Driouche Adnane', 1:'Jens Stoltenberg' ,
                 2 : 'Julia Gillard', 3: 'Magaret Tarcher', 4: 'Nelson Mandela'}

class FFT_Process():

    @staticmethod
    def audio_to_fft_one(audio, sr) : 
    
        audio = tf.squeeze(audio, axis=-1)
        print(f'shape of audio {audio.shape}')
        fft = tf.signal.fft(
                tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
            )
        print(f'shape of fft-1 {fft.shape}')
        fft = np.reshape(fft, (sr, 1))
        print(f'shape of fft-2 {fft.shape}')
        fft = tf.expand_dims(fft, axis=-1)
        print(f'shape of fft-2 {fft.shape}')
        return tf.math.abs(fft[ : (audio.shape[0] // 2), : , :])

    @staticmethod
    def path_to_audio(path, sr =16000):
        """Reads and decodes an audio file."""
        audio = tf.io.read_file(path)
        audio, _ = tf.audio.decode_wav(audio, 1, sr)
        return audio

    @staticmethod 
    def preprocess(path, sr=16000):
        audio_test = FFT_Process.path_to_audio(path)
        fft_test = FFT_Process.audio_to_fft_one(audio_test, sr=sr)
        fft_test = np.reshape(fft_test,(-1,8000,1))
        return fft_test

    @staticmethod
    def get_prediction(path, sr = 16000):
        model = load_model('Models\DL_models\SR_FFT_RavdessMe.h5')
        input_test = FFT_Process.preprocess(path, sr =sr)
        prediction = model.predict(input_test)
        perc_pred = max(prediction[0])
        id_speakers = prediction.argmax(axis = 1)

        return perc_pred, SPEAKERS_DICT[id_speakers[0]]


class CNN_Process():
    @staticmethod
    def read_audio(path, sr =16000):
        ''' It returns samples and sample rate from a given audio's path '''
        return lb.load(path, sr)

    @staticmethod
    def log_specgram(path, sample_rate=16000, window_size=20, step_size=10, eps=1e-10):
        ''' It returns logarithm of spectrogram values from a given audio's path '''
        audio, sample_rate = CNN_Process.read_audio(path, sr = sample_rate)
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        freqs, times, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
        return freqs, times, np.log(spec.T.astype(np.float32) + eps)

    @staticmethod 
    def preprocess():
        # get spectrogram as grayscale
        _,_,img_test = CNN_Process.log_specgram('output.wav')
        print(img_test.shape)   
        # resize image
        resized_test = cv2.resize(img_test.T, (32,32), interpolation = cv2.INTER_AREA)
        print(resized_test)
        # To gray scale
        gray_test = cv2.cvtColor(resized_test, cv2.COLOR_BGR2GRAY)

        gray_test = np.reshape(gray_test,(-1,32,32,1))
        return gray_test

    @staticmethod
    def get_prediction():
        model = load_model('Models\DL_models\sr_ravdess&me_cnn.h5')
        input_test = CNN_Process.preprocess()
        prediction = model.predict(input_test)
        perc_pred = max(prediction[0])
        id_speakers = prediction.argmax(axis = 1)

        return perc_pred, SPEAKERS_DICT[id_speakers[0]]  
 

class SVM_Process():
    
    @staticmethod
    def read_audio(path, sr =16000):
        ''' '''
        return lb.load(path, sr)

    @staticmethod  
    def extract_MFCC(samples, sr = 16000, n_mfcc=13): 
        ''' '''
        mfccs = lb.feature.mfcc(y=samples, sr=sr, n_mfcc= n_mfcc)
        return mfccs
    @staticmethod
    def get_feature():
        data, sr = SVM_Process.read_audio('output.wav')
        # Extracting MFCC 
        data = SVM_Process.extract_MFCC(data) 
        data_trunc = np.hstack((np.mean(data, axis=1), np.std(data, axis=1),
                                skew(data, axis = 1), np.max(data, axis = 1),
                                np.median(data, axis = 1), np.min(data, axis = 1)))
        data_trunc = np.reshape(data_trunc, (1, -1))
        return data_trunc

    @staticmethod
    def standarize_audio(data):
        file = open("Models\ML_models\scaler.pkl", 'rb')
        scaler = pickle.load(file)
        file.close()
        return scaler.transform(data)
        
    @staticmethod
    def process_svm(): 
        data = SVM_Process.get_feature()
        scaled_data = SVM_Process.standarize_audio(data)
        return scaled_data

    @staticmethod
    def predict_svm():
        file = open("Models\ML_models\sr_svm.pkl", 'rb')
        model = pickle.load(file)
        file.close()
        x_test = SVM_Process.process_svm()
        id_speaker, percentage = model.predict(x_test), max(model.predict_proba(x_test)[0])
        return percentage, id_speaker



class GNB_Process():

    @staticmethod
    def predict_gnb():
        file = open("Models\ML_models\sr_gnb.pkl", 'rb')
        model = pickle.load(file)
        file.close()
        x_test = SVM_Process.process_svm()
        id_speaker, percentage = model.predict(x_test), max(model.predict_proba(x_test)[0])
        return percentage, id_speaker

