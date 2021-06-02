import plotly
import numpy as np
import pandas as pd
import librosa as lb
import streamlit as st
from scipy import signal
import plotly.express as px
import plotly.graph_objs as go
from plotly.graph_objs import *

class Visualize : 

    @staticmethod
    def read_audio(path, sr =16000):
        ''' It returns samples and sample rate from a given audio's path '''
        return lb.load(path, sr)


    @staticmethod
    def log_specgram(path= None,audio =None, sample_rate=16000, window_size=20, step_size=10, eps=1e-10):
        ''' It returns logarithm of spectrogram values from a given audio's path '''
        if (audio is not None): 
            audio = audio
        else :
            audio, sample_rate = Visualize.read_audio(path, sr = sample_rate)
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
    def plot_oscillogram(path= None,X_audio =None, Y_audio = None, sr =16000):
        ''' It plots wave a given audio's path '''
        if (X_audio is not None) and (Y_audio is not None): 
            x_s = X_audio
            audio = Y_audio
        else :
            audio, sample_rate = Visualize.read_audio(path, sr = sr)
            x_s= np.linspace(0, audio.shape[0]/sr, audio.shape[0])
        df_audio  = pd.DataFrame(
        {'time': x_s,
        'frequency': audio,
        })
        df_audio = df_audio.rename(columns={'time':'index'}).set_index('index')  
        # col1, col2 = st.beta_columns([3, 1])
        st.line_chart(df_audio, height=180)


    @staticmethod
    def plot_spectrogram(colours, path=None, audio= None, sr =16000):
        if (audio is not None): 
            freq,time,spec = Visualize.log_specgram(audio= audio, sample_rate= sr)
        else:
            freq,time,spec = Visualize.log_specgram(path= path, sample_rate= sr)
        trace = {
            "type" : "heatmap", 
            "x" : time,
            "y" : freq,
            "z" : spec.T,
            "colorscale" : colours

        }
        data = Data([trace])
        layout = {
            "xaxis" : {"title" : {"text": "Time"}},
            "yaxis" : {"title" : {"text": "Frequency"}}
        }

        fig = Figure(data=data, layout=layout)
        st.plotly_chart(fig)

    @staticmethod
    def plot(path, type_of_plot) : 
        if type_of_plot == 'Spectrogram' :
            return Visualize.plot_spectrogram(path=path)
        elif type_of_plot == "Oscillogram" : 
            return Visualize.plot_oscillogram(path = path)
        else : 
            return st.write("The other plot")

    @staticmethod
    def plot_vad(X_audio, Y_audio, type_of_plot, colours= None):
        if type_of_plot == 'Spectrogram' : 
            # st.subheader('Spectrogram of original audio')
            Visualize.plot_spectrogram(colours, path='output.wav')
            return Visualize.plot_spectrogram(colours , audio=Y_audio)
        elif type_of_plot == "Oscillogram" : 
            # st.subheader('Oscillogram of original audio')
            Visualize.plot_oscillogram('output.wav')
            return Visualize.plot_oscillogram(X_audio=X_audio, Y_audio=Y_audio)
