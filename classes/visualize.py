import plotly
import numpy as np
import pandas as pd
import librosa as lb
import seaborn as sns
import streamlit as st
from scipy import signal
import plotly.express as px
import plotly.graph_objs as go
from plotly.graph_objs import Data
from plotly.graph_objs import Figure as fg
from matplotlib.figure import Figure

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

        fig = fg(data=data, layout=layout)
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

    @staticmethod
    def spectrum():
        '''
        This function returns an ay dist for the desired wr
        '''

        audio, sr = Visualize.read_audio('output.wav', sr = 16000)
        x_s= np.linspace(0, audio.shape[0]/sr, audio.shape[0])
        df_audio  = pd.DataFrame(
        {'time': x_s,
        'frequency': audio,
        })

        fourier_transform = np.fft.rfft(audio)

        abs_fourier_transform = np.abs(fourier_transform)

        power_spectrum = np.square(abs_fourier_transform)

        frequency = np.linspace(0, sr/2, len(power_spectrum))

        fig1 = Figure()
        ax = fig1.subplots()
        sns.lineplot(x=frequency[:int(sr/4)], y=power_spectrum[:int(sr/4)], ax = ax, linewidth=0.2)
        ax.legend()
        ax.set_ylabel('Power', fontsize=12)
        ax.set_xlabel('Frequency', fontsize=12)
        ax.grid(zorder=0,alpha=.2)
        st.pyplot(fig1) 

    @staticmethod
    def oscillogram():
        '''
        This function returns an ay dist for the desired wr
        '''

        audio, sr = Visualize.read_audio('output.wav', sr = 16000)
        x_s= np.linspace(0, audio.shape[0]/sr, audio.shape[0])
        df_audio  = pd.DataFrame(
        {'time': x_s,
        'frequency': audio,
        })

        fig2 = Figure()
        ax2 = fig2.subplots()
        sns.lineplot(data=df_audio, x="time", y="frequency", ax = ax2, color='red', linewidth=0.2)
        ax2.set_ylabel('Amplitude', fontsize=12)
        ax2.set_xlabel('Seconds', fontsize=12)
        ax2.grid(zorder=0,alpha=.2)
        st.pyplot(fig2) 

    @staticmethod 
    def spectrogram():
        audio, sr = Visualize.read_audio('output.wav', sr = 16000)
        freqs,times,spec = Visualize.log_specgram(audio= audio, sample_rate= sr)

        fig3 = Figure()
        ax3 = fig3.subplots()
        ax3.imshow(spec.T, aspect='auto', origin='lower', 
                extent=[times.min(), times.max(), freqs.min(), freqs.max()])
        ax3.set_ylabel('Freqs in Hz', fontsize=12)
        ax3.set_xlabel('Seconds', fontsize=12)
        st.pyplot(fig3)

    @staticmethod
    def acoustic_char():
        st.write(' ')
        st.write('# Acoustic Characteristics')
        row2_1, row2_2= st.beta_columns(2)
        with row2_1 : 
            st.subheader('Oscillogram of Audio')
            Visualize.oscillogram()
        with row2_2: 
            st.subheader('Spectrogram of Audio')
            Visualize.spectrogram()
        
        st.subheader('Spectrum of Audio')
        Visualize.spectrum()
