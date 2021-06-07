from classes.visualize import Visualize
from classes.reduce_noise import Handling_Noise
from matplotlib.figure import Figure
from plotly.graph_objs import Data
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st 

class test_page : 
    @staticmethod
    def intro():
        description = f"""

        # TESTING CODE PAGE

        ---
        """
        st.write(description, unsafe_allow_html=True)

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
        sns.lineplot(x=frequency[:int(sr/4)], y=power_spectrum[:int(sr/4)], ax = ax)
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
    def show():
        test_page.intro()
        st.write('')
        row2_1, row2_2= st.beta_columns(2)
        with row2_1 : 
            st.subheader('Oscillogram of Audio')
            test_page.oscillogram()
        with row2_2: 
            st.subheader('Spectrogram of Audio')
            test_page.spectrogram()
        
        st.subheader('Spectrum of Audio')
        test_page.spectrum()



            