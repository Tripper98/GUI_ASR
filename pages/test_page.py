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
        sns.lineplot(x=frequency[:1000], y=power_spectrum[:1000], ax = ax)
        # sns.kdeplot(data=df['air_yards'], color='#CCCCCC',
        #                 fill=True, label='NFL Average',ax=ax)
        # sns.kdeplot(data=receiver['air_yards'], color=COLORS.get(team),
        #                 fill=True, label=player,ax=ax)
        ax.legend()
        ax.set_xlabel('Air Yards', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.grid(zorder=0,alpha=.2)
        # ax.set_axisbelow(True)
        # ax.set_ylim([0,16e4])
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
        

        # receiver=df.loc[(df.receiver==player) & (df.posteam==team) &
        #                 (df.week>= start_week) & (df.week<= stop_week)]

        fig2 = Figure()
        ax2 = fig2.subplots()
        sns.lineplot(data=df_audio, x="time", y="frequency", ax = ax2, color='red', linewidth=0.2)
        # sns.kdeplot(data=df['air_yards'], color='#CCCCCC',
        #                 fill=True, label='NFL Average',ax=ax)
        # sns.kdeplot(data=receiver['air_yards'], color=COLORS.get(team),
        #                 fill=True, label=player,ax=ax)
        ax2.legend()
        ax2.set_xlabel('Air Yards', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.grid(zorder=0,alpha=.2)
        # ax.set_axisbelow(True)
        # ax.set_xlim([-10,55])
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
            st.subheader('Air Yards Distribution')
            test_page.oscillogram()
        with row2_2: 
            st.subheader('Air Yards Distribution')
            test_page.spectrogram()
        
        st.subheader('Air Yards Distribution')
        test_page.spectrum()



            