import streamlit as st 
import plotly.express as px
from classes.visualize import Visualize

class Visualization : 
    @staticmethod
    def intro():
        page_icon = "https://user-images.githubusercontent.com/46791116/119490544-dd24a180-bd54-11eb-88dd-a9432f197615.png"
        description = f"""
        <div align='center'>
        <img src={page_icon}
        width="100" height="100">

        # VISUALIZATION
        </div>

        <details>

        <summary>
        <samp>What is a Spectrogram?</samp>
        </summary>

        Nanani nana

        </details>

        <details>

        <summary>
        <samp>What is an Oscillogram?</samp>
        </summary>

        Nanani nana

        </details>

        ---
        """
        st.write(description, unsafe_allow_html=True)

    @staticmethod
    def show(type_of_plot, colours = None):
        Visualization.intro()
        if type_of_plot == "Spectrogram":
            Visualize.plot_spectrogram(colours, path = 'output.wav')
        else:
            Visualize.plot('output.wav', type_of_plot)