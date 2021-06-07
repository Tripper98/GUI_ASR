from classes.visualize import Visualize
from classes.reduce_noise import Handling_Noise
import streamlit as st 

class noise_page : 
    @staticmethod
    def intro():
        page_icon = "https://user-images.githubusercontent.com/46791116/120996731-7eace980-c77e-11eb-8505-66151464cdd8.png"
        description = f"""
        <div align='center'>
        <img src={page_icon}
        width="100" height="100">

        # REDUCING THE NOISE
        </div>

        <details>

        <summary>
        <samp>What is method 1?</samp>
        </summary>

        Nanani nana

        </details>

        <details>

        <summary>
        <samp>What is method 2?</samp>
        </summary>

        Nanani nana

        </details>

        ---
        """
        st.write(description, unsafe_allow_html=True)

    @staticmethod
    def show(noise_selectbox):
        noise_page.intro()
        Handling_Noise.reduce_noise(noise_selectbox)
        Visualize.plot_oscillogram('output.wav')
        Visualize.plot_oscillogram('non_noise.wav')
