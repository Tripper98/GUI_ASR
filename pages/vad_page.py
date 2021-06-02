import streamlit as st 
from classes.vad import VAD
from classes.visualize import Visualize

class vad_page : 
    @staticmethod
    def intro():
        page_icon = "https://user-images.githubusercontent.com/46791116/119490443-c2522d00-bd54-11eb-846b-31e5ffa2fc36.png"
        description = f"""
        <div align='center'>
        <img src={page_icon}
        width="100" height="100">

        # VOICE ACTIVITY DETECTION

        [![Twitter](https://badgen.net/badge/icon/Twitter?icon=twitter&label)](https://twitter.com/pytorch_ignite)
        [![GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/pytorch-ignite/code-generator)
        [![Release](https://badgen.net/github/tag/pytorch-ignite/code-generator/?label=release)](https://github.com/pytorch-ignite/code-generator/releases/latest)

        </div>
        
        <details>

        <summary>
        <samp>What is a Voice Activity Detection?</samp>
        </summary>

        Nanani nana

        </details>

        ---
        """
        st.write(description, unsafe_allow_html=True)
        

    @staticmethod
    def show(vad_selectbox, threshold, vad_type, colours= None):
        Speech , Non_speech = VAD.vad_dyali('output.wav', threshold=threshold)
        vad_page.intro()
        if vad_selectbox == "Non-Speech regions" : 
            Visualize.plot_vad(X_audio=Speech[0], Y_audio= Speech[1], type_of_plot=vad_type, colours = colours)
        else : 
            Visualize.plot_vad(X_audio=Non_speech[0], Y_audio= Non_speech[1], type_of_plot=vad_type, colours = colours)

        