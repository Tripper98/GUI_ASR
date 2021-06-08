import streamlit as st

class Home : 

    @staticmethod
    def show():
        page_title = "AUTOMATIC SPEECH RECOGNITION"
            # henceforth logo
            # https://user-images.githubusercontent.com/46791116/119359585-b194af00-bca1-11eb-8365-a0c4e5bcca68.png
        page_icon = "https://user-images.githubusercontent.com/46791116/119359955-09cbb100-bca2-11eb-9d83-b3bb41c64a87.png"
        description = f"""
        <div align='center'>
        <img src={page_icon}
        width="100" height="100">

        # AUTOMATIC SPEECH RECOGNITION


        [![Twitter](https://badgen.net/badge/icon/Twitter?icon=twitter&label)](https://twitter.com/pytorch_ignite)
        [![GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/pytorch-ignite/code-generator)
        [![Release](https://badgen.net/github/tag/pytorch-ignite/code-generator/?label=release)](https://github.com/pytorch-ignite/code-generator/releases/latest)

        </div>

        <details>

        <summary>
        <samp>Learn More</samp>
        </summary>

        #### AUTOMATIC SPEECH RECOGNITION, what is it ?

        - "AUTOMATIC SPEECH RECOGNITION" is a streamlit application to produce quick-start python code
        for common training tasks in deep learning.
        - Code is using PyTorch framework and PyTorch-Ignite library can be configured using the UI.

        #### Why to use AUTOMATIC SPEECH RECOGNITION ?

        - Start working on a task without rewriting everything from scratch: Kaggle competition, client prototype project, etc.

        </details>

        <details>

        <summary>
        <samp>Architecture of project</samp>
        </summary>

        <div align='center'>
        <img src={page_icon} width="100" height="100">
        </div>

        </details>

        <details open="true">
        <summary>
        <samp>Get Started</samp>
        </summary>

        #### How to use it ?

        1. ⚙️ Set parameters & Record (or upload) your audio.
        2. 📊 Visualize your audio.
        3. 🔊 Detect speech regions & non speech regions from your original audio.
        3. 🔇 Reduce The Noise.
        4. 🚀 Choose a model & Identify the speaker.

        </details>

        ---
        """
        # st.set_page_config(page_title=page_title, page_icon=page_icon)
        st.write(description, unsafe_allow_html=True)