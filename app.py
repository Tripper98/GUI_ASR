# import streamlit.components.v1 as components
# from classes.visualization import Visualize
# from scipy.io.wavfile import write
# import sounddevice as sd
# import streamlit as st
# import soundfile as sf
# import librosa as lb
# import time


# # Description of project 
# # st.sidebar.title("Description of project 💡")
# sr = 16000
# channels = 1
# st.sidebar.image('./img/henceforth.png')
# st.sidebar.write("---")

# # Show progress
# def show_progress(sec) : 
#     with st.empty():
#         for seconds in range(sec):
#             st.write(f"👂 Recording...")
#             time.sleep(0.5)
#             st.write("✔️ Recorded!")


# # Setting params 
# with st.sidebar.beta_expander("⚙️ Setting parameters "):
    
#     st.title("Recording duration")
#     duration = st.slider("", 0.0, 10.0, 3.0) 
#     "---"
#     st.title("Sample rate")
#     sr = st.slider("", 8000, 44000, 16000) 
#     "---"
#     st.title("Type of record")
#     audio_or_video = st.radio("", ("Audio", "Video"))
#     "---"
#     if st.button("Start Recording"):
#         # st.write("Start recording...")
#         recording = sd.rec(int(duration * sr), samplerate=sr, channels=channels).reshape(-1)
#         show_progress(int(duration))
#         # sd.wait()
#         write('output.wav', sr, recording)


# # Visualize data 
# with st.sidebar.beta_expander("📊 Visualizations "):
#     add_selectbox = st.selectbox(
#         "",
#         ("Spectrogram", "Oscillogram", "Mobile phone")
#     )
#     # button_visualization = st.button("Start visualization")
#     if st.button("Start visualization"):
#         Visualize.plot_oscillogram('output.wav')
#         # st.write("Visualize :) ")


# # Voice Activity Detection
# with st.sidebar.beta_expander("🔊 Voice Activity Detection "):
#     add_selectbox = st.selectbox(
#         "",
#         ("Speech regions", "Non-Speech regions")
#     )
#     button_vad = st.button("Detect")

# # Testing the model 
# with st.sidebar.beta_expander("📌 Testing The Model"):
#     st.title("Choose the model")
#     add_selectbox = st.selectbox(
#         "",
#         ("MFCC-SVM", "MFCC-GNB", "CNN of spectrogram", "Last one")
#     )
#     if st.button("Identify the speaker "):
#         st.write("The speaker is : ")





#     page_title = "AUTOMATIC SPEECH RECOGNITION"
#     # henceforth logo
#     # https://user-images.githubusercontent.com/46791116/119359585-b194af00-bca1-11eb-8365-a0c4e5bcca68.png
#     page_icon = "https://user-images.githubusercontent.com/46791116/119359955-09cbb100-bca2-11eb-9d83-b3bb41c64a87.png"
#     description = f"""
# <div align='center'>
# <img src={page_icon}
# width="100" height="100">

# # AUTOMATIC SPEECH RECOGNITION

# Application to generate your training scripts with [PyTorch-Ignite](https://github.com/pytorch/ignite).

# [![Twitter](https://badgen.net/badge/icon/Twitter?icon=twitter&label)](https://twitter.com/pytorch_ignite)
# [![GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/pytorch-ignite/code-generator)
# [![Release](https://badgen.net/github/tag/pytorch-ignite/code-generator/?label=release)](https://github.com/pytorch-ignite/code-generator/releases/latest)

# </div>

# <details>

# <summary>
# <samp>Learn More</samp>
# </summary>

# #### AUTOMATIC SPEECH RECOGNITION, what is it ?

# - "AUTOMATIC SPEECH RECOGNITION" is a streamlit application to produce quick-start python code
# for common training tasks in deep learning.
# - Code is using PyTorch framework and PyTorch-Ignite library can be configured using the UI.

# #### Why to use AUTOMATIC SPEECH RECOGNITION ?

# - Start working on a task without rewriting everything from scratch: Kaggle competition, client prototype project, etc.

# </details>

# <details open="true">
# <summary>
# <samp>Get Started</samp>
# </summary>

# #### How to use it ?

# 1. 📃 Choose a Template.
# 2. ⚙️ Adjust the configuration in the left sidebar. _(click on > if closed)_
# 3. 🔬 Inspect the code in the central widget.
# 4. 📦 Download the source code.
# 5. 🚀 Use it for your project.

# </details>

# ---
# """
# # st.set_page_config(page_title=page_title, page_icon=page_icon)
# st.write(description, unsafe_allow_html=True)

# # audio_file = lb.load('output.wav', sr =16000)
# # st.audio(audio_file, format='audio/wav')
