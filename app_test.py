from os import X_OK
from pages.test_page import test_page
from pages.noise_page import noise_page
import librosa as lb
import streamlit as st
import plotly.express as px
from streamlit.proto.Selectbox_pb2 import Selectbox
from pages.vad_page import vad_page
from pages.home_page import Home
from pages.model_page import model_page
from pages.params_page import Params
from pages.visualization_page import visualization_page

# Vars 
params_button = False
colours_vad = None
colours = None
file = False

# Henceforth Logo 
st.sidebar.image("https://user-images.githubusercontent.com/46791116/119359585-b194af00-bca1-11eb-8365-a0c4e5bcca68.png")
st.sidebar.write("---")


# Setting params 
with st.sidebar.beta_expander("⚙️ Setting parameters "):
    
    st.title("Recording duration")
    duration = st.slider("", 0, 10, 3) 
    # "---"
    st.title("Sample rate")
    sr = st.slider("", 8000, 44000, 16000) 
    # "---"
    # st.title("Type of record")
    # audio_or_video = st.radio("", ("Audio", "Video"))
    # "---"
    rec_upload = st.radio("",("Record an audio", "Upload an audio"))
    if rec_upload == "Record an audio" : 
        params_button = st.button("Start Recording")
        if params_button:
            Params.show(rec_upload, duration, sr)
    else : 
        file = st.file_uploader("", type = ['wav'])
        if file : 
            Params.show(rec_upload, duration, sr, file=file)


# Visualize data 
with st.sidebar.beta_expander("📊 Visualizations "):
    vis_selectbox = st.selectbox(
        "Choose type of plot",
        ("Oscillogram", "Spectrogram", "3D representation")
    )
    if vis_selectbox == "Spectrogram" :
            named_colorscales = px.colors.named_colorscales()
            default_ix = named_colorscales.index('turbo')
            colours = st.selectbox(('Choose a colour pallete'), named_colorscales, index=default_ix)
    vis_button = st.button("Visualize")

# Voice Activity Detection
with st.sidebar.beta_expander("🔊 Voice Activity Detection "):
    st.title("Silence's threshold")
    threshold = st.slider("", 0, 200, 7) 
    vad_type = st.selectbox(
        "",
        ("Oscillogram", "Spectrogram")
    )
    vad_selectbox = st.selectbox(
        "",
        ("Speech regions", "Non-Speech regions")
    )
    # vad_button = st.button("Detect")

    if vad_type == "Spectrogram" :
            named_colorscales = px.colors.named_colorscales()
            default_ix = named_colorscales.index('turbo')
            colours_vad= st.selectbox(('Choose a colour'), named_colorscales, index=default_ix)
    vad_button = st.button("Detect")

# Handling the noise
with st.sidebar.beta_expander("🔇 Reducing the Noise "):
    noise_selectbox = st.selectbox(
        "Choose The Reducer Method",
        ("VisuShrink", "BayesShrink")
    )
    noise_button = st.button("Reduce")


# Testing the model 
with st.sidebar.beta_expander("🚀 Testing The Model"):
    model_radio = st.radio("Choose Data",("RAVDESS", "HDFASR"))
    # st.title("Choose the model")
    approach_selectbox = st.selectbox(
        "Choose the Approach",
        ("Machine Learning", "Deep Learning")
    )
    if approach_selectbox == "Machine Learning":
         model_selectbox = st.selectbox(
        "Choose the model",
        ("MFCC-SVM", "MFCC-GNB")
        )
    else : 
        model_selectbox = st.selectbox(
        "Choose the model",
        ("CNN of spectrogram", "FFT-Conv1D")
         )
    model_button = st.button("Identify")

# # Handling the noise
# with st.sidebar.beta_expander("Test Code party"):
#     test_button = st.button("Test")

print(model_button)

if model_button :
    model_page.show(approach_selectbox, model_selectbox, model_radio)
elif vad_button : 
    vad_page.show(vad_selectbox,threshold, vad_type, colours= colours_vad)
elif vis_button : 
    visualization_page.show(vis_selectbox, colours = colours)
elif noise_button :
    noise_page.show(noise_selectbox)
# elif test_button: 
#     test_page.show()
else: 
    Home.show()

