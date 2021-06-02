import os
import time
import streamlit as st 
import soundfile as sf
import sounddevice as sd    
from pydub import AudioSegment
from scipy.io.wavfile import write
from classes.visualize import Visualize


class Params : 
    # @staticmethod
    # def intro():
    #     page_icon = "https://user-images.githubusercontent.com/46791116/119503223-a35a9780-bd62-11eb-8382-6919502c1471.png"
    #     description = f"""
    #     <div align='center'>
    #     <img src={page_icon}
    #     width="100" height="100">

    #     # AUTOMATIC SPEECH RECOGNITION

    #     Application to generate your training scripts with [PyTorch-Ignite](https://github.com/pytorch/ignite).

    #     [![Twitter](https://badgen.net/badge/icon/Twitter?icon=twitter&label)](https://twitter.com/pytorch_ignite)
    #     [![GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/pytorch-ignite/code-generator)
    #     [![Release](https://badgen.net/github/tag/pytorch-ignite/code-generator/?label=release)](https://github.com/pytorch-ignite/code-generator/releases/latest)

    #     </div>

    #     ---
    #     """
    #     st.write(description, unsafe_allow_html=True)

    @staticmethod
    def show_progress(sec, radio_type ) : 
        with st.spinner(f"{radio_type} the audio..."):
            time.sleep(sec)
            st.success('✔️ Done!')
        # with st.empty():
        #     for seconds in range(int(sec)):
        #         st.write(f"{radio_type} the audio...")
        #         time.sleep(0.5)
        #         st.write("✔️ Done!")
    
    @staticmethod
    def record(duration, sr, radio_type):
        recording = sd.rec(int(duration * sr), samplerate=sr, channels=2)
        Params.show_progress(int(duration), radio_type )
        sd.wait()   
        output_file = 'output.wav'
        if os.path.exists(output_file) : 
            os.remove(output_file)
        # write('output.wav', sr, recording)
        sf.write(output_file, recording, sr, format='wav', subtype='PCM_16')

    @staticmethod
    def uploaded(file, duration, sr):
        output_file = 'output.wav'
        if os.path.exists(output_file) : 
            os.remove(output_file)
        data, sr = Visualize.read_audio(file,sr)
        sf.write(output_file, data, sr, format='wav', subtype='PCM_16')
        # In case if u wanna segment ur audio
        # N = data.shape[0]/sr
        # if N > duration : 
        #     t1 = (N-duration)*1000
        #     N = N*1000
        #     newAudio = AudioSegment.from_wav(file)
        #     newAudio = newAudio[t1:N]
        #     newAudio.export(output_file, format="wav")
        # else :

    @staticmethod
    def show(rec_upload, duration, sr, file =None):
        # Params.intro()
        if rec_upload == "Record an audio" : 
            Params.record(duration, sr, "Recording")
        else : 
            Params.show_progress(int(duration/3), "Uploading")
            Params.uploaded(file, duration, sr)
