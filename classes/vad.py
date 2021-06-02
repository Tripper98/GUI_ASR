


import numpy as np 
import streamlit as st
from scipy.fft import fft
from classes.visualize import Visualize

class VAD() : 

    @staticmethod
    def signal_energy(signal): 
        ''' It calculates the energy of signal '''
        yf = fft(signal)
        return (yf*np.conj(yf)).sum()

    @staticmethod
    def frame_audio(samples, sr =16000, l_frame = 0.03): 
        ''' '''
        
        frames = []
        N = samples.shape[0]
        win_len = l_frame*sr 
        ll = N//win_len
        i = 0
        while (i <= N):
            frames.append(
                dict(
                    start = int(i),
                    stop = int(i+win_len),
                value = samples[int(i):int(i+win_len)]))
            i+= win_len
        
        return frames

    @staticmethod
    def vad_dyali(audio_path, sr= 16000, threshold = 0, frame_len= 0.03, overlap=0.015):
        ''' It returns speech & non-speech regions from a given audio's path '''
        s, sr = Visualize.read_audio(audio_path, sr) 
        
        N = s.shape[0]
        # Framing signal dyalna 
        frames = VAD.frame_audio(s, sr, frame_len)
        normalized_energy =[] 
        segments = []
        
        for f in frames : 
            # Calculating signal energy of each frame
            energy = VAD.signal_energy(f['value'])
            if(energy>threshold) : 
                segments.append(dict(
                    start = f['start'],
                    stop = f['stop'],
                is_speech = True))
            else : 
                segments.append(dict(
                    start = f['start'],
                    stop = f['stop'],
                is_speech = False))
        
        arr_to_concatenate = [ s[segment['start']:segment['stop']] for segment in segments if segment['is_speech']]
        arr_to_concatenate_2 = [ s[segment['start']:segment['stop']] for segment in segments if not segment['is_speech']]
        
        len_arr = len(arr_to_concatenate)
        speech_samples = np.concatenate(arr_to_concatenate)
        non_speech_samples = np.concatenate(arr_to_concatenate_2)
        X_speech = np.linspace(0, len(speech_samples)/sr,len(speech_samples) )
        X_non_speech = np.linspace(0, len(non_speech_samples)/sr,len(non_speech_samples) )

        # X = np.linspace(0, N/sr, N) 
        # fig, axs = plt.subplots(3)
        # fig.set_figheight(10)
        # fig.set_figwidth(15)
        # fig.suptitle('Plot of Original Speech, Speech regions & Non-speech regions')
        # axs[0].plot(X, s, linewidth=0.4)
        # axs[0].set_title('Original Speech')
        # axs[1].plot(X_speech, speech_samples, linewidth=0.4)
        # axs[1].set_title('Speech Regions')
        # axs[2].plot(X_non_speech, non_speech_samples, 'r', alpha= 0.4)
        # axs[2].set_title('Non speech regions')
    
        return (X_speech, speech_samples), (X_non_speech, non_speech_samples)
