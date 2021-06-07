import os
import librosa as lb
import soundfile as sf
from librosa.core import audio
from skimage.restoration import denoise_wavelet

class Handling_Noise():

    @staticmethod
    def read_audio(path, sr =16000):
        ''' It returns samples and sample rate from a given audio's path '''
        return lb.load(path, sr)

    @staticmethod 
    def reduce_noise(method):
        audio, sr = Handling_Noise.read_audio('output.wav')
        denoised_audio = denoise_wavelet(audio, method= method, mode='soft',
                                        wavelet_levels= 3, wavelet='sym8',
                                        rescale_sigma=True)
        output_file = 'non_noise.wav'
        if os.path.exists(output_file) : 
            os.remove(output_file)
        sf.write(output_file, denoised_audio, sr, format='wav', subtype='PCM_16')
        