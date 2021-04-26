import os
import math

import dash
import dash_html_components as html
import dash_core_components as dcc
import numpy as np
import pandas as pd
import plotly.express as px

import time

import sys
sys.path.append('C:/Users/hp/Desktop/cours cs 3A/projects/voice cloning app/Real-Time-Voice-Cloning')

from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import scipy
import pydub

import json

# with open('new_embeddings.json') as f:
with open('latest_embeddings.json') as f:    
    new_embeddings = json.load(f)

celebrities = [el['name'] for el in new_embeddings]

os.chdir('C:/Users/hp/Desktop/cours cs 3A/projects/voice cloning app/Real-Time-Voice-Cloning')

encoder_weights = Path("./encoder/saved_models/pretrained.pt")
vocoder_weights = Path("./vocoder/saved_models/pretrained.pt")
syn_dir = Path("./synthesizer/saved_models/pretrained/pretrained.pt")
encoder.load_model(encoder_weights)
synthesizer = Synthesizer(syn_dir)
vocoder.load_model(vocoder_weights)


text= "I believe in living in the present and making each day count."
embed = new_embeddings[4]['embed']

print(text)
print(celebrity)

print("Synthesizing new audio...")
specs = synthesizer.synthesize_spectrograms([text], [embed],)

print("Vocoder generating waveform")

start_time = time.time()
generated_wav = vocoder.infer_waveform(specs[0])
print("--- %s seconds ---" % (time.time() - start_time))

temp = generated_wav
generated_wav_new = np.pad(temp, (0, synthesizer.sample_rate), mode="constant")
generated_wav_new = encoder.preprocess_wav(generated_wav_new)

sf.write("./generated/new_test.wav", generated_wav_new.astype(np.float32), synthesizer.sample_rate)

# def write(f, sr, x, normalized=False):
#     """numpy array to MP3"""
#     channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
#     if normalized:  # normalized array - each item should be a float in [-1, 1)
#         y = np.int16(x * 2 ** 15)
#     else:
#         y = np.int16(x)
#     song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
#     song.export(f, format="mp3", bitrate="320k",)

import sounddevice as sd
sd.stop()
sd.play(generated_wav, synthesizer.sample_rate,)



import soundfile as sf
sf.write("new_wav.wav", generated_wav.astype(np.float32), synthesizer.sample_rate)




from scipy.io.wavfile import write

save_path = 'C:/Users/hp/Desktop/cours cs 3A/projects/voice cloning app/Real-Time-Voice-Cloning'
write('output.wav',rate = synthesizer.sample_rate,data = generated_wav_normalized,)

y = np.int16(generated_wav_new * 2 ** 15)
song = pydub.AudioSegment(y.tobytes(), frame_rate=synthesizer.sample_rate, sample_width=2, channels=1)
song.export('output_new.wav',format='wav',bitrate="320k")

song.frame_rate

scaled = np.int16(generated_wav_new/np.max(np.abs(generated_wav_new)) * 32767)
write('test.wav',16000,scaled)

from scipy.io.wavfile import write


scaled = np.int16(data/np.max(np.abs(data)) * 32767)
write('test.wav', synthesizer.sample_rate, scaled)







