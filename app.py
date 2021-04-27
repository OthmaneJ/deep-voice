import os
import math

import dash
import dash_html_components as html
import dash_core_components as dcc
# import plotly.express as px
# from IPython.display import Audio
# from IPython.utils import io
# from synthesizer.inference import Synthesizer
# from encoder import inference as encoder
# from vocoder import inference as vocoder
from pathlib import Path
# import numpy as np
import librosa
import scipy
import pydub
import soundfile as sf
import json

with open('latest_embeddings.json') as f:
  new_embeddings = json.load(f)


celebrities = [el['name'] for el in new_embeddings]

# encoder_weights = Path("./encoder/saved_models/pretrained.pt")
# vocoder_weights = Path("./vocoder/saved_models/pretrained.pt")
# syn_dir = Path("./synthesizer/saved_models/pretrained/pretrained.pt")
# encoder.load_model(encoder_weights)
# synthesizer = Synthesizer(syn_dir)
# vocoder.load_model(vocoder_weights)


external_stylesheets = [
    "https://use.fontawesome.com/releases/v5.0.7/css/all.css",
    'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css',
    'https://fonts.googleapis.com/css?family=Roboto&display=swap'
]

# app = dash.Dash(__name__)

app = dash.Dash(
    __name__, 
    external_stylesheets=external_stylesheets,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    suppress_callback_exceptions=True
)

server = app.server

app.layout = html.Div(
    [
        html.H4(children="AI Celebrity Voice Cloning"),
        dcc.Markdown("Clone the voice of your favourite celebrity using Deep Learning."),
        dcc.Markdown("""
    **Instructions:** Choose your favourite celebrity from the scroll down and type a sentence (between 10 and 20 words) that you wish your celebrity would say, then click submit and wait for about 10 seconds
    """,
        ),
        dcc.Markdown("**Choose your celebrity**"),
        html.Div(html.Img(id='celebrity_img',src='https://m.media-amazon.com/images/M/MV5BMTc1MDI0MDg1NV5BMl5BanBnXkFtZTgwMDM3OTAzMTE@._V1_SY1000_CR0,0,692,1000_AL_.jpg',style={'width':'200px'}),style={'marginTop':'10px',"marginBottom":'10px'}),
        dcc.Dropdown(id="celebrity-dropdown",options=[{'label':celebrity,'value':i} for i,celebrity in enumerate(celebrities)]),
        html.Div(id="slider-output-container"),
        dcc.Markdown("**Type a sentence and click submit**"),
        html.Div(dcc.Textarea(id="transcription_input",maxLength=300,rows=2,style={'width':'100%'},
                              value ='I believe in living in the present and making each day count. I don’t pay much attention to the past or the future.')),
        html.Div(html.Button('Submit', id='submit', n_clicks=0)),
        html.Br(),
        dcc.Loading(id="loading-1",
                    children=[html.Audio(id="player",src = "./assets/generated/new_test.wav", controls=True, style={
          "width": "100%",
        })],type='default'),
        html.H4('How would you rate the quality of the audio ?'),
        dcc.Slider(id='rating',max=5,min=1,step=1,marks={i: f'{i}' for i in range(1, 6)},),
        # dcc.Graph(id="waveform", figure=fig),
        html.Div(html.Button('Rate', id='rate-button', n_clicks=0)),
        html.H4("Please put a rating up here!",id='rating-message'),
        dcc.ConfirmDialog(id='confirm',message="Too many words (>50) or too little (<10) may effect the quality of the audio, continue at your own risk ^^'"),
        # html.A(children=[html.Img(src='https://cdn.buymeacoffee.com/buttons/default-orange.png',alt="Buy Me Coffee",height="41",width="174")],href='https://www.buymeacoffee.com/OthmaneJ'),
    
    ]
    ,style={'textAlign': 'center','marginRight':'100px','marginLeft':'100px','marginTop':'50px','marginBottom':'50px'})

# Set picture of celebrity
@app.callback(
    dash.dependencies.Output('celebrity_img','src'),
    [dash.dependencies.Input('celebrity-dropdown','value')]
)

def display_image(celebrity):
  return new_embeddings[celebrity]['img']


# Transcribe audio
@app.callback(
    dash.dependencies.Output("confirm", "displayed"),
    [dash.dependencies.Input("submit","n_clicks"),
     ],
    [dash.dependencies.State("celebrity-dropdown","value"),
     dash.dependencies.State("transcription_input", "value")],
)

def display_warning(n_clicks,celebrity,value):
    n_words=  len(value.split(' '))
    print(n_words)
    if n_words>50 or n_words<10:
        return True
    return False

#  Transcribe audio
@app.callback(
    dash.dependencies.Output("player", "src"),
    [dash.dependencies.Input("submit","n_clicks"),
     ],
    [dash.dependencies.State("celebrity-dropdown","value"),
     dash.dependencies.State("transcription_input", "value")],
)

def vocalize(n_clicks,celebrity,value):
    text= value
    embed = new_embeddings[celebrity]['embed']
    print(text)
    print(celebrity)
    print("Synthesizing new audio...")
    specs = synthesizer.synthesize_spectrograms([text], [embed],)

    print("Vocoder generating waveform")
    generated_wav = vocoder.infer_waveform(specs[0])
    temp = generated_wav
    generated_wav_new = np.pad(temp, (0, synthesizer.sample_rate), mode="constant")
    generated_wav_new = encoder.preprocess_wav(generated_wav_new)
    sf.write("./assets/generated/new_test.wav", generated_wav_new.astype(np.float32), synthesizer.sample_rate)

    return 'assets/generated/new_test.wav'

@app.callback(
    dash.dependencies.Output("rating-message", "value"),
    [dash.dependencies.Input("rate-button","n_clicks"),
     ],
    [dash.dependencies.State("rating","value")], 
)

def print_rating(n_clicks,rating):
    print(rating)
    return 'your rating is ' + str(rating)


if __name__ == "__main__":
    app.run_server(debug=True)
