# from IPython.display import Audio
# from IPython.utils import io
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import scipy
import pydub
import json
import argparse

encoder_weights = Path("./encoder/saved_models/pretrained.pt")
vocoder_weights = Path("./vocoder/saved_models/pretrained.pt")
syn_dir = Path("./synthesizer/saved_models/pretrained/pretrained.pt")
encoder.load_model(encoder_weights)
synthesizer = Synthesizer(syn_dir)
vocoder.load_model(vocoder_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type = str, help = "path of the audio sample")
    parser.add_argument("--name", type = str, help = "name of the celebrity")
    parser.add_argument("--img_url", type = str, help = "url of the image")

    args = parser.parse_args()

    if args.name:

        outfile = args.path
        in_fpath = Path(outfile)
        print("preprocessing the training audio file")
        reprocessed_wav = encoder.preprocess_wav(in_fpath)
        original_wav, sampling_rate = librosa.load(in_fpath)
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        embed = encoder.embed_utterance(preprocessed_wav)
        try:
            with open('latest_embeddings.json') as f:
                new_embeddings = json.load(f)
        except:
            new_embeddings = []

        new_embeddings.append({'name':args.name,'embed':embed.tolist(),'img':args.img_url})
        

        with open('latest_embeddings.json', 'w') as fp:
            json.dump(new_embeddings, fp)
    else:

        meta_data = [{"path":"./train_data/Gal Gadot on Wonder Woman Costumes and Her Eye-Opening Pregnancy _ Screen Tests _ W Magazine-JMz4uYECaaA.wav","name":"Gal Gadot","img":"https://m.media-amazon.com/images/M/MV5BMjUzZTJmZDItODRjYS00ZGRhLTg2NWQtOGE0YjJhNWVlMjNjXkEyXkFqcGdeQXVyMTg4NDI0NDM@._V1_.jpg"},
                    {"path":"./train_data/Brad Pitt on His First Kiss, What He Wore to Prom, and His Early Days as an Extra _ W Magazine-EOLafh8DPFM.wav","name":"Brad Pitt","img":"https://m.media-amazon.com/images/M/MV5BMjA1MjE2MTQ2MV5BMl5BanBnXkFtZTcwMjE5MDY0Nw@@._V1_SY1000_CR0,0,665,1000_AL_.jpg"},
                    {"path":"./train_data/Bradley Cooper on Movies That Make Him Cry & His Crush on Sienna Miller _ Screen Tests _ W Magazine-bGtd-RjzQuI.wav","name":"Bradley Cooper","img":"https://m.media-amazon.com/images/M/MV5BMjUzZTJmZDItODRjYS00ZGRhLTg2NWQtOGE0YjJhNWVlMjNjXkEyXkFqcGdeQXVyMTg4NDI0NDM@._V1_.jpg"},
                    {"path":"./train_data/Emma Watson's speech on gender equality-dSHJYyRViIU.wav","name":"Emma Watson","img":"https://m.media-amazon.com/images/M/MV5BMTQ3ODE2NTMxMV5BMl5BanBnXkFtZTgwOTIzOTQzMjE@._V1_SY1000_CR0,0,810,1000_AL_.jpg"},
                    {"path":"./train_data/Margot Robbie On Tonya Harding and Her Favorite Halloween Costumes _ Screen Tests _ W Magazine-kv9rW4l1bB0.wav","name":"Margot Robbie","img":"https://m.media-amazon.com/images/M/MV5BMTgxNDcwMzU2Nl5BMl5BanBnXkFtZTcwNDc4NzkzOQ@@._V1_SY999_SX750_AL_.jpg"},
                    {"path":"./train_data/Scarlett Johansson on Black Widow, Spike Jonze, and Chris Evans _ Screen Tests _ W Magazine-5noBdgcGPVU.wav","name":"Scarlett Johansson","img":"https://m.media-amazon.com/images/M/MV5BMTM3OTUwMDYwNl5BMl5BanBnXkFtZTcwNTUyNzc3Nw@@._V1_SY1000_CR0,0,824,1000_AL_.jpg"},
                    ]

        new_embeddings = []
        for i in range(len(meta_data)):
            outfile = meta_data[i]["path"]
            in_fpath = Path(outfile)
            print("preprocessing the training audio file")
            reprocessed_wav = encoder.preprocess_wav(in_fpath)
            original_wav, sampling_rate = librosa.load(in_fpath)
            preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
            embed = encoder.embed_utterance(preprocessed_wav)

            new_embeddings.append({'name':meta_data[i]['name'],'embed':embed.tolist(),'img':meta_data[i]['img']})
        

        with open('latest_embeddings.json', 'w') as fp:
            json.dump(new_embeddings, fp)


    # python embed.py --path "./train_data/Gal Gadot on Wonder Woman Costumes and Her Eye-Opening Pregnancy _ Screen Tests _ W Magazine-JMz4uYECaaA.wav" --name "Gal Gadot" --img "https://m.media-amazon.com/images/M/MV5BMjUzZTJmZDItODRjYS00ZGRhLTg2NWQtOGE0YjJhNWVlMjNjXkEyXkFqcGdeQXVyMTg4NDI0NDM@._V1_.jpg"