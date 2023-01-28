import argparse
import numpy as np
import torch
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none

parser = argparse.ArgumentParser()
#parser.add_argument('input_filename', help='input text file name')
parser.add_argument('output_filename', help='output wave file name')

args = parser.parse_args()

#input_fname = args.input_filename
output_fname = args.output_filename

fs, lang = 24000, "Japanese"
model='exp/tts_train_raw_phn_jaconv_pyopenjtalk_prosody/178epoch.pth'
config='exp/tts_train_raw_phn_jaconv_pyopenjtalk_prosody/config.yaml'

text2speech = Text2Speech.from_pretrained(
    model_file=model,
    train_config=config,
    vocoder_tag="parallel_wavegan/jsut_parallel_wavegan.v1",
    device="cuda",
)

pause = np.zeros(30000, dtype=np.float32)

#with open(input_fname, 'r') as f:
#    x = f.read()

x = "私はサッカーが好きです"
sentence_list = x.split('<pause>')

wav_list = []

for sentence in sentence_list:
    with torch.no_grad():
        result = text2speech(sentence)["wav"]
        wav_list.append(np.concatenate([result.view(-1).cpu().numpy(), pause]))

final_wav = np.concatenate(wav_list)

from scipy.io.wavfile import write
write(output_fname, rate=text2speech.fs, data=final_wav)