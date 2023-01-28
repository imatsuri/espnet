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
model='/work/abelab4/m_iwa/espnet/egs2/jsut_5th/tts1/exp/tts_train_raw_phn_jaconv_none/valid.loss.ave_5best.pth'
config='exp/tts_train_raw_phn_jaconv_none/config.yaml'

text2speech = Text2Speech.from_pretrained(
    model_file=model,
    train_config=config,
    vocoder_tag="parallel_wavegan/jsut_parallel_wavegan.v1",
    device="cuda",
)

pause = np.zeros(30000, dtype=np.float32)

#with open(input_fname, 'r') as f:
#    x = f.read()

x = "^ s o ] o d e s u n e _ k a [ o n a ] N k a n a r a # k o [ n o # k u [ s u r i t o k a _ t e ] y a # a [ s h i ] n a r a # k o [ n o # k u [ s u r i g a # y o [ k u i d e m a ] s u n e $"
sentence_list = x.split('<pause>')

wav_list = []

for sentence in sentence_list:
    with torch.no_grad():
        result = text2speech(sentence)["wav"]
        wav_list.append(np.concatenate([result.view(-1).cpu().numpy(), pause]))

final_wav = np.concatenate(wav_list)

from scipy.io.wavfile import write
write(output_fname, rate=text2speech.fs, data=final_wav)