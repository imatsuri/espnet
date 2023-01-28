import argparse
import numpy as np
import torch
from espnet2.bin.tts_inference import Text2Speech
from scipy.io.wavfile import write

def speech(it, ow, model):
    fs, lang = 44100, "Japanese"
    pause = np.zeros(30000, dtype=np.float32)
    text2speech = Text2Speech.from_pretrained(
        model_file=model,
        device="cuda"
        )
    wav_list = []

    with open(it, "r") as f:
        for line in f:
            sentence_list = line.split('<pause>')
            for sentence in sentence_list:
                with torch.no_grad():
                    result = text2speech(sentence)["wav"]
                    wav_list.append(np.concatenate([result.view(-1).cpu().numpy(), pause]))
        
        final_wav = np.concatenate(wav_list)
        write(ow, rate=text2speech.fs, data=final_wav)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename', help='input text file name')
    parser.add_argument('output_filename', help='output wave file name')
    parser.add_argument('model', help='model path')
    args = parser.parse_args()

    speech(args.input_filename, args.output_filename, args.model)

if __name__ == "__main__":
    main()