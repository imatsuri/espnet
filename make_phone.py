import argparse
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer

def file2phoneme(out_file, in_file):
    phone = PhonemeTokenizer(g2p_type = "pyopenjtalk_prosody")

    out = open(out_file, "w")

    with open(in_file, "r") as f:
        for line in f:
            line = line.replace(' ','').rsplit()
            token = phone.text2tokens(line)
            tok_str = " ".join(token)
            out.write(tok_str+"\n")
        out.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename', help='input text file name')
    parser.add_argument('output_filename', help='output wave file name')

    args = parser.parse_args()
    input_file = args.input_filename
    output_file = args.output_filename
    file2phoneme(output_file, input_file)

if __name__ == '__main__':
    main()

