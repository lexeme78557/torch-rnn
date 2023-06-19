import torch
import argparse
import string

from model import *

all_characters = string.printable
n_characters = len(all_characters)

# Turning a string into a tensor
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor



# increasing the temperature will increase chaos
# WARNING: temperature which are too high may result in gibberish
def generate(decoder, start_str='A', predict_len=1000, temperature=0.8, cuda=False):
    hidden = decoder.init_hidden(1)
    start_input = Variable(char_tensor(start_str).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        start_input = start_input.cuda()
    predicted = start_str

    # Use starting string to "build up" hidden state
    for p in range(len(start_str) - 1):
        _, hidden = decoder(start_input[:,p], hidden)
        
    inp = start_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the trained model as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_one = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_one]
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    return predicted


if __name__ == '__main__':
# Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('-s', '--start_str', type=str, default='A')
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    decoder = torch.load(args.filename)
    del args.filename
    print(generate(decoder, **vars(args)))

