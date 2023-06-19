import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import random
import string
import unidecode
import time
import math

from tqdm import tqdm

from model import *
from generate import *

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--cuda', action='store_true')
args = argparser.parse_args()

if args.cuda:
    print("Using CUDA")

file = unidecode.unidecode(open(args.filename).read())
file_len = len(file)

all_characters = string.printable
n_characters = len(all_characters)

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

def random_training_set(chunk_len=256, batch_size = 128):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def train(inp, target):
    hidden = decoder.init_hidden(128)
    if args.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for c in range(256):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(128, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / 256

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)


def time_since(past):
    s = time.time() - past
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# Initialize models and start training
decoder = CharRNN(
    n_characters,
    n_characters,
)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    decoder.cuda()

start = time.time()
all_losses = []
loss_avg = 0

try:
    print("Training for %d epochs..." % 5000)
    for epoch in tqdm(range(1, 5000 + 1)):
        loss = train(*random_training_set(256, 128))
        loss_avg += loss

        if epoch % 250 == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / 5000 * 100, loss))
            print(generate(decoder, 'Wh', 100, cuda=args.cuda), '\n')

    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

