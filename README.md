# torch-rnn

A PyTorch implementation of [char-rnn](https://github.com/karpathy/char-rnn) with more details from this [blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## Training 

Download this [Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).  Or try with your own text!

Run `train.py` with the dataset filename to train

```
python train.py shakespeare.txt --cuda=True/False

```

## Text Generation

Run `generate.py` with the saved model from training and a starting string to begin generation from

```
 python generate.py shakespeare.pt -s="A" --cuda=True/False

```

