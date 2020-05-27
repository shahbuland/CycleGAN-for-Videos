from nn.train import train
from nn.constants import *
from nn.models import CycleGAN
from vids2data import vid_io

model = CycleGAN()
if USE_CUDA: model.cuda()

A = vid_io.load("A.npy.gz")
B = vid_io.load("B.npy.gz")

# Train model on A and B for 15000 iterations
train(model, A, B, 15000)
