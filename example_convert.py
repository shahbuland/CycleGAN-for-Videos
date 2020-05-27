from nn.models import CycleGAN
from nn.constants import *
from nn.ops import Tensor, npimage
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchsummary import summary
from vids2data import vid_io
from nn.util import *

# What to convert to?
# I.e "A" for B to A
# "B" for B to A
convert_to = "B"

# Model setup
model = CycleGAN()
if USE_CUDA: model.cuda()
model.load_checkpoint()
model.eval()

# Converting from type A to B
if convert_to == "B":
    A = vid_io.load("A_test.npy.gz")[0:100]
    vid_io.arr2vid(A, "A.mp4") # Save base video for comparison

    # Last number is chunk size
    fake_B = chunk_A_to_B(model, A, 1)
    vid_io.arr2vid(fake_B, "Fake_B.mp4") # Faked video

    rec_A = chunk_B_to_A(model, fake_B, 1)
    vid_io.arr2vid(rec_A, "Rec_A.mp4") # Reconstructed video

# Converting from type B to A
if convert_to == "A":
    B = vid_io.load("B_test.npy.gz")[0:100]
    vid_io.arr2vid(B, "B.mp4")
    
    fake_A = chunk_B_to_A(model, B, 1)
    vid_io.arr2vid(fake_A, "Fake_A.mp4")

    rec_B = chunk_A_to_B(model, fake_A, 1)
    vid_io.arr2vid(rec_B, "Rec_B.mp4")
