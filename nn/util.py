import torch
import numpy as np
from .constants import *
import matplotlib.pyplot as plt

CUDA = USE_CUDA

# Converts large np arrays through model in chunks
def chunk_A_to_B(model, A_data, chunk_size):
    frames = A_data.shape[0]
    num_chunks = frames//chunk_size
    chunks = [A_data[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    B_chunks = []

    for i, chunk in enumerate(chunks):
        chunk = torch.from_numpy(chunk).float()
        if CUDA: chunk = chunk.cuda()
        chunk = chunk/255
        chunk = chunk.permute(0,3,1,2) # NHWC to NCHW
        chunk = model.A_to_B(chunk)

        chunk = chunk.permute(0,2,3,1) # reverse above
        chunk = chunk.detach().cpu().numpy()
        chunk = chunk * 255
        chunk = chunk.astype(np.uint8)
        B_chunks.append(chunk)
        print("[" + str(i) + " / " + str(num_chunks) + "]")

    return np.concatenate(B_chunks, axis = 0)

def chunk_B_to_A(model, B_data, chunk_size):
    frames = B_data.shape[0]
    num_chunks = frames//chunk_size
    chunks = [B_data[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    A_chunks = []

    for i, chunk in enumerate(chunks):
        chunk = torch.from_numpy(chunk).float()
        if CUDA: chunk = chunk.cuda()
        chunk = chunk/255
        chunk = chunk.permute(0,3,1,2) # NHWC to NCHW
        chunk = model.B_to_A(chunk)
        
        chunk = chunk.permute(0,2,3,1) # reverse above
        chunk = chunk.detach().cpu().numpy()
        chunk = chunk * 255
        chunk = chunk.astype(np.uint8)
        
        A_chunks.append(chunk)
        print("[" + str(i) + " / " + str(num_chunks) + "]")

    return np.concatenate(A_chunks, axis = 0)
        
