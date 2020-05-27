import torch
from torch import nn
import torch.nn.functional as F
from .constants import *
import numpy as np

# Fixes T wrt data type
def Tensor(T):
	T = T.float()
	if USE_CUDA: T = T.cuda()
	return T

# Gets np image from model tensor
def npimage(A):
	A = A.detach().cpu().numpy()
	A = np.moveaxis(A,1,3)	
	return A

# Gets labels for discriminator
def get_labels(val,size):
	return Tensor(val*torch.ones(size,1,4,4))
