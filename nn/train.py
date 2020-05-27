from .constants import *
from .ops import Tensor, npimage

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchsummary import summary

def train(model, A, B, iterations):
        
        data_sizes = [A.shape[0],B.shape[0]]
        print("Dataset shapes:", data_sizes)

        # Model summary
        #summary(model, [(3, 256, 256),(3,256,256)])

        if LOAD_CHECKPOINTS:
                model.load_checkpoint()
        
        # Returns batch in form of (A,B)
        def get_batch(batch_size):
                indsA = torch.randint(0,data_sizes[0],(batch_size,))
                indsB = torch.randint(0,data_sizes[1],(batch_size,))

                batch_A = A[indsA]
                batch_B = B[indsB]

                if batch_size == 1:
                        batch_A = np.expand_dims(batch_A, 0)
                        batch_B = np.expand_dims(batch_B, 0)

                batch_A = torch.from_numpy(batch_A).permute(0,3,1,2).float()/255
                batch_B = torch.from_numpy(batch_B).permute(0,3,1,2).float()/255
                # Make sure both are right size
                batch_A = F.interpolate(batch_A, size=[256,256])
                batch_B = F.interpolate(batch_B, size=[256,256])
                
                if USE_CUDA:
                        batch_A = batch_A.cuda()
                        batch_B = batch_B.cuda()
                return batch_A, batch_B

        # Draws samples
        def save_samples(title):
                fig,axs = plt.subplots(4,4)
                A_sample, B_sample = get_batch(4)
                B_fake, A_fake = model.A_to_B(A_sample), model.B_to_A(B_sample)

                A_sample = npimage(A_sample)
                B_sample = npimage(B_sample)
                A_fake = npimage(A_fake)
                B_fake = npimage(B_fake)

                for r in range(4):
                        axs[r][0].imshow(A_sample[r])
                        axs[r][1].imshow(B_fake[r])
                        axs[r][2].imshow(B_sample[r])
                        axs[r][3].imshow(A_fake[r])

                plt.savefig(title+".png")
        
        # Actual training loop
        for ITER in range(iterations):
                
                # Train discriminators
                for i in range(N_CRITIC):
                        A_batch,B_batch = get_batch(BATCH_SIZE)
                        D_A_loss,D_B_loss = model.train_disc_on_batch(A_batch,B_batch)
                
                # Train generators
                A_batch,B_batch = get_batch(BATCH_SIZE)
                G_A_loss, G_B_loss = model.train_gen_on_batch(A_batch, B_batch)
                
                # Write things down
                print("[",ITER,"/",ITERATIONS,"]")
                print("D A Loss:",D_A_loss)
                print("G A Loss:",G_A_loss)
                print("D B Loss:",D_B_loss)
                print("G B Loss:",G_B_loss)

                if (ITER+1) % SAMPLE_INTERVAL == 0:
                        save_samples(str(ITER))
                if (ITER+1) % CHECKPOINT_INTERVAL == 0:
                        model.save_checkpoint()
