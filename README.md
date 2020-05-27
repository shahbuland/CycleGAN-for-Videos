# CycleGAN-for-Videos  
  This is an unfinished project from way back that I decided to complete and share. It's a pretty simple implementation of a vanilla cyclegan (meaning architecture is same as it was in the [original paper](https://arxiv.org/pdf/1703.10593.pdf)). There is a lot of added functionality specifically for using it with videos as that is what I primarily made it for (i.e. feeding it a video of day and a video of night and then it learning to convert between the two). I will outline some key stuff with hyperparameters, setup and usage below. 
# Requirements  
Python 3.6  
CUDA, CUDNN (You can turn this off via USE_CUDA in nn/constants.py  but training without it will be very slow)  
Pytorch (and all its dependencies)  
Sci-kit Video  
FFmpeg  
FFmpy  
# Tips for training    
  This should run on any decent machine (I'm using gtx 1080) with image size of 256 and batch size of 1. Also, it is worth nothing that you should not have to increase the image size. The model should be trained on 256x256 images but it should be able to work with any larger images as well afterwards. 
  Hyperparameters are all set in nn/constants.py as being the same as the original paper, though you may want to tweak the losses. I provide some further explanation for the losses in nn/loss_funcs.py. For notation, I refer to images in the two domains as A and B respectively with G_AB being the generator that converts A images into B images and G_BA being its opposite. D_A refers to the discriminator judging A images and D_B the same for B images. Also any variable capitalized is likely from nn/constants.py.   
# Videos  
  vids2data provides functionality for converting videos into compressed (gz) npy files. Simply put videos you would like converted into source_videos folder, run vids2data/convert.py (may want to check out vids2data/constants.py). Now from output_data, take the two files you would file to train on, name them A_data.npy.gz and B_data.npy.gz respectively. To use default example_train.py place in root folder. For example_convert.py, rename A_test.npy.gz or B_test.npy.gzm
  # Usage  
  example_train.py and example_convert.py should show all functions needed to train on video data and to convert videos. During training parameters are saved in params.pt file. Sometimes stuff here takes a lot of time since its computationally expensive. Videos with larger frames than 256x256 take a while. You can speed all this up by using videos that are: shorter (You can index data arrays in the example to get a smaller slice), smaller, have lower framerates. Also, to prevent out of memory errors, I do model I/O in chunks for converting. If you increase the chunk size it will take more memory but run a bit faster.
