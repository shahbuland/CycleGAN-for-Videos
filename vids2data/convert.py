import numpy as np
import skvideo.io as skv
import os
from constants import *
from vid_io import save

# This function converts video (given path) into a npy file
# Can specify a frame skip as well to not return all frames in video

def vid2arr(source,destination,frame_skip=FRAME_SKIP):
        # String for rescaling video
        size_str = str(VIDEO_HEIGHT)+"x"+str(VIDEO_WIDTH)
        # Read in video as array
        A = skv.vread(source,outputdict={"-s":size_str})
        # Preprocessing steps
        if NORM_STYLE == "SIGMOID":
                A = A.astype(np.float64)
                A = A/255
        elif NORM_STYLE == "TANH":
                A = A.astype(np.float64)
                A = A/127.5 - 1
        # A is now a preprocessed array

        if frame_skip is not None:
                A = A[::frame_skip]

        # Use vid_io's save or np default save
        if COMPRESS:
                save(destination, A)
        else:
                np.save(destination, A)

# Iterate through all files in source videos
for video_path in os.listdir("./source_videos/"):
        if video_path.endswith(".mp4"):
                vid2arr("./source_videos/"+video_path,"./output_data/"+video_path+".npy")
	
