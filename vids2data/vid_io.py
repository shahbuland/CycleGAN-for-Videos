import skvideo.io as skv
import gzip
import numpy as np

# Function to convert np arrays into video files
# Assume N x H x W x Channels shape
# Takes arr and path for video to be put in
# Adjust the speed_mult value if videos are sped up or slowed
# (higher slows it down)
def arr2vid(arr, path, fps = '20', speed_mult = '2'):
    assert(len(arr.shape) == 4)

    # Compress output vid? makes writing to a video signifigantly faster
    compress = False

    speed = 'setpts=' + speed_mult + '*PTS'
    if compress:
        outdict = {'-r': fps, '-vcodec' : 'libx265', '-crf' : '30', '-filter:v' : speed}
    else:
        outdict = {'-r': fps, '-filter:v' : speed}
        
    skv.vwrite(path, arr, outputdict = outdict)

# Load array from gz
def load(path):
    f = gzip.GzipFile(path, "r")
    res = np.load(f)
    f.close()
    return res

# Save array to gz
def save(name, A):
    f = gzip.GzipFile(name+".gz", "w")
    np.save(file = f, arr = A)
    f.close()
    
    
