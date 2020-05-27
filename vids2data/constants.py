

NORM_STYLE = None # TANH or SIGMOID or NONE
# NONE is almost always better as uint8 lowers file size + memory usage
# Conversions to float should be done by model with batches

VIDEO_HEIGHT = 256
VIDEO_WIDTH = 256

FRAME_SKIP = 10 # Take every n-th frame from video
# Loweres file size drastically. Not to mention, for training data
# Having every frame may be kind of pointless since in a 60fps video
# Two subsequent frames are most likely pretty much the same
# For the original paper, dataset sizes were in the scale of thousands

# Compress to gz
COMPRESS = True
