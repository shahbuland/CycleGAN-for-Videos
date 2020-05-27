# Dataset info

IMG_SIZE = 256
CHANNELS = 3

# Hyperparams
LEARNING_RATE = 2e-4

# Training

# Weights for loss functions
CYCLE_WEIGHT = 10
IDT_WEIGHT = 0.5 * CYCLE_WEIGHT
ADV_WEIGHT = 1

BATCH_SIZE = 1 # Batch size for training
USE_CUDA = True
SAMPLE_INTERVAL = 10 # Interval at which to draw samples
CHECKPOINT_INTERVAL = 10 # Interval at which to save checkpoints
N_CRITIC = 1 # If you want to train discriminator more than generator
LOAD_CHECKPOINTS = True
