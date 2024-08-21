import os

## Setup
BACH_SIZE = 4
BACH_SIZE_TEST = 20
EPOCHS = 50

## Optimizer
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

## Scheduler
STEP_SIZE = 10
GAMMA = 0.1 

## Dropout
DROPOUT_PROB = 0.1

## Early stopping
PATIENCE_EARLY_STOPPING=3

### Folder
DIRECTORY = f"runs/DeepLabV3"
DIRECTORY_CHECKPOINTS = f"runs/DeepLabV3/checkpoints"
HISTORY_PATH = os.path.join(DIRECTORY, 'deeplabv3_history.pickle')
#METRICS_PATH = os.path.join(DIRECTORY, 'unet_metrics.pickle')
