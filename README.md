# Self Driving Vehicle with Behvaior cloning

## Preprocessing
    image_norm.py - Calcuate the standardization per channel for the entire dataset
    preprocessing.py - Prepare the Images for processing by converting to YUV image, convert image to [0, 1], and steering to [-1, 1] + handle files conversion for simple handling

## Model
    nn.py - Model that was taken from the Nvidia paper: https://arxiv.org/pdf/1604.07316v1
    
    1152 Neurons for the entire model, however I concated an extra neuron to handle steering

    3 EPOCH with ~10,000 front facing images
    90% accuracy for model 

    Note: Augmentation for Images hasn't been implemented

## Training & Validation
    training.py - Where training and validation occurs for the model 

    python3 training.py and choose whether to train or validate a model
