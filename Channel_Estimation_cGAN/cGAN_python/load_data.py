import scipy
import scipy.misc
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import  random
from scipy.io import loadmat
import h5py
import time
import tensorflow as tf
import sys

sys.path.insert(1, '/home/Documents/Doutorado/tese/baseline/Channel_Estimation_cGAN/cGAN_python')
from mimo_channels_data_generator2 import RandomChannelMimoDataGenerator


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
layers = tf.keras.layers

###############################################################################
# Channel data configuration
# fix random seed for reproducibility
seed = 1
np.random.seed(seed)

# Parameters
global Nt
global Nr
Nt = 16  # num of Rx antennas, will be larger than Nt for uplink massive MIMO
Nr = 8  # num of Tx antennas
# the sample is a measurement of Y values, and their collection composes an example. The channel estimation
min_randomized_snr_db = -1
max_randomized_snr_db = 1

# must be done per example, each one having a matrix of Nr x numSamplesPerExample of complex numbers
numSamplesPerExample = 256  # number of channel uses, input and output pairs
# if wants a gradient calculated many times with same channel
numExamplesWithFixedChannel = 1
numSamplesPerFixedChannel = (
    numExamplesWithFixedChannel * numSamplesPerExample
)  # coherence time
# obs: it may make sense to have the batch size equals the coherence time
batch = 1  # numExamplesWithFixedChannel

num_test_examples = 2000
channel_train_input_file = "../Data_Generation_matlab/Gan_Data/channel_data.mat"
print("Reading dataset... ",channel_train_input_file)
method = "manual"

# Generator
training_generator = RandomChannelMimoDataGenerator(
    batch_size=batch,
    Nr=Nr,
    Nt=Nt,
    # num_clusters=num_clusters,
    numSamplesPerFixedChannel=numSamplesPerFixedChannel,
    # numSamplesPerExample=numSamplesPerExample, SNRdB=SNRdB,
    numSamplesPerExample=numSamplesPerExample,
    # method='random')
    method=method,
    file = channel_train_input_file
)
if True:
    training_generator.randomize_SNR = True
    training_generator.min_randomized_snr_db = min_randomized_snr_db
    training_generator.max_randomized_snr_db = max_randomized_snr_db
else:
    training_generator.randomize_SNR = True
    training_generator.SNRdB = 0

##############################################################################
# Model configuration
##############################################################################

#multimodal
training_generator.randomize_SNR = False
training_generator.method = "manual"        
    


def load_image_train():
    """load, jitter, and normalize"""
    inputs, outputs = training_generator.get_examples(9234)
    # real_image = outputs.reshape((-1,outputs.shape[1]//2, outputs.shape[2], 2))
    # input_image = inputs.reshape((-1,inputs.shape[1], inputs.shape[2]//2, 2))

    return outputs, inputs        

def load_image_test():
       
    inputs, outputs = training_generator.get_examples(1960)
    # real_image = outputs.reshape((-1,outputs.shape[1]//2, outputs.shape[2], 2))
    # input_image = inputs.reshape((-1,inputs.shape[1], inputs.shape[2]//2, 2))        
    return outputs, inputs
