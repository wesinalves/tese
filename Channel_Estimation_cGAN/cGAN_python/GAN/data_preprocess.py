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

#sys.path.insert(1, '/home/Documents/Doutorado/tese/baseline/Channel_Estimation_cGAN/cGAN_python')
sys.path.insert(1, '/home/wesinalves/cGAN_python')
from mimo_channels_data_generator2 import RandomChannelMimoDataGenerator


# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.compat.v1.enable_eager_execution(config=config)
gpus = tf.config.experimental.list_physical_devices('GPU') 
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

layers = tf.keras.layers

###############################################################################
# Channel data configuration
# fix random seed for reproducibility
seed = 1
np.random.seed(seed)

# training parameters
epochs = 100

# Parameters
global Nt
global Nr
Nt = 64  # num of Rx antennas, will be larger than Nt for uplink massive MIMO
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

num_test_examples = 10000
channel_train_input_file = "../Data_Generation_matlab/Gan_Data/siso_fixed.mat"
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

training_generator.method = "manual"        
train_data = training_generator.get_examples(2000)
inputs, outputs = train_data
real_image = outputs.reshape((-1,outputs.shape[1]//2, outputs.shape[2], 2))
input_image = inputs.reshape((-1,inputs.shape[1], inputs.shape[2]//2, 2))
    


def load_image_train(path, batch_size = 1):
    """load, jitter, and normalize"""
    # with h5py.File(path, 'r') as file:
    #     real_image = np.transpose(np.array(file['output_da']))
          
    # with h5py.File(path, 'r') as file:
    #     input_image = np.transpose(np.array(file['input_da']))        
    

    print(real_image.shape)    
    print(input_image.shape)    
        
    SIZE_IN= real_image.shape
    list_im=list(range(0, SIZE_IN[0]))

    batch_im = random.sample(list_im,SIZE_IN[0])
    real_h = real_image[batch_im,:,:,:]
    input_y = input_image[batch_im,:,:,:]

    n_batches = int(SIZE_IN[0] / batch_size)
    
    for i in range(n_batches-1):
        imgs_A = real_h[i*batch_size:(i+1)*batch_size]
        imgs_B = input_y[i*batch_size:(i+1)*batch_size]
        
    
        yield imgs_A, imgs_B
    
        



def load_image_test(path, batch_size = 1):
       
    # with h5py.File(path, 'r') as file:
    #     real_image = np.transpose(np.array(file['output_da_test']))

        
    # with h5py.File(path, 'r') as file:
    #     input_image = np.transpose(np.array(file['input_da_test']))
    training_generator.randomize_SNR = False
    training_generator.SNRdB = 0
    inputs, outputs = training_generator.get_examples(500)
    real_image = outputs.reshape((-1,outputs.shape[1]//2, outputs.shape[2], 2))
    input_image = inputs.reshape((-1,inputs.shape[1], inputs.shape[2]//2, 2))
        
    SIZE_IN= real_image.shape

    
    n_batches = int(SIZE_IN[0] / batch_size)
    
    for i in range(n_batches-1):
        imgs_A = real_image[i*batch_size:(i+1)*batch_size]
        imgs_B = input_image[i*batch_size:(i+1)*batch_size]
        
    
        yield imgs_A, imgs_B
        
def load_image_test_y(path, size=500):
       
    # with h5py.File(path, 'r') as file:
    #     real_image = np.transpose(np.array(file['output_da_test']))

        
    # with h5py.File(path, 'r') as file:
    #     input_image = np.transpose(np.array(file['input_da_test']))
    training_generator.randomize_SNR = False    
    inputs, outputs = training_generator.get_examples(1)
    inputs, outputs = training_generator.get_examples(size)    
    real_image = outputs.reshape((-1,outputs.shape[1]//2, outputs.shape[2], 2))
    input_image = inputs.reshape((-1,inputs.shape[1], inputs.shape[2]//2, 2))        
        
    
    return real_image, input_image
