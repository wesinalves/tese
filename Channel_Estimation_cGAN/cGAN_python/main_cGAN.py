import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from GAN.cGANGenerator import Generator
from GAN.cGANDiscriminator import Discriminator
from GAN.data_preprocess import load_image_train, load_image_test, load_image_test_y
from tempfile import TemporaryFile
from scipy.io import loadmat, savemat
import datetime
import h5py
import hdf5storage
import skfuzzy as fuzz
from mimo_channels_data_generator2 import RandomChannelMimoDataGenerator
import argparse

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.compat.v1.enable_eager_execution(config=config)
gpus = tf.config.experimental.list_physical_devices('GPU') 
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
layers = tf.keras.layers



# data path
path = "../Data_Generation_matlab/Gan_Data/Gan_0_dBIndoor2p4_64ant_32users_8pilot.mat"
path_test = "../Data_Generation_matlab/Gan_Data/Gan_0_dBIndoor2p4_64ant_32users_8pilot.mat"
# path = "../Data_Generation_matlab/Gan_Data/channel_data.mat"
# path_test = "../Data_Generation_matlab/Gan_Data/channel_data.mat"


# batch = 1 produces good results on U-NET
BATCH_SIZE = 1              

# model
generator = Generator()
discriminator = Discriminator()
# optimizer
generator_optimizer = tf.compat.v1.train.AdamOptimizer(2e-4, beta1=0.5)
discriminator_optimizer = tf.compat.v1.train.RMSPropOptimizer(2e-5)
#discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(2e-4, beta1=0.5)

"""
Discriminator loss:
The discriminator loss function takes 2 inputs; real images, generated images
real_loss is a sigmoid cross entropy loss of the real images and an array of ones(since the real images)
generated_loss is a sigmoid cross entropy loss of the generated images and an array of zeros(since the fake images)
Then the total_loss is the sum of real_loss and the generated_loss

Generator loss:
It is a sigmoid cross entropy loss of the generated images and an array of ones.
The paper also includes L2 loss between the generated image and the target image.
This allows the generated image to become structurally similar to the target image.
The formula to calculate the total generator loss = gan_loss + LAMBDA * l2_loss, where LAMBDA = 100. 
This value was decided by the authors of the paper.
"""


def discriminator_loss(disc_real_output, disc_generated_output):
    """disc_real_output = [real_target]
       disc_generated_output = [generated_target]
    """
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(disc_real_output), logits=disc_real_output)  # label=1
    generated_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(disc_generated_output), logits=disc_generated_output)  # label=0
    total_disc_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(generated_loss)
    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target, l2_weight=100):
    """
        disc_generated_output: output of Discriminator when input is from Generator
        gen_output:  output of Generator (i.e., estimated H)
        target:  target image
        l2_weight: weight of L2 loss
    """
    # GAN loss
    gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(disc_generated_output), logits=disc_generated_output)
    # L2 loss
    l2_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = tf.reduce_mean(gen_loss) + l2_weight * l2_loss
    return total_gen_loss


def generated_image(model, test_input, tar, t=0):
    """Dispaly  the results of Generator"""
    prediction = model(test_input)
    #plt.figure(figsize=(15, 15))
    display_list = [np.squeeze(test_input[:,:,:,0]), np.squeeze(tar[:,:,:,0]), np.squeeze(prediction[:,:,:,0])]
    

    title = ['Input Y', 'Target H', 'Prediction H']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i]) 
        plt.axis("off")
    plt.savefig(os.path.join("generated_img", "img_"+str(t)+".png"))



def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image)                      # input -> generated_target
        #generator.summary()
        disc_real_output = discriminator(target)  # [input, target] -> disc output
        #discriminator.summary()
        disc_generated_output = discriminator(gen_output)  # [input, generated_target] -> disc output
        #print("*", gen_output.shape, disc_real_output.shape, disc_generated_output.shape)
        
        

        # calculate loss
        gen_loss = generator_loss(disc_generated_output, gen_output, target)   # gen loss
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)  # disc loss

    # gradient
    generator_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # apply gradient
    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))
    
    #print(disc_loss.numpy())
    return gen_loss, disc_loss


def train(epochs):
    nm = []
    ep = []
    start_time = datetime.datetime.now()
    num_test_examples = 250
    realim, inpuim = load_image_test_y(path_test, 250)
    
    for epoch in range(epochs):
        print("-----\nEPOCH:", epoch)           
        # train
        for bi, (target, input_image) in enumerate(load_image_train(path)):
            elapsed_time = datetime.datetime.now() - start_time
            gen_loss, disc_loss = train_step(input_image, target)
            print("B/E:", bi, '/' , epoch, ", Generator loss:", gen_loss.numpy(), ", Discriminator loss:", disc_loss.numpy(), ', time:',  elapsed_time)
        # # generated and see the progress
        # for bii, (tar, inp) in enumerate(load_image_test(path_test)):            
        #     if bii == 100:
        #         generated_image(generator, inp, tar, t=epoch+1  )

        #save checkpoint
        #if (epoch + 1) % 2 == 0:
        ep.append(epoch + 1)
        #generator.save_weights(os.path.join("weights/generator_"+str(epoch)+".h5"))
        #discriminator.save_weights(os.path.join("weights/discriminator_"+str(epoch)+".h5"))        
        prediction = generator(inpuim)        
        error = realim - prediction
        mseTest = np.mean(error[:] ** 2)
        print("overall MSE = ", mseTest)
        mean_nmse = mseTest / (8 * 64) # (Nr * Nt)
        print("overall NMSE = ", mean_nmse)
        nmses = np.zeros((num_test_examples,))
        for i in range(num_test_examples):
            this_H = realim[i]
            this_error = error[i]
            nmses[i] = np.mean(this_error[:] ** 2) / np.mean(this_H[:] ** 2)

        #nmse_db = fuzz.nmse(np.squeeze(realim), np.squeeze(prediction))
        nmses_db = 10 * np.log10(nmses)
        nm.append(np.mean(nmses_db))
        print(nmses_db)
        

        nm.append(fuzz.nmse(np.squeeze(realim), np.squeeze(prediction)))
        
        # if epoch == epochs-1:
        #     nmse_epoch = TemporaryFile()
        #     np.save(nmse_epoch, nm)
        
        # Save the predicted Channel 
        # matfiledata = {} # make a dictionary to store the MAT data in
        # matfiledata[u'predict_Gan_0_dB_Indoor2p4_64ant_32users_8pilot'] = np.array(prediction) # *** u prefix for variable name = unicode format, no issues thru Python 3.5; advise keeping u prefix indicator format based on feedback despite docs ***
        # hdf5storage.write(matfiledata, '.', 'Results/Eest_cGAN_'+str(epoch + 1)+'_0db_Indoor2p4_64ant_32users_8pilot.mat', matlab_compatible=True)
        
        # plt.figure()
        # plt.plot(ep,nm,'^-r')
        # plt.xlabel('Epoch')
        # plt.ylabel('NMSE')
        # plt.show();
    
    generator.save_weights(os.path.join("weights/generator_raymob.h5"))
    
    return nm, ep

def test():
    ###############################################################################
    # Channel data configuration
    # fix random seed for reproducibility
    seed = 1
    np.random.seed(seed)

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

    num_test_examples = 250    
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

    #multimodal
    SNRdB_values = np.arange(-21, 22, 3)
    training_generator.randomize_SNR = False
    training_generator.method = "manual" 

    all_nmse_db_average = np.zeros((SNRdB_values.shape))
    all_nmse_db_min = np.zeros((SNRdB_values.shape))
    all_nmse_db_max = np.zeros((SNRdB_values.shape))
    
    model = Generator()    
    X_channel_test, outputs = training_generator.get_examples(1)
    input_images = X_channel_test.reshape((-1,X_channel_test.shape[1], X_channel_test.shape[2]//2, 2))
    tmp = model(input_images)
    model.load_weights('weights/generator_raymob.h5')
    #model.load_weights('weights/generator2.h5')


    it = 0
    for SNRdB in SNRdB_values:
        training_generator.SNRdB = SNRdB
        # get rid of the last example in the training_generator's memory (flush it)
        # X_channel_test, outputs = training_generator.get_examples(1)
        # # now get the actual examples:
        X_channel_test, outputs = training_generator.get_examples(num_test_examples)
        real_images = outputs.reshape((-1,outputs.shape[1]//2, outputs.shape[2], 2))
        input_images = X_channel_test.reshape((-1,X_channel_test.shape[1], X_channel_test.shape[2]//2, 2))

        #real_images, inpuim = load_image_test_y(path)   
        predictedOutput = model(input_images)
        num_test_examples = len(real_images)
        
        predictedOutput = model(input_images)        
        print(predictedOutput.shape)
        print(real_images.shape)
        #predictedOutput = model.predict(input_images)
        error = real_images - predictedOutput
        mseTest = np.mean(error[:] ** 2)
        print("overall MSE = ", mseTest)
        mean_nmse = mseTest / (Nr * Nt)
        print("overall NMSE = ", mean_nmse)
        nmses = np.zeros((num_test_examples,))
        for i in range(num_test_examples):
            this_H = real_images[i]
            this_error = error[i]
            nmses[i] = np.mean(this_error[:] ** 2) / np.mean(this_H[:] ** 2)

        print("NMSE: mean", np.mean(nmses), "min", np.min(nmses), "max", np.max(nmses))
        nmses_db = 10 * np.log10(nmses)
        print(
            "NMSE dB: mean",
            np.mean(nmses_db),
            "min",
            np.min(nmses_db),
            "max",
            np.max(nmses_db),
        )

        all_nmse_db_average[it] = np.mean(nmses_db)
        all_nmse_db_min[it] = np.min(nmses_db)
        all_nmse_db_max[it] = np.max(nmses_db)

        it += 1

        del X_channel_test
        del outputs
        #del mat 
    
    logdir = 'Results/'

    output_filename = (
        f"all_nmse_cgan.txt"
    )
    output_filename = os.path.join(logdir, output_filename)
    np.savetxt(output_filename, (all_nmse_db_average, all_nmse_db_min, all_nmse_db_max))
    print("Wrote file", output_filename)
    print("*******************\n{}".format(np.mean(all_nmse_db_average)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Configure options')
    parser.add_argument('--step', nargs='*', default=['train'],
        choices=['train','test'], help='which step: train or test', required=True)
    args = parser.parse_args()
    logdir = 'Results/'

    if 'train' in args.step:
        # train
        print('start training')
        #train(epochs=20)        
        nm, ep = train(epochs=20)        
        print('end training')        
        output_filename = os.path.join(logdir, 'history_train.txt')
        np.savetxt(output_filename, (nm))
        print('avarage nmse: ', np.mean(nm))
    if 'test' in args.step:
        print('start testing')
        test()
        print('end testing')
    
    # plt.figure()
    # plt.plot(ep,nm,'^-r')
    #plt.xlabel('Epoch')
    #plt.ylabel('NMSE')
    # plt.show()