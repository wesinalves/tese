import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K




# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.compat.v1.enable_eager_execution(config=config)
gpus = tf.config.experimental.list_physical_devices('GPU') 
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

layers = tf.keras.layers

"""
The architecture of generator is a modified U-Net.
There are skip connections between the encoder and decoder (as in U-Net).
"""


class EncoderLayer(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides_s = 2, apply_batchnorm=True, add = False, padding_s = 'same'):
        super(EncoderLayer, self).__init__()
        initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
        conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides_s,
                             padding=padding_s, kernel_initializer=initializer, use_bias=False)        
        #conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides_s,
        #                     padding=padding_s, use_bias=False)
        ac = layers.LeakyReLU()
        self.encoder_layer = None
        if add:
            self.encoder_layer = tf.keras.Sequential([conv])
        elif apply_batchnorm:
            bn = layers.BatchNormalization()
            self.encoder_layer = tf.keras.Sequential([conv, bn, ac])
        else:
            self.encoder_layer = tf.keras.Sequential([conv, ac])

    def call(self, x):
        return self.encoder_layer(x)


class DecoderLayer(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides_s = 2, apply_dropout=False, add = False):
        super(DecoderLayer, self).__init__()
        initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
        dconv = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides_s,
                                       padding='same', kernel_initializer=initializer, use_bias=False)
        # initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
        # dconv = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides_s,
        #                                padding='same', use_bias=False)
        bn = layers.BatchNormalization()
        ac = layers.ReLU()
        self.decoder_layer = None
        
        if add:
            self.decoder_layer = tf.keras.Sequential([dconv])      
        elif apply_dropout:
            drop = layers.Dropout(rate=0.5)
            self.decoder_layer = tf.keras.Sequential([dconv, bn, drop, ac])
        else:
            self.decoder_layer = tf.keras.Sequential([dconv, bn, ac])
            
        
            

    def call(self, x):
        return self.decoder_layer(x)
    
    


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        # Resize Input
        p_layer_1 = DecoderLayer(filters=2, kernel_size=4, strides_s = 4, apply_dropout=False, add = True) 
        p_layer_2  = DecoderLayer(filters=2, kernel_size=4, strides_s = 4, apply_dropout=False, add = True)
        p_layer_3  = EncoderLayer(filters=2, kernel_size=(6,1),strides_s = (16,1), apply_batchnorm=False, add = True)
        # p_layer_7  = EncoderLayer(filters=2, kernel_size=(6,1),strides_s=(4,1), apply_batchnorm=False, add = True)
        # p_layer_6  = DecoderLayer(filters=2, kernel_size=4, strides_s = 2, apply_dropout=False, add = True)
        # p_layer_8  = DecoderLayer(filters=2, kernel_size=4, strides_s = 2, apply_dropout=False, add = True)
        # p_layer_9  = DecoderLayer(filters=2, kernel_size=4, strides_s = 2, apply_dropout=False, add = True)
                                
        self.p_layers = [p_layer_1,p_layer_2,p_layer_3]
        
        
        
        #encoder
        encoder_layer_1 = EncoderLayer(filters=64*1,  kernel_size=4, apply_batchnorm=False)   
        encoder_layer_2 = EncoderLayer(filters=64*2, kernel_size=4 )       
        encoder_layer_3 = EncoderLayer(filters=64*4, kernel_size=4 )       
        encoder_layer_4 = EncoderLayer(filters=64*8, kernel_size=4 )       
        encoder_layer_5 = EncoderLayer(filters=64*8, kernel_size=4 )       
        encoder_layer_6 = EncoderLayer(filters=64*8, kernel_size=4 )       
        self.encoder_layers = [encoder_layer_1, encoder_layer_2, encoder_layer_3, encoder_layer_4,
                               encoder_layer_5]

        # deconder
        decoder_layer_1 = DecoderLayer(filters=64*8, kernel_size=4, apply_dropout=True)   
        decoder_layer_2 = DecoderLayer(filters=64*8, kernel_size=4,apply_dropout=True)   
        decoder_layer_3 = DecoderLayer(filters=64*8, kernel_size=4,apply_dropout=True)   
        decoder_layer_4 = DecoderLayer(filters=64*4, kernel_size=4)    
        self.decoder_layers = [decoder_layer_1, decoder_layer_2, decoder_layer_3, decoder_layer_4]

       
        H_normalization_factor = np.sqrt(64 * 8) # Nt x Nr        
        initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
        #self.last = layers.Conv2D(filters=2, kernel_size=4, strides=(4,1), padding='same',
        #                                    kernel_initializer=initializer, activation='tanh')
        self.last = layers.Conv2D(filters=2, kernel_size=4, strides=(16,1), padding='same', activation='tanh', kernel_initializer=initializer)
        self.normalization = layers.Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1))
        #self.reshape = layers.Reshape((8,64,2))

    def call(self, x):
        # pass the encoder and record xs
        for p_layer in self.p_layers:
            x = p_layer(x)

        encoder_xs = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encoder_xs.append(x)        
        encoder_xs = encoder_xs[:-1][::-1]    # reverse
        assert len(encoder_xs) == 4

        # pass the decoder and apply skip connection
        for i, decoder_layer in enumerate(self.decoder_layers):            
            x = decoder_layer(x)            
            x = tf.concat([x, encoder_xs[i]], axis=-1)     # skip connect
            #x = tf.add(x, encoder_xs[i])     # skip connect
        x = self.last(x)        
        x = self.normalization(x) # last with normalization        

        #return self.last(x)        
        return x        


