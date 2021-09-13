'''
Script to extract channels from raytrace data
Authors:
Wesin Ribeiro
Marcus Yuichi
2019
'''
from scipy.io import loadmat, savemat
import numpy as np
import mimo_channels

##################################
### Script configuration
##################################

def processChannelRandomGeo(data_folder, dataset):  
    Nr = 8
    Nt = 16
    #outputFile = data_folder + '/channel_data/random_ray100_angle50.mat'
    outputFile = data_folder + '/channel_data/stat_beijing.mat'

    #################################
    #### Start processing
    #################################

          
    import os
    import sys
    
    numRays = 25
    numEpisodes = 10000    
    Ht = np.zeros((numEpisodes, Nr,Nt))
    
    numChannels = 0
    numValidChannels = 0
    print('Processing ...')
    print(numEpisodes)
    for i in range(numEpisodes):
        numChannels += 1
        # Check valid rays for user
        numValidChannels += 1
        #gain = np.random.randn(numRays) + i*np.random.randn(numRays)
        #gain_in_dB = 20*np.log10(np.abs(gain))
        gain_in_dB = -158.12771606445312 + 12.007607460021973 * np.random.randn(numRays)
        #nominal_values = [-42.1414, 46.4871, 56.1069, 24.0976]
        angle_spread = 50
        #AoD_az = nominal_values[np.random.randint(0,3)] + angle_spread*np.random.randn(numRays)
        AoD_az = -4.810908317565918 + 62.9975471496582*np.random.randn(numRays)
        #AoA_az = nominal_values[np.random.randint(0,3)] + angle_spread*np.random.randn(numRays)
        AoA_az = -3.3930130004882812 + 36.421607971191406*np.random.randn(numRays)
        # AoD_az = 360*np.random.uniform(size=numRays);
        # AoA_az = 360*np.random.uniform(size=numRays);
        # RxAngle = episodeRays[iScene, i, 0:numUserRays, 8][0]
        # RxAngle = RxAngle + 90.0
        # if RxAngle > 360.0:
        #     RxAngle = RxAngle - 360.0
        # #Correct ULA with Rx orientation
        # AoA_az = - RxAngle + AoA_az #angle_new = - delta_axis + angle_wi;
        #phase = np.angle(gain)*180/np.pi
            
        # Ht[i,:,:] = mimo_channels.getNarrowBandULAMIMOChannel(\
        #     AoD_az, AoA_az, gain_in_dB, Nt, Nr, pathPhases=phase)
        Ht[i,:,:] = mimo_channels.getNarrowBandULAMIMOChannel(\
            AoD_az, AoA_az, gain_in_dB, Nt, Nr)
            
    
        
    print('### Finished processing channels')
    print('%d Total of channels'%numChannels, end='')
    print('%d Total of valid channels'%numValidChannels, end='')

    
    # permute dimensions before reshape: scenes before episodes
    # found out np.moveaxis as alternative to permute in matlab
    Harray = np.reshape(Ht, (numEpisodes, Nr, Nt))
    Hvirtual = np.zeros(Harray.shape, dtype='complex128')
    scaling_factor = 1 / np.sqrt(Nr * Nt)

    for i in range(numEpisodes):
        m = np.squeeze(Harray[i,:,:])
        Hvirtual[i,:,:] = scaling_factor * np.fft.fft2(m)

    savemat(outputFile, {'Harray': Harray, 'Hvirtual':Hvirtual})


# estatística de rosslyn
# Episode: 2085### Finished processing channels
# 20860 Total of channels
# 11194 Total of valid channels
# gain >>>> mu: -131.4732666015625, std: 4.535895824432373
# AoD >>>> mu: 20.396900177001953, std: 39.6958122253418
# AoA >>>> mu: 3.9949631690979004, std: 71.52169036865234
# rays >>>> mu: 25, std: 0


# estatística de beijing
# Episode: 49### Finished processing channels
# 20000 Total of channels
# 14532 Total of valid channels
# gain >>>> mu: -158.12771606445312, std: 12.007607460021973
# AoD >>>> mu: -4.810908317565918, std: 62.9975471496582
# AoA >>>> mu: -3.3930130004882812, std: 36.421607971191406
# rays >>>> mu: 25, std: 0

