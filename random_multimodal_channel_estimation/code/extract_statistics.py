'''
Script to extract channels from raytrace data
Authors:
Wesin Ribeiro
Marcus Yuichi
2019
'''
from scipy.io import loadmat, savemat
import numpy as np
import argparse

##################################
### Script configuration
##################################

def extract_statistics(data_folder, dataset):  
    #inputFile = dataset + '/ray_tracing_data_s008_carrier60GHz/rosslyn_mobile_60GHz_ts0.1s_V_Lidar_e*.hdf5'
    inputFile = dataset + '/ray_tracing_data_s007_carrier60GHz/beijing_mobile_60GHz_ts1s_VP_e*.hdf5'
    inputType = 'h5' # mat or h5
    Nr = 8
    Nt = 16

    #################################
    #### Start processing
    #################################

    import h5py
    import os
    import sys
    
    episodeRays = h5py.File(inputFile.replace('*', str(0)), 'r').get('allEpisodeData')
    numUsers = episodeRays.shape[1]
    numScenes = episodeRays.shape[0]
    numRays = episodeRays.shape[2]
    print(numRays,numScenes,numUsers)
    numEpisodes = 0
    while True:
        if not os.path.isfile(inputFile.replace('*',str(numEpisodes))):
            break
        numEpisodes += 1
        
    Ht = np.zeros((numEpisodes,numScenes,numUsers,Nr,Nt))
    
    currentEpisode = 0
    numChannels = 0
    numValidChannels = 0
    print('Processing ...')
    print(numEpisodes)
    for iEpisode in range(numEpisodes):
        
        print('\r\t Episode: %d'%iEpisode, end='')
        currentFile = inputFile.replace('*', str(iEpisode))
        
        episodeRays = h5py.File(currentFile, 'r').get('allEpisodeData')
        
        for iScene in range(numScenes):
            Huser = np.zeros((numUsers,Nr,Nt))
            gain_in_dB = []
            AoD_az = []
            AoA_az = []
            rays = []
            for iUser in range(numUsers):
                numChannels += 1
                # Check valid rays for user
                numUserRays = 0
                for iRay in range(numRays):
                    if np.isnan(episodeRays[iScene,iUser,iRay,:]).all():
                        break
                    numUserRays += 1
                
                if numUserRays == 0:
                    Huser[iUser,:,:] = np.nan*np.zeros((1,Nr,Nt))
                else:
                    numValidChannels += 1
                    rays.append(numUserRays)
                    gain_in_dB.append(np.mean(episodeRays[iScene, iUser, 0:numUserRays, 0]))
                    AoD_az.append(np.mean(episodeRays[iScene, iUser, 0:numUserRays, 3]))
                    AoA_az.append(np.mean(episodeRays[iScene, iUser, 0:numUserRays, 5]))
                    
                                            
                        
    print('### Finished processing channels')
    print('\t %d Total of channels'%numChannels)
    print('\t %d Total of valid channels'%numValidChannels)
    print(f'\t gain >>>> mu: {np.mean(gain_in_dB)}, std: {np.std(gain_in_dB)}')
    print(f'\t AoD >>>> mu: {np.mean(AoD_az)}, std: {np.std(AoD_az)}')
    print(f'\t AoA >>>> mu: {np.mean(AoA_az)}, std: {np.std(AoA_az)}')
    print(f'\t rays >>>> mu: {np.mean(rays)}, std: {np.std(rays)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure the files before training the net.')
    parser.add_argument('dataset', help='Path to the Raymobtime dataset directory')
    parser.add_argument('data_folder', help='Where to place the processed data')
    args = parser.parse_args()
    extract_statistics(args.data_folder, args.dataset)
