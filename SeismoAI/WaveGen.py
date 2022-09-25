import random
import numpy as np
import scipy.signal

def generateWave(quake, train):
    cor = np.ones(1600)
    fal = np.zeros(1600)
    y = np.random.rand(1600)

    if quake == True:
        win = scipy.signal.hann(3)
        filtered = scipy.signal.convolve(y, win, mode='same') / sum(win)
        delta = np.diff(filtered)
        delta = np.resize(delta, 1600)
        if train == True:
            filtered = np.vstack((filtered, delta, cor)).T
        elif train == False:
            filtered = np.vstack((filtered, delta)).T
    
    elif quake == False:
        win = scipy.signal.hann(30)
        filtered = scipy.signal.convolve(y, win, mode='same') / sum(win)
        delta = np.diff(filtered)
        delta = np.resize(delta, 1600)
        if train == True:
            filtered = np.vstack((filtered, delta, fal)).T
        elif train == False:
            filtered = np.vstack((filtered, delta)).T
        
    with open("NoiseArrayTest.txt", "ab") as f:
        np.savetxt(f, filtered)
    
    if train == True:
        lines = open("NoiseArrayTrain.txt").readlines()
        random.shuffle(lines)
        open("NoiseArrayTrain.txt", 'w').writelines(lines)

generateWave(quake = False, train = False)
generateWave(quake = True, train = False)
generateWave(quake = False, train = False)