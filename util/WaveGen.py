import random
import numpy as np
import scipy.signal

def generateWave(quake, train):
    cor = np.ones(1000)
    fal = np.zeros(1000)
    y = np.random.rand(1600)

    quake = False
    train = False

    if quake == True:
        win = scipy.signal.hann(3)  
    else:
        win = scipy.signal.hann(30)
        
    filtered = scipy.signal.convolve(y, win, mode='same') / sum(win)

    filtered = np.multiply(filtered,random.randint(1, 100))

    filtered = filtered[300:-300]
    delta = np.diff(filtered)
    delta = np.resize(delta, 1000)

    if train == True:
        if quake == False:
            filtered = np.vstack((filtered, delta, fal)).T
        else:
            filtered = np.vstack((filtered, delta, cor)).T
    else:
        filtered = np.vstack((filtered, delta)).T
        
    with open("datasets/NoiseArrayTestNoise.txt", "ab") as f:
        np.savetxt(f, filtered)
    
    if train == True:
        lines = open("datasets/NoiseArrayTrain.txt").readlines()
        random.shuffle(lines)
        open("datasets/NoiseArrayTrain.txt", 'w').writelines(lines)

generateWave()
