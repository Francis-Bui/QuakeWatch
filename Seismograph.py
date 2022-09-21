import numpy as np
import random

a = [12,20,12,10,5,12,50,90,250,125,50,12,5,10,12,20,12]  # seismograph data

class EarthquakeDetector:
    def __init__(self, 
                seismograph_count=5, 
                sample_rate_hz=17, 
                alert_callback=False):

        self.seismograph_count = seismograph_count
        self.alert_callback = alert_callback
        self.b = np.zeros((seismograph_count, sample_rate_hz), dtype=int)  
        self.spikeArray = np.zeros((seismograph_count, sample_rate_hz - 1), dtype=int)  # create empty bool array in integer form
        for x in range(seismograph_count):

            self.b[x] = a  # split dataset into number of seismographs
            self.b[x] = self.b[x] * (random.randint(1, 100))  # give seismographs random scale

    pass

    def new_samples(self, seismograph_id, samples):
        passFlag = False 
        self.spikeArray[seismograph_id] = abs(np.diff(samples[seismograph_id])) > (
            sum(samples[seismograph_id]) / len(samples[seismograph_id]))  # array is true where IROC is higher than AROC
        self.spikeArray = np.rot90(self.spikeArray)  # rotate array in order to add rows
        for idx in range(len(self.spikeArray)): 
            if (self.spikeArray[idx].sum(axis=0)) >= 1 and passFlag == False:  # check if any quakes have been detected by adding columns
                print((np.sum(self.spikeArray[idx].sum(axis=0))),"out of",self.seismograph_count,"seismographs detected an earthquake",)
                passFlag = True
                if (self.spikeArray[idx].sum(axis=0)) == self.seismograph_count:  # check if all siesmographs detected a quake
                    self.alert_callback = True

        self.spikeArray = np.rot90(self.spikeArray, -1)  # reset array

    pass

if __name__ == "__main__":

    ed = EarthquakeDetector()
    ed.new_samples(seismograph_id=0, samples=ed.b)
    ed.new_samples(seismograph_id=1, samples=ed.b)
    ed.new_samples(seismograph_id=2, samples=ed.b)
    ed.new_samples(seismograph_id=3, samples=ed.b)
    ed.new_samples(seismograph_id=4, samples=ed.b)

    if (ed.alert_callback == True):
            print("Earthquake confirmed by all seisemographs!")