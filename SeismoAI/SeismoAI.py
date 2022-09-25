import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

dataset = np.loadtxt('datasets\NoiseArrayTrain.txt', delimiter=' ')  # seismograph data

class EarthquakeDetector:

    def __init__(self, 
                seismograph_count=5, 
                sample_rate_hz=4800, 
                alert_callback=False):

        json_file = open('model\model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        seismic_model = tf.keras.models.model_from_json(loaded_model_json)
        seismic_model.load_weights("model\model.h5")

        self.seismograph_count = seismograph_count
        self.alert_callback = alert_callback
        self.b = np.zeros((seismograph_count, sample_rate_hz, 1), dtype=int)  

        for x in range(seismograph_count):
            self.b[x] = (seismic_model.predict(dataset) > 0.5).astype(int)  # split dataset into number of seismographs

    pass

    def new_samples(self, seismograph_id, samples):

        fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, sharey=True)
        
        ax.plot(samples[:,0], linestyle="-", marker=".", lw=1, markersize=1)
        ax.set_title("Dataset")

        ax2.plot(samples[:,0], linestyle="-", marker=".", lw=1, markersize=1)
        ax2.plot(self.b[seismograph_id], color="r", lw=8, markersize=8, alpha=0.5)
        ax2.set_title("Earthquake Detected")

        plt.show()

    pass

if __name__ == "__main__":

    ed = EarthquakeDetector()
    ed.new_samples(seismograph_id=0, samples=dataset)


    if (ed.alert_callback == True):
            print("Earthquake confirmed by all seisemographs!")