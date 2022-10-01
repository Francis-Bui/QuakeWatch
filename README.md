# QuakeWatch
🌎 AI powered seismograph, trained to detect earthquakes at any scale. 🧠

![GUI Demo](https://raw.githubusercontent.com/Francis-Bui/QuakeWatch/main/png/GUI.png)

This repo includes the dataset generator, model trainer, datasets, and main form.

This AI built off of tensorflow and keras is able to detect an earthquake with a 98.91% accuracy.

![training stats](https://raw.githubusercontent.com/Francis-Bui/QuakeWatch/main/png/training.png)

It is trained with specially generated waveforms that simulate idle seismic noise and active seismic noise at random scales. It analyzes the delta between each interval in the dataset and the raw dataset information as well. It then uses a binary cross entropy system to compare it's answer against the training dataset.

The main form plots the raw data along with where the AI detected an earthquake. This data is cross-checked "n" amount of times against itself to avoid false positives.
