from datetime import date
import numpy as np
import tensorflow as tf

dataset = np.loadtxt('C:/Users/witzp/Documents/SeismoAI/NoiseArrayNew.txt', delimiter=' ')

X = dataset[:,0:2]
y = dataset[:,2]

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(12, input_shape=(2,), activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=1000, batch_size=5)

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
