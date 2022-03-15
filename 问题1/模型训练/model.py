import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

x = np.array(pd.read_csv('x.csv'))[:20000,]
y = np.array(pd.read_csv('y.csv'))[:20000]

model = keras.models.Sequential()
model.add(keras.layers.Dense(600, activation = 'relu'))
model.add(keras.layers.Dense(600, activation = 'relu'))
model.add(keras.layers.Dense(300, activation = 'relu'))
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(1,activation = 'relu'))

model.compile(loss ='mse', optimizer = 'adam')
history = model.fit(x, y, epochs = 1000, batch_size = 500)
model.save('model.h5')
pd.DataFrame(history.history).plot()
plt.show()


