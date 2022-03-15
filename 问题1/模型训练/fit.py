import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


x = np.array(pd.read_csv('x.csv').drop('Unnamed: 0',axis =1))[:20000,]
y = np.array(pd.read_csv('y.csv').drop('carid',axis =1))[:20000]

model = load_model('model.h5')
model.compile(loss ='mse', optimizer = 'adam')

history = model.fit(x, y, epochs = 1000, batch_size = 500)
model.save('model.h5')
pd.DataFrame(history.history).plot()
plt.show()