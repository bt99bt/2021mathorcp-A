import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

x = np.array(pd.read_csv('x.csv').drop('Unnamed: 0',axis = 1))[20000:,]
y = np.array(pd.read_csv('y.csv').drop('carid',axis = 1))[20000:]

model = load_model('model.h5')

y_pre = model.predict(x)

y_pre = np.array(y_pre)
y = np.array(y).reshape(-1,1)

ape = np.abs(y-y_pre)/y
mape = np.sum(np.abs(y-y_pre)/y)/len(y)
accuracy = (np.sum(np.abs(y-y_pre)/y <= 0.05))/len(y)

ans = pd.concat([pd.DataFrame(y), pd.DataFrame(y_pre)],axis=1)
ans.columns = ['real_price', 'pre_price']

print(mape)
print(accuracy)
print(0.2*(1-mape)+0.8*accuracy)

ans.to_csv('ans.csv')