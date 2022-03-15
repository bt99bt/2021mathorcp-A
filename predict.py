import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

x = pd.read_csv('x.csv')
input = x.drop('carid', axis = 1)

model = load_model('model.h5')
y_pre = np.array(model.predict(np.array(input)))

ans = pd.concat([x['carid'].astype(int), pd.DataFrame(y_pre)],axis=1)
ans.columns = ['carid', 'pre_price']
ans = ans.set_index('carid')
print(ans)
ans.to_csv('附件3：估价模型结果.txt',sep = '\t',header=False)
