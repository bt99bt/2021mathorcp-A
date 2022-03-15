import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

filepath = '原始数据——读取.csv'
df_data_pool = pd.read_csv(filepath, engine='python')

# 日期取年函数
def time_series_to_num(time_series):
    '''函数接受pd.Series，返回pd.Series'''
    list = []
    for i in time_series:
        if type(i) != str:
            list.append(i)
        else:
            i = i[:4]
            list.append(i)
    return pd.Series(list)
# aF12处理函数
def aF12(series):
    '''函数接受pd.Series，返回np.array'''
    a = []
    b = []
    c = []
    for i in series:
        if type(i) != str:
            a.append(np.nan)
            b.append(np.nan)
            c.append(np.nan)
        else :
            list = i.split('*')
            a.append(list[0])
            b.append(list[1])
            c.append(list[2])
    return np.array(a), np.array(b), np.array(c)
# aF11处理函数
def aF11(series):
    '''函数接受pd.Series，返回np.array'''
    list = []
    for i in series:
        if type(i) == str:
            if len(i) < 4:
                    list.append(eval(i))
            else:
                    list.append(9)
        else:
            list.append(i)
    return np.array(list)
# 众数填充函数
def fillna_mode(series):
    '''函数接受pd.Series，返回pd.Series'''
    mode = series.mode()[0]
    series = series.fillna(mode)
    return series
# 平均数填充函数
def fillna_average(series):
    '''函数接受pd.Series，返回pd.Series'''
    average = series.mean()
    series = series.fillna(average)
    return series
# aF13处理函数(日期取年)
def aF13(series):
    list = []
    for i in series:
        if i == np.nan:
            list.append(i)
        else:
            list.append(str(i)[0:4])
    return pd.Series(list)
# 定义归一化函数
def minmaxscaler(series):
    x = np.array(series)
    return pd.Series((x-x.min())/(x.max() - x.min()))

#原始数据所有特征
features_pool = ['carid', 'tradeTime', 'brand', 'serial', 'model',
                 'mileage', 'color', 'cityId', 'carCode', 'transferCount',
                 'seatings', 'registerDate', 'licenseDate', 'country', 'makertype', 'modelyear',
                 'displacement', 'gearbox', 'oiltype', 'newprice', 'aF1', 'aF2', 'aF3',
                 'aF4', 'aF5', 'aF6', 'aF7', 'aF8', 'aF9', 'aF10', 'aF11', 'aF12', 'aF13', 'aF14', 'aF15',
                ]

# 日期取年
for i in ['tradeTime', 'registerDate', 'licenseDate', 'aF7', 'aF15']:
    df_data_pool[i] = time_series_to_num(df_data_pool[i])

# 处理aF12
df_data_pool['aF12_1'], df_data_pool['aF12_2'], df_data_pool['aF12_3'] = aF12(df_data_pool['aF12'])
df_data_pool.drop('aF12',axis=1, inplace=True)

#处理aF11（直接取相加结果）
df_data_pool['aF11'] = aF11(df_data_pool['aF11'])
df_data_pool = df_data_pool.astype(float)

#处理aF13
df_data_pool['aF13'] = aF13(df_data_pool['aF13'])

#新建一列车辆已使用时间
df_data_pool['timespan'] = df_data_pool[['tradeTime',"licenseDate"]].apply(lambda x:x['tradeTime']-x['licenseDate'],axis = 1)

###############################    转化成数字格式    ######################################
df_data_pool = df_data_pool.astype(float)


# 以下特征缺失值众数填充
'''tradeTime, brand, serial, model, color, cityId, carCode, transferCount, 
seatings, registerDate, licenseDate, country, makertype, modelyear, gearbox, 
oiltype, aF1, aF2, aF3, aF6, aF7, aF8, aF9, aF10, aF11, aF13, aF14, aF15'''

for i in ['tradeTime', 'brand', 'serial', 'model', 'color', 'cityId', 'carCode', 'transferCount',
'seatings', 'registerDate', 'licenseDate', 'country', 'makertype', 'modelyear', 'gearbox',
'oiltype', 'aF1', 'aF2', 'aF3', 'aF6', 'aF7', 'aF8', 'aF9', 'aF10', 'aF11', 'aF13', 'aF14', 'aF15']:
    df_data_pool[i] = fillna_mode(df_data_pool[i])

#以下特征缺失值均值填充
'''mileage, displacement, newprice,aF4, aF5, aF12_1, aF12_2, aF12_3, timespan'''

for i in ['mileage', 'displacement', 'newprice', 'aF4', 'aF5', 'aF12_1', 'aF12_2', 'aF12_3', 'timespan']:
    df_data_pool[i] = fillna_average(df_data_pool[i])

final_features_pool = ['carid', 'tradeTime', 'brand', 'serial', 'model','mileage',
                       'color', 'cityId', 'carCode', 'transferCount','seatings',
                       'registerDate', 'licenseDate', 'country', 'makertype',
                       'modelyear','displacement', 'gearbox', 'oiltype', 'newprice',
                       'aF1', 'aF2', 'aF3','aF4', 'aF5', 'aF6', 'aF7', 'aF8', 'aF9',
                       'aF10', 'aF11', 'aF13', 'aF14', 'aF15','aF12_1', 'aF12_2', 'aF12_3',
                       'timespan']

#归一化所有数据
scaler = MinMaxScaler(feature_range=(1,10))
for i in ['tradeTime', 'brand', 'serial', 'model','mileage',
                       'color', 'cityId', 'carCode', 'transferCount','seatings',
                       'registerDate', 'licenseDate', 'country', 'makertype',
                       'modelyear','displacement', 'gearbox', 'oiltype', 'newprice',
                       'aF1', 'aF2', 'aF3','aF4', 'aF5', 'aF6', 'aF7', 'aF8', 'aF9',
                       'aF10', 'aF11', 'aF13', 'aF14', 'aF15','aF12_1', 'aF12_2', 'aF12_3',
                       'timespan']:
    df_data_pool[i] = scaler.fit_transform(np.array(df_data_pool[i]).reshape(-1,1))



df_data_pool = df_data_pool.set_index(['carid'])

df_data_pool.to_csv('x.csv')



