import time
import keras
from keras.layers import Dense, LSTM
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
import csv

window_size = 4  # 윈도우 크기 설정
features = 3

def list_to_dataframe(matrix): # test 결과 데이터프레임 변환 및 저장
    df = pd.DataFrame(matrix, columns=['Lat', 'Lon', 'Time'])
    return df

# .npy 파일에서 정규화 X 테스트 데이터 불러오기
x_test = np.load('x_test_3d_nor.npy')

# .csv 파일에서 GPS Y 테스트 데이터 불러오기
y_test = pd.read_csv("y_test_GPS.csv)

# Load Train Model > input: x_test, output: yhat
model = load_model('model150_256_200.h5')
np.set_printoptions(precision=10)
yhat = model.predict(x_test) # yhat에는 정규화된 값으로 저장되어 있음

# yhat 역전환
# my_yhat = list_to_dataframe(yhat)
# df0, df1 = my_yhat['Lat'],my_yhat['Lon']
# rescaled_pred_Lat = scaler1.inverse_transform(np.array(df0).reshape(-1,1))
# rescaled_pred_Lon = scaler2.inverse_transform(np.array(df1).reshape(-1,1))

# yhat = np.concatenate((rescaled_pred_Lat, rescaled_pred_Lon), axis=1).tolist()
# column_names = ['Lat', 'Lon']  # 열 이름을 지정
# combined_array = pd.DataFrame(yhat, columns=column_names)

# 역전환한 yhat 저장하기(정규화 > GPS)
# save_path1 = "yhat_GPS.csv"
# combined_array.to_csv(save_path1, index=False)

# Test Results: df 변환 및 저장
def list_to_dataframe(matrix):
    df = pd.DataFrame(matrix, columns=['Lat', 'Lon', 'Time'])
    return df
# my_ytest = list_to_dataframe(y_test)
my_yhat = list_to_dataframe(yhat)

# Save Yhat and y_test
# save_path1 = "y_test_.csv"
save_path2 = "yhat.csv"
# my_ytest.to_csv(save_path1, index=False)
my_yhat.to_csv(save_path2, index=False)
