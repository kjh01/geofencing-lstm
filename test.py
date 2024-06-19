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

# .npy 파일에서 정규화 X 테스트 데이터 불러오기
x_test = np.load('x_test_3d_nor.npy')

# .csv 파일에서 GPS Y 테스트 데이터 불러오기
y_test = pd.read_csv("y_test_" + path + "_GPS.csv)

# Load Train Model > input: x_test, output: yhat
model = load_model('model150_256_200.h5')
np.set_printoptions(precision=10)
yhat = model.predict(x_test) # yhat에는 정규화된 값으로 저장되어 있음

# Test Results: df 변환 및 저장
def list_to_dataframe(matrix):
    df = pd.DataFrame(matrix, columns=['Lat', 'Lon', 'Time'])
    return df
my_yhat = list_to_dataframe(yhat)

# Save Yhat and y_test
save_path2 = "yhat_" + path + ".csv"
my_yhat.to_csv(save_path2, index=False)
