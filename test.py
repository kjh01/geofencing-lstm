import time
import keras
from keras.layers import Dense, LSTM
from keras.models import load_model
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
import csv

window_size = 4  # 윈도우 크기 설정
features = 3

# 3차원 테스트 데이터(x,y) 불러오기
np.save('geofencing-lstm/Test_Dataset_pro/x_test_3d_nor.npy', x_test_3d_nor)
np.save('geofencing-lstm/Train_Dataset_pro/y_train_nor.npy', y_train_nor)
np.save('geofencing-lstm/Test_Dataset_pro/y_test_nor.npy', y_test_nor)

# .npy 파일에서 정규화 데이터 불러오기
x_test = np.load('geofencing-lstm/Test_Dataset_pro/x_test_3d_nor.npy')
y_test = pd.read_csv("C:/Users/user/PycharmProjects/Geofencing_main/01.Research/02.Preprocessing02/Ydata_ytest_yhat/ytest_1m/ytest_out_NG/y_test150_c_GPS.csv)

# Load Train Model
model = load_model('01.Research/02.Preprocessing02/Saved_models/model_in/model_in_NG/model150_256_200.h5')
np.set_printoptions(precision=10)
yhat = model.predict(x_test)

# Test Results: df 변환 및 저장
def list_to_dataframe(matrix):
    df = pd.DataFrame(matrix, columns=['Lat', 'Lon', 'Time'])
    return df
my_ytest = list_to_dataframe(y_test)
my_yhat = list_to_dataframe(yhat)

# Save Yhat and y_test
save_path1 = "01.Research/02.Preprocessing02/Ydata_ytest_yhat/ytest_1m/ytest_in_NG/y_test150_c_256_ep200.csv"
save_path2 = "01.Research/02.Preprocessing02/Ydata_ytest_yhat/yhat_1m/yhat_in_NG/yhat150_c_256_ep200.csv"
my_ytest.to_csv(save_path1, index=False)
my_yhat.to_csv(save_path2, index=False)
