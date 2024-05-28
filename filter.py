from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import pandas as pd
import numpy as np
import csv
import os

window_size = 4  # 윈도우 크기 설정
features = 3

def sliding(origin_data, window_size):
    window_sliding_data_Xtrain = origin_data.values[np.arange(window_size)[None, :]
                                                            + np.arange(origin_data.shape[0] - window_size)[:, None]]
    window_sliding_data_Ytrain = origin_data.values[window_size:]

    return window_sliding_data_Xtrain, window_sliding_data_Ytrain

def process_and_save_files(input_dir, window_size):
    result_x, result_y = None, None
    # 입력 디렉토리 내의 모든 파일을 처리 
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"): # 각 파일을 하나씩 불러옴 > 서로 다른 날의 데이터가 같은 윈도우에 들어가지 않도록
            input_file_path = os.path.join(input_dir, filename)

            # CSV 파일을 DataFrame으로 읽어옴
            data = pd.read_csv(input_file_path)

            # sliding 함수를 호출하여 데이터를 슬라이딩하고 결과를 받음
            window_sliding_data_Xtrain, window_sliding_data_Ytrain = sliding(data, window_size)

            # 최초 데이터일 경우 초기화, 그렇지 않으면 이어붙임.
            if result_x is None:
                result_x = window_sliding_data_Xtrain
                result_y = window_sliding_data_Ytrain
            else:
                result_x = np.vstack((result_x, window_sliding_data_Xtrain))
                result_y = np.vstack((result_y, window_sliding_data_Ytrain))

    return result_x, result_y

def list_to_dataframe(matrix):
    df = pd.DataFrame(matrix, columns=['Lat', 'Lon'])
    return df

# 입력 디렉토리 설정
input_dir_train = "C:/Users/user/PycharmProjects/Geofencing_main/01.Research/02.Preprocessing02/Create_random_path/Paths_by_individuals/train_1m/train_in/train_in_p150_NG"  # train 입력 디렉토리 경로
input_dir_test = "C:/Users/user/PycharmProjects/Geofencing_main/01.Research/02.Preprocessing02/Create_random_path/Paths_by_individuals/test_1m/test_ckmp/test_1m_c"  # test 입력 디렉토리 경로

# 입력 디렉토리 내의 raw 파일을 슬라이딩하고 결과를 반환 > 3D
x_train, y_train = process_and_save_files(input_dir_train, window_size)
x_test, y_test = process_and_save_files(input_dir_test, window_size)

# 슬라이딩한 y_test 데이터(gps) .csv로 저장 > results.py에서 'GPS로 역전환한 yhat'과 y_test(gps)의 오차 거리와 map을 추출하기 위함
y_test_gps = list_to_dataframe(y_test[:,:2])
y_test_gps.to_csv("C:/Users/user/PycharmProjects/Geofencing_main/01.Research/02.Preprocessing02/Ydata_ytest_yhat/ytest_1m/ytest_out_NG/y_test150_c_GPS.csv", index=False)

# 훈련/테스트 데이터 변환 > 3D
y_train = np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))
y_test = np.reshape(y_test, (y_test.shape[0], 1, y_test.shape[1]))

# 주어진 3D 리스트들을 2D 배열로 변환 > *한꺼번에 모든 입력 데이터를 정규화하기 위해서
x_train_2d = np.concatenate(x_train) 
y_train_2d = np.concatenate(y_train)
x_test_2d = np.concatenate(x_test)
y_test_2d = np.concatenate(y_test)

# 모든 데이터 합치기 > *한꺼번에 모든 raw 데이터를 정규화하기 위해서
all_data = np.concatenate([x_train_2d, y_train_2d, x_test_2d, y_test_2d], axis=0) # 2D

# GPS 정규화
scaler = MinMaxScaler() 
all_data_normalized = scaler.fit_transform(all_data.reshape(-1, 3)) # 2D

# 정규화한 전체 데이터에서 각 훈련/테스트 데이터 사이즈만큼 자르기
x_train_2d_normalized = all_data_normalized[:len(x_train_2d)] # 2D
y_train_2d_normalized = all_data_normalized[len(x_train_2d):len(x_train_2d)+len(y_train_2d)] # 2D
x_test_2d_normalized = all_data_normalized[len(x_train_2d)+len(y_train_2d):len(x_train_2d)+len(y_train_2d)+len(x_test_2d)] # 2D
y_test_2d_normalized = all_data_normalized[len(x_train_2d)+len(y_train_2d)+len(x_test_2d):] # 2D

# 3차원 배열을 저장할 리스트 초기화
x_train_3d = []
x_test_3d = []

# 각 행에서 4개씩(window size) 추출하여 3차원 배열로 변환
def x_3d_reshape(intial_list, x_2d_nor, window_size):
    for i in range(0, len(x_2d_nor) - window_size + 1, window_size):
        window = x_2d_nor[i:i + window_size]  # 윈도우 크기만큼 행 추출
        intial_list.append(window)
    return np.array(intial_list) # 리스트를 넘파이 배열로 변환

x_train_3d_nor = x_3d_reshape(x_train_3d, x_train_2d_normalized, window_size) # 3D
x_test_3d_nor = x_3d_reshape(x_test_3d, x_test_2d_normalized, window_size)    # 3D
y_train_nor = np.array(y_train_2d_normalized) # 2D 리스트를 넘파이 배열로 변환
y_test_nor = np.array(y_test_2d_normalized) # 2D 리스트를 넘파이 배열로 변환

# 정규화한 3차원 및 2차원 배열(.npy) 저장하기
np.save('geofencing-lstm/Train_Dataset_pro/x_train_3d_nor.npy', x_train_3d_nor)
np.save('geofencing-lstm/Test_Dataset_pro/x_test_3d_nor.npy', x_test_3d_nor)
np.save('geofencing-lstm/Train_Dataset_pro/y_train_nor.npy', y_train_nor)
np.save('geofencing-lstm/Test_Dataset_pro/y_test_nor.npy', y_test_nor)

# .npy 파일에서 정규화 데이터 불러오기
loaded_array = np.load('x_train_3d_nor.npy')
print(loaded_array)
