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
    # 입력 디렉토리 내의 모든 파일을 처리합니다.
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            input_file_path = os.path.join(input_dir, filename)

            # CSV 파일을 DataFrame으로 읽어옵니다.
            data = pd.read_csv(input_file_path)

            # sliding 함수를 호출하여 데이터를 슬라이딩하고 결과를 받습니다.
            window_sliding_data_Xtrain, window_sliding_data_Ytrain = sliding(data, window_size)

            # 최초 데이터일 경우 초기화, 그렇지 않으면 이어붙입니다.
            if result_x is None:
                result_x = window_sliding_data_Xtrain
                result_y = window_sliding_data_Ytrain
            else:
                result_x = np.vstack((result_x, window_sliding_data_Xtrain))
                result_y = np.vstack((result_y, window_sliding_data_Ytrain))

    return result_x, result_y

# train / 입력 및 출력 디렉토리 설정
input_directory = "C:/Users/user/PycharmProjects/Geofencing_main/01.Research/02.Preprocessing02/Create_random_path/Paths_by_individuals/train_1m/train_in/train_in_p150_NG"  # train 입력 디렉토리 경로
# 입력 디렉토리 내의 파일을 슬라이딩하고 결과를 저장합니다.
x_train, y_train = process_and_save_files(input_directory, window_size)
# test
input_directory2 = "C:/Users/user/PycharmProjects/Geofencing_main/01.Research/02.Preprocessing02/Create_random_path/Paths_by_individuals/test_1m/test_ckmp/test_1m_c"  # test 입력 디렉토리 경로
x_test, y_test = process_and_save_files(input_directory2, window_size)

# 훈련 데이터 변환
y_train = np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))
# 테스트 데이터 변환
y_test = np.reshape(y_test, (y_test.shape[0], 1, y_test.shape[1]))

# 주어진 3D 리스트들을 2D 배열로 변환
x_train_2d = np.concatenate(x_train)
y_train_2d = np.concatenate(y_train)
x_test_2d = np.concatenate(x_test)
y_test_2d = np.concatenate(y_test)

all_data = np.concatenate([x_train_2d, y_train_2d, x_test_2d, y_test_2d], axis=0)
# print(all_data)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
all_data_normalized = scaler.fit_transform(all_data.reshape(-1, 3))

x_train_3d_normalized = all_data_normalized[:len(x_train_2d)]
y_train_3d_normalized = all_data_normalized[len(x_train_2d):len(x_train_2d)+len(y_train_2d)]
x_test_3d_normalized = all_data_normalized[len(x_train_2d)+len(y_train_2d):len(x_train_2d)+len(y_train_2d)+len(x_test_2d)]
y_test_3d_normalized = all_data_normalized[len(x_train_2d)+len(y_train_2d)+len(x_test_2d):]

# 3차원 배열을 저장할 리스트 초기화
x_train_3d = []
x_test_3d = []

# 각 행에서 4개씩 추출하여 3차원 배열로 변환
for i in range(0, len(x_train_3d_normalized) - window_size + 1, window_size):
    window = x_train_3d_normalized[i:i + window_size]  # 윈도우 크기만큼 행 추출
    x_train_3d.append(window)

# 각 행에서 4개씩 추출하여 3차원 배열로 변환
for i in range(0, len(x_test_3d_normalized) - window_size + 1, window_size):
    window = x_test_3d_normalized[i:i + window_size]  # 윈도우 크기만큼 행 추출
    x_test_3d.append(window)
    
# 리스트를 넘파이 배열로 변환
x_train = np.array(x_train_3d)
x_test = np.array(x_test_3d)
y_train = np.array(y_train_3d_normalized)
y_test = np.array(y_test_3d_normalized)

# 3차원 배열 저장하기
######### code 추가 필요 #########
