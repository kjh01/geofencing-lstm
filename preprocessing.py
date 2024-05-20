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
