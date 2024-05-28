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

# 3차원 훈련 데이터(x,y) 불러오기
x_train = np.load('x_train_3d_nor.npy')
y_train = np.load('y_train_nor.npy')

# Start Recording Training Time
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
time_callback = TimeHistory()

# Structure of Model 
def LSTM():
    model = keras.models.Sequential([
        # keras.layers.Input(input_shape=(time_step, features)), # input_shape 은 사용하지 X
        keras.layers.Input(shape=(x_train.shape[1], features)), # features = 3
        keras.layers.LSTM(256, return_sequences=True, name='LSTM'),
        keras.layers.LSTM(128, return_sequences=True, name='LSTM_0'),
        keras.layers.LSTM(64, return_sequences=True, name='LSTM_1'), # 1024 -> 512 -> 256 -> 128 -> 64 -> 2
        keras.layers.LSTM(32, return_sequences=True, name='LSTM_2'),
        keras.layers.LSTM(16, return_sequences=True, name='LSTM_3'),
        # keras.layers.Flatten(),
        # keras.layers.LSTM(8),
        keras.layers.LSTM(8, return_sequences=True, name='LSTM_4'), #
        keras.layers.Flatten(), #
        keras.layers.Dense(4), #
        # keras.layers.Dense(4, activation="swish")
        keras.layers.Dense(3)
        # keras.layers.Dense(2, activation="sigmoid") # activation="sigmoid" 를 추가하면 원하는 값이 나오지 않습니다!! 와후!!
        # activation="sigmoid" 01 비교하는 거
    ])
    return model
model = LSTM()
model.summary()

# Train
start = dt.datetime.now()
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, np.array(y_train), epochs=200, batch_size=4)
model.save('01.Research/02.Preprocessing02/Saved_models/model_in/model_in_NG/model150_256_200.h5')

# Stop Recording Training Time
end = dt.datetime.now()

# Save Training Time
times = end - start
print(times)
