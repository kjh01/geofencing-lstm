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
x_train = pd.read_csv()
y_train = pd.read_csv()

# Start Recording Training Time
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
time_callback = TimeHistory()
