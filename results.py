import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

path = "p" # 테스트 경로 (c/k/m/p)
IO = "out" # 학습 데이터셋에 테스트 데이터셋이 포함/미포함되었는지 (in/out)
g = "G" # 장거리 경로 포함/미포함 (G/NG)

df1 = pd.read_csv("01.Research/02.Preprocessing02/Ydata_ytest_yhat/ytest_1m/ytest_out_G/y_test150_" + path + "_GPS.csv")
df2 = pd.read_csv("01.Research/02.Preprocessing02/Ydata_ytest_yhat/yhat_restoration_1m/yhat_"+ IO + "_" + g + "/yhat150_" + path + "_re_256_ep200.csv")

y_test = df1.values.tolist()
yhat = df2.values.tolist()

def calculate_distance(coords1, coords2):
    # 지구의 반경 (단위: km)
    radius = 6371.0

    # 위도와 경도를 라디안 단위로 변환
    lat1 = radians(coords1[0])
    lon1 = radians(coords1[1])
    lat2 = radians(coords2[0])
    lon2 = radians(coords2[1])

    # Haversine 공식을 이용하여 두 지점 사이의 거리 계산
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = radius * c

    # 거리를 킬로미터(km)로 변환
    distance_km = distance

    return distance_km # distance_meter

def calculate_distance_list(test, pred):
    # 오차 거리를 저장할 리스트
    distance_list = []

    # test와 pred의 길이가 동일한지 확인
    if len(test) != len(pred):
        raise ValueError("The lengths of test and pred should be the same.")

    # 각 test와 pred 쌍에 대해 오차 거리 계산
    for coords1, coords2 in zip(test, pred):
        distance = calculate_distance(coords1, coords2)
        distance_list.append(distance)

    return distance_list

# 오차 거리 계산
distance_list = calculate_distance_list(y_test, yhat)

# 평균 오차거리 km
print('평균 오차거리(km): {} km'.format(sum(distance_list)/len(distance_list)))

# 최대 오차거리 km
print('최대 오차거리(km): {} km'.format(max(distance_list)))

# 분산 거리 km²
variance = np.var(distance_list)
print('분산: {} km²'.format(variance))

# MSE km²
mse = np.mean(np.square(distance_list))
print('ytest와 yhat의 MSE : {} km²'.format(mse))

# 오차 거리 txt 저장
######### code 추가 필요 #########
