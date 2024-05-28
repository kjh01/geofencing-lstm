'''
- 오차 거리 계산 코드
- y_test와 yhat: plot 및 map 저장 코드
'''

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import lstm_model
import filter

path = "p" # 테스트 경로 (c/k/m/p)
YN = "out" # 학습 데이터셋에 테스트 데이터셋이 포함/미포함되었는지 (in/out)
g = "G" # 장거리 경로 포함/미포함 (G/NG)

df1 = pd.read_csv("01.Research/02.Preprocessing02/Ydata_ytest_yhat/ytest_1m/ytest_out_G/y_test150_" + path + "_GPS.csv")
df2 = pd.read_csv("01.Research/02.Preprocessing02/Ydata_ytest_yhat/yhat_restoration_1m/yhat_"+ YN + "_" + g + "/yhat150_" + path + "_re_256_ep200.csv")

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


'''
아래는 y_test 좌표 & yhat 좌표 plot(.png) 출력 및 map(.html) 저장하는 코드
'''

import folium
from geopy.distance import geodesic, great_circle
from matplotlib import pyplot as plt

df = pd.read_csv("01.Research/03.AnomalyDetection/GPSdata/school/ysh/ipin_2405_ysh_all.csv")

# 필요한 컬럼만 선택
df = df[['요청시간', '참위치 경도', '참위치 위도']]

# '요청시간' 컬럼을 datetime 타입으로 변환
df['요청시간'] = pd.to_datetime(df['요청시간'])
group = df.sort_values(by='요청시간').reset_index(drop=True)

week = 4
week_kr = "금"
date = "03"

# 특정 날짜 필터링
s_df = group[group['요청시간'].dt.date == pd.to_datetime('2024-05-03').date()]

def convert_time(time_str, w):
    h, m, s = time_str.split(':')
    converted = int(f"{w}{int(h):02d}{int(m):02d}{int(s):02d}")
    return converted

s_df['Time'] = s_df['요청시간'].apply(lambda x: x.time().strftime('%H:%M:%S'))
s_df['WeekTime'] = s_df['Time'].apply(lambda x: convert_time(x, week)) # 0 = 월요일, ...

# 0 값을 제거
Real = s_df[(s_df[['참위치 위도', '참위치 경도']] != 0).all(axis=1)]

# 기준 위도와 경도 설정
base_lat = Real['참위치 위도'].iloc[0]
base_lon = Real['참위치 경도'].iloc[0]

# 거리 계산
Real['Distance_Lat'] = Real.apply(lambda row: great_circle((base_lat, base_lon), (row['참위치 위도'], base_lon)).kilometers, axis=1)
Real['Distance_Lon'] = Real.apply(lambda row: great_circle((base_lat, base_lon), (base_lat, row['참위치 경도'])).kilometers, axis=1)

# Scatter plot
plt.scatter(Real['Distance_Lon'], Real['Distance_Lat'], s=20, color='green', label='Real')
plt.xlabel('Distance_Lon (km)')
plt.ylabel('Distance_Lat (km)')
plt.legend()
plt.grid(True)
plt.title("ipin_2405" + date)
plt.savefig("01.Research/03.AnomalyDetection/GPSdata/plot_png/ysh/map_ipin_05" + date + "_" + week_kr +".png")
plt.show()

# 지도 생성
if not Real.empty:
    map_center = [Real['참위치 위도'].iloc[0], Real['참위치 경도'].iloc[0]]
    m = folium.Map(location=map_center, zoom_start=15)

    for idx, row in Real.iterrows():
        if pd.notnull(row['참위치 위도']) and pd.notnull(row['참위치 경도']):
            folium.Marker([row['참위치 위도'], row['참위치 경도']]).add_to(m)

    m.save("01.Research/03.AnomalyDetection/GPSdata/html/ysh/map_ipin_05" + date + "_" + week_kr +".html")
