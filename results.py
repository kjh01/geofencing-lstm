'''

- main.py: 학습 데이터 셋 크기 및 여러 파라미터 설정. filter.py를 통해 train 데이터셋과 test 데이터셋을 뽑고 
    lstm_model.py에서 모델 불러와서 학습 및 예측
    
- filter.py:
    - 학습 데이터 셋: 3차원 reshape + window size(4) + strides(1) > 학습 데이터 셋 전처리
    - 테스트 데이터 셋: 3차원 reshape + window size(4) + strides(1) > 학습 데이터 셋 전처리
- train.py: 전처리한 학습 데이터 셋 입력> 학습 모델 생성 및 저장.
- test.py: 전처리한 테스트 데이터 셋 입력> 생성된 학습 모델로 예측 및 결과 저장
- results.py: 
    - 저장한 결과 plot 및 오차 거리 계산
    - 오차 거리 계산 코드
    - y_test와 yhat: plot 및 map 저장 코드

parameters = {
    path = "p",     # c / k / m / p
    YN = "out",     # in / out : 테스트 데이터셋이 훈련 데이터셋에 포함되었는지에 대한 유무 확인
    g = "G",        # G / NG
    features = 3,
    window_size = 4
}
'''

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import train
import test
import filter

path = "p" # 테스트 경로 (c/k/m/p)
YN = "out" # 학습 데이터셋에 테스트 데이터셋이 포함/미포함되었는지 (in/out)
g = "G" # 장거리 경로 포함/미포함 (G/NG)

# GPS y_test와 정규화 yhat 불러오기
df1 = pd.read_csv("y_test_" + path + "_GPS.csv")
df2 = pd.read_csv("yhat_" + path + "_.csv")

# 정규화 yhat 역전환용 Scaler 불러오기(filter.py)
scaler1 = joblib.load("scaler_lat.pkl")
scaler2 = joblib.load("scaler_lon.pkl")

# 정규화 yhat 역전환
df20 = df2['Lat']
df21 = df2['Lon']
rescaled_pred_Lat = scaler1.inverse_transform(np.array(df20).reshape(-1,1))
rescaled_pred_Lon = scaler2.inverse_transform(np.array(df21).reshape(-1,1))

# 역전환한 GPS Yhat 저장하기
yhat = np.concatenate((rescaled_pred_Lat, rescaled_pred_Lon), axis=1).tolist()
column_names = ['Lat', 'Lon']  # 열 이름을 지정
combined_array = pd.DataFrame(yhat, columns=column_names)
save_path1 = "yhat_" + path + "_GPS.csv"
combined_array.to_csv(save_path1, index=False)

# GPS y_teat와 GPS yhat 다시 불러오기
y_test = pd.read_csv("y_test_GPS.csv")
yhat = pd.read_csv("yhat_GPS.csv")
y_test = df1.values.tolist()
yhat = df2.values.tolist()

'''
아래는 오차거리 계산 및 저장하는 코드
'''
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
print("오차 거리 계산(km): \n{}\n".format(distance_list))

# 평균 오차거리 km
avg_err_distance = sum(distance_list)/len(distance_list)
print('평균 오차거리(km): {} km\n'.format(avg_err_distance))

# 최대 오차거리 km
max_err_distance = max(distance_list)
print('최대 오차거리(km): {} km\n'.format(max_err_distance))

# 분산 거리 km²
variance = np.var(distance_list)
print('분산: {} km²\n'.format(variance))

# MSE km²
mse = np.mean(np.square(distance_list))
print('ytest와 yhat의 MSE : {} km²\n'.format(mse))

# 오차 거리 txt 저장
with open(f"results_{path}.txt", 'w') as file:
    file.write("오차 거리 계산(km): \n{}\n\n".format(distance_list))
    file.write('평균 오차거리(km): {} km\n\n'.format(avg_err_distance))
    file.write('최대 오차거리(km): {} km\n\n'.format(max_err_distance))
    file.write('분산: {} km²\n\n'.format(variance))
    file.write('ytest와 yhat의 MSE : {} km²\n\n'.format(mse))


'''
아래는 y_test 좌표 & yhat 좌표 plot(.png) 출력 및 map(.html) 저장하는 코드
'''
import folium
from geopy.distance import geodesic, great_circle
from matplotlib import pyplot as plt

y_test = pd.read_csv("y_test_" + path + "_GPS.csv")
yhat = pd.read_csv("yhat_" + path + "_GPS.csv")

'''
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
'''

# 기준 위도와 경도 설정
base_lat_hat = yhat['Lat'][0]
base_lon_hat = yhat['Lon'][0]
base_lat = y_test['Lat'][0]
base_lon = y_test['Lon'][0]

# 거리 계산
yhat['Distance_Lat'] = yhat.apply(lambda row: great_circle((base_lat_hat, base_lon_hat), (row['Lat'], base_lon_hat)).kilometers, axis=1)
yhat['Distance_Lon'] = yhat.apply(lambda row: great_circle((base_lat_hat, base_lon_hat), (base_lat_hat, row['Lon'])).kilometers, axis=1)

y_test['Distance_Lat'] = y_test.apply(lambda row: great_circle((base_lat, base_lon), (row['Lat'], base_lon)).kilometers, axis=1)
y_test['Distance_Lon'] = y_test.apply(lambda row: great_circle((base_lat, base_lon), (base_lat, row['Lon'])).kilometers, axis=1)

# Scatter plot
plt.scatter(Real['Distance_Lon'], Real['Distance_Lat'], s=20, color='green', label='Real')
plt.xlabel('Distance_Lon (km)')
plt.ylabel('Distance_Lat (km)')
plt.legend()
plt.grid(True)
plt.title("ipin_24" _" + path )
plt.savefig("png_ipin_" + path + ".png")
plt.show()

# 지도 생성 및 저장
if not y_test.empty and not yhat.empty:
    map_center = [y_test['Lat'].iloc[0], y_test['Lon'].iloc[0]]
    m = folium.Map(location=map_center, zoom_start=15)

    # y_test 데이터 빨간 마커로 추가
    for idx, row in y_test.iterrows():
        if pd.notnull(row['Lat']) and pd.notnull(row['Lon']):
            folium.Marker([row['Lat'], row['Lon']],  icon=folium.Icon(color='red')).add_to(m)

    # yhat 데이터 파란 마커로 추가
    for idx, row in yhat.iterrows():
        if pd.notnull(row['Lat']) and pd.notnull(row['Lon']):
            folium.Marker([row['Lat'], row['Lon']], icon=folium.Icon(color='blue')).add_to(m)

    m.save(
        "map_ipin_" + path + ".html")
