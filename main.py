'''
- main.py: 학습 데이터 셋 크기 및 여러 파라미터 설정. filter.py를 통해 train 데이터셋과 test 데이터셋을 뽑고 
    lstm_model.py에서 모델 불러와서 학습 및 예측
- filter.py:
    - 학습 데이터 셋: 3차원 reshape + window size(4) + strides(1) > 학습 데이터 셋 전처리
    - 테스트 데이터 셋: 3차원 reshape + window size(4) + strides(1) > 학습 데이터 셋 전처리
- lstm_model.py: 전처리한 학습 데이터 셋 입력> 학습 모델 생성 및 저장.
- test.py: 전처리한 테스트 데이터 셋 입력> 생성된 학습 모델로 예측 및 결과 저장
- results.py: 저장한 결과 plot 및 오차 거리 계산

parameters = {
    path = "p",     # c / k / m / p
    YN = "out",     # in / out : 테스트 데이터셋이 훈련 데이터셋에 포함되었는지에 대한 유무 확인
    g = "G",        # G / NG
    features = 3,
    window_size = 4
}
'''

import lstm_model
import results
import filter

path = "p"     # c / k / m / p
YN = "out"     # in / out : 테스트 데이터셋이 훈련 데이터셋에 포함되었는지에 대한 유무 확인
g = "G"        # G / NG
features = 3
window_size = 4
