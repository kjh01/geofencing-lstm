'''
- main.py: 학습 데이터 셋 크기 설정.
- preprocessing.py:
    - 학습 데이터 셋: 3차원 reshape + window size(4) + strides(1) > 학습 데이터 셋 전처리
    - 테스트 데이터 셋: 3차원 reshape + window size(4) + strides(1) > 학습 데이터 셋 전처리
- train.py: 전처리한 학습 데이터 셋 > 학습 모델 생성 및 저장.
- test.py: 전처리한 테스트 데이터 셋 > 생성된 학습 모델로 예측 및 결과 저장
- results.py: 저장한 결과 plot 및 오차 거리 계산
'''

'''
parameters = {
    path = "p",     # c / k / m / p
    YN = "out",     # in / out : 테스트 데이터셋이 훈련 데이터셋에 포함되었는지에 대한 유무 확인
    g = "G",        # G / NG
    features = 3,
    window_size = 4
}
'''
