import numpy as np
import pandas as pd

# 행과 열의 개수 설정
rows = 100
columns = 128

# 평균과 표준편차 설정
mean = 50
std_devs = [1]  # 표준편차 리스트로 변경

# 각 표준편차에 대해 파일 생성
for std_dev in std_devs:
    # 정규 분포에서 난수 생성 (평균=50, 표준편차=std_dev)
    random_data = np.random.normal(loc=mean, scale=std_dev, size=(rows, columns))
    
    # 0과 100 사이로 클리핑 (범위를 0~1으로 제한)
    random_data = np.clip(random_data, 0, 100)
    
    # 평균과 표준편차 계산
    current_mean = np.mean(random_data)
    current_std = np.std(random_data)
    
    # 평균 50과 표준편차 20으로 맞추기 위한 조정
    adjusted_data = (random_data - current_mean) / current_std * std_dev + mean

    # DataFrame으로 변환
    df = pd.DataFrame(adjusted_data)
    
    # 파일 이름 생성
    file_name = f'random_{rows/1000000}M_{columns}.csv'
    
    # CSV 파일로 저장
    df.to_csv(file_name, index=False)

    print(f'{file_name} 생성 완료')

    # 조정된 데이터의 평균과 표준편차 확인
    print(f'{file_name}의 평균: {np.mean(adjusted_data)}, 표준편차: {np.std(adjusted_data)}')
