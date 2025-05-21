# INTRODUCTION
## CODE/.py 소개(기능, 목적)
- `downloadDataSet.py`
kaggle 로부터 데이터 다운 

- `load_dataSet.py`
`.csv` 파일로부터 데이터셋을 불러와 저장.

- `dataInspection.py`
데이터셋의 정보 출력.

- `dataPreprocessing.py`
데이터 전처리 기능 함수 저장된 파일.  
outlier, 범주형->숫자형 등등..

- `train_model.py`
모델 학습 (DecisionTree, K-Means) 함수 저장된 파일.
random Forest, Gradient Boosting 등은 `.ipynb` 에만 선언해서 따로 추가하겠습니다. 

- `showTheResult.py`
학습한 모델 평가 및 결과 시각화 함수 저장된 파일.

- `DataScienceTermP.py`
위 6개의 `.py` 파일 속에서 선언된 함수들을 불러와 텀프로젝트 과정 전반을 수행.

- `AUTOMATIC.py`
`DataScienceTermP.py`와 마찬가지로 데이터 로드, 전처리, 학습 각 과정을 자동으로 처리해주는 함수 저장.
그러나 함수 호출 시,  사용할 파라미터를 설정하여 (ex, 범주형 데이터 처리 방식: one hot OR label?) 
각 과정의 성능 비교 API. 
