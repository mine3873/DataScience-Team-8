# Data Science
``` python
# import DataScience Class for running each step of end-to-end BIG DATA process
from DataScience import DataScience
```

``` python
process = DataScience()
```

``` python
# load data
process.load_dataSet()
```
  
``` python
# print statistical info in data
process.print_statistical()
```

``` python
# run the step of preprocessing
process.preprocessing(
    dealing_outlier=True,
    run_normalize=False,
    selectBestFeatures=True,
    numOfBestFeatures = 10,
    method='one_hot'
)
```

``` python
# run the step of training model.
process.trainModel(
    model='decisionTree',
    useSmoth=False,
    test_size=0.2
)
```
  
``` python
# evaluate the trained model
process.evaluate(
    printResult=True,
    tuneThreshold=True
)
```
  
## CODE/.py
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

- `AUTOMATIC.py`
`DataScienceTermP.py`와 마찬가지로 데이터 로드, 전처리, 학습 각 과정을 자동으로 처리해주는 함수 저장.
그러나 함수 호출 시,  사용할 파라미터를 설정하여 (ex, 범주형 데이터 처리 방식: one hot OR label?) 
각 과정의 성능 비교 API. 

- `DataScience.py`
end-to-end Big Data의 각 단계 별 함수 선언한 클래스 파일

