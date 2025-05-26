#from DataScienceTermP import preprocessed_df

#df = preprocessed_df
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ------------------------------------
# Decision Tree
# ------------------------------------
def train_decisionTree(df, 
                        useSmoth = False,
                        test_size = 0.2,
                        param_grid = {
                            'criterion': ['gini', 'entropy'],
                            'max_depth': [3, 5, 7, 10, None],
                            'min_samples_leaf': [1, 3, 5, 7, 10],
                            'class_weight': [None, 'balanced']
                            },
                        target='Churn'
                        ):
    """_summary_
        Given some parameters for training decision tree model, 
        train decision tree models with optional parameters and return the best model.
        
        모델 학습 함수, 
    Args:
        df (pd.DataFrame): 
            학습에 사용할 데이터셋.
        param_grid (dict, optional): 
            parameters for training decision tree model. among these decision model, the best performance model is selected. 
                Defaults to { 'criterion': ['gini', 'entropy'],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_leaf': [1, 5, 10] }.
            모델 학습에 사용할 파라미터 값들 -> DecisionTreeClassifier 클래스 내의 파라미터 값. 
            각 설정한 값의 조합으로 모델들을 학습하여, 가장 나은 성능의 모델 선택. -> bestModel.
            
        Returns: bestModel, test_X, test_y
    """
    X = df.drop(target, axis=1)
    y = df[target]
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=1
    )
    
    if useSmoth:
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=1),
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    bestModel = grid_search.best_estimator_
    
    return bestModel, X_test, y_test

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_randomForest(df,
                    param_grid = {
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [3, 5, 7, 10, None],
                        'min_samples_leaf': [1, 3, 5, 7, 10],
                        'class_weight': [None, 'balanced']
                        },
                    target='Churn'
                    ):
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=1
    )
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=1),
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    return best_model, X_test, y_test


# ------------------------------------
# K-means clustering
# ------------------------------------

def elbow_method(
    df,
    maximum_n_Cluster = 6
    ):
    """_summary_
        elbow method -> 최적의 클러스터 수 확인.
    Args:
        df (pd.DataFrame): 데이터셋
        KRange (range(), optional): 확인할 클러스트의 범위. Defaults to range(2,6 + 1).
            -> 그냥 maximum_n_Cluster로 변환
        maximum_n_Cluster (integer, optional): 확인할 클러스트의 최대 수. 그냥 가독성 향상으로..
    """
    X = df.copy()
    scores = []
    KRange = range(1,maximum_n_Cluster+1)
    
    for k in KRange:
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=1
        ).fit(X)
        scores.append(model.inertia_)
        
    plt.figure(figsize=(8, 4))
    plt.plot(KRange, scores, marker='o')
    plt.xlabel("n_cluster")
    plt.ylabel("inertia")
    plt.title("Elbow method")
    plt.grid(True)
    plt.show()


def profileCluster(df, n_cluster=3, n_init = 10):
    """_summary_
        K Means clustering으로 헉습 
    Args:
        df (pd.DataFrame): 데이터셋
        n_cluster (int, optional): 최종 나눌 클러스터의 수. Defaults to 3.
        n_init (int, optional): 군집 중심점 초기화 -> 학습, 이 과정의 수. 
            Default -> 10인 경우, 각 과정에서 가장 좋은 결과를 반환. 높을 수록 좋음. 그러나 느림. 
            -> 100으로 설정해서 다시 결과 비교해보겠습니다..
            
        추가적으로 같은 클러스터의 데이터끼리의 평균값을 내기 때문에, 
        ( 1(YES), 0(NO) ) 와 같이, 범주형 데이터를 나타내는 Boolean 컬럼의 경우, 평균 값으로 계산되어
        0.58, 0.88 등과 같이 계산되어 출력됨.
        (computedValue >= 0.5).astype(int) 로, 바꿔서 다시 살펴보겠습니다..
    Returns:
        각 데이터 Row 별 분류된 cluster 추가된 데이터셋 반환. (학습 결과 데이터셋)
        각 클러스터별 데이터 평균치 데이터셋. (각 클러스터의 특징 파악)
    """
    X = df.copy()
    
    model = KMeans(
        n_clusters=n_cluster,
        random_state=1,
        n_init=n_init
    )
    X['Cluster'] = model.fit_predict(X)
    
    profile = X.groupby('Cluster').mean(numeric_only=True)
    
    print("cluster Profiles: ")
    print(profile)
    
    return X, profile

