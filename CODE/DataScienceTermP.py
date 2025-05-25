import pandas as pd
from load_dataSet import df

# statistical info 
from dataInspection import printInfo
printInfo(df)

# 학습 목적 데이터 선언
train_df = df.copy()


# ------------------------------
# Decision Tree classifier
# with one hot 
# ------------------------------

# 데이터 전처리
from dataPreprocessing import preprocessing, computeCorrelation

df_decisionTree = preprocessing(
    df=train_df,
    dealing_outlier=True,
    convert_YESNO_TO_01=True,
    convert_No_Service_to_No=False,
    run_normalize=False,
    method='one_hot'
    )

# Decision Tree 학습 
from train_model import train_decisionTree
best_decision_model, X_test, y_test = train_decisionTree(
    df=df_decisionTree,
)


# 모델 평가 
from showTheResult import returnThePerformance_decisionTree,showTheDecisionTree, find_bsetThreshold

# 학습된 Decision Tree 모델의 성능 평가
bestThreshold = find_bsetThreshold(bestModel=best_decision_model, X_test=X_test, y_test=y_test)

returnThePerformance_decisionTree(
    bestModel=best_decision_model,
    X_test=X_test,
    y_test=y_test,
    printResult=True,
    threshold=bestThreshold
    )

# 학습된 Decision Tree 모델 시각화
showTheDecisionTree(
    model=best_decision_model,
    feature_names=df_decisionTree.columns,
    max_depth=3,
    fontsize=15
)


"""
# one hot vs label
# No [phone, Internet] Service -> NO ? 
# 여부에 따른 전처리 -> 모델 훈련 -> 평가 비교 자동화 함수
from AUTOMATIC import compare_performance_decisionTree_with_parameters

compare_performance_decisionTree_with_parameters(
    df=train_df,
    param_grid={
        'convert_No_service_to_No': [False],
        'encoding_method': ['one_hot'],
        'selectBestFeatures': [True],
        'numOfBestFeatures':  [5,10,15,20,25],
        }
    )
"""

"""
# ------------------------------
# K-Means
# ------------------------------
from train_model import elbow_method, profileCluster

from dataPreprocessing import preprocessing
train_df = df.copy()
df = preprocessing(
    df=train_df,
    dealing_outlier=True,
    convert_YESNO_TO_01=True,
    convert_No_Service_to_No=True,
    run_normalize=True,
    method='one_hot'
)

elbow_method(df=df)
clustered_df, cluster_profile = profileCluster(df, n_cluster=3)

from showTheResult import plot_clusterBoxplots, numOfClient_clusters, show_cluster_with_pca, caculate_ChurnRate_cluster, evaluate_clustering, show_ThePairplot

# clustered 된 데이터 셋 내에서 features 값들 간에 boxplot을 통한 비교
plot_clusterBoxplots(clustered_df, features=['MonthlyCharges', 'tenure', 'TotalCharges'])

# 각 cluster의 고객 수 출력
numOfClient_clusters(df=clustered_df)

# cluster된 결과를 2차원으로 출력 with PCA,
show_cluster_with_pca(df=clustered_df)

# 각 cluster 별 churn rate 출력
caculate_ChurnRate_cluster(df=clustered_df)

# cluster 평가
evaluate_clustering(clustered_df)

# features 들의 pairplot 출력
show_ThePairplot(df=clustered_df, features=['MonthlyCharges', 'tenure', 'TotalCharges'])
"""