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
from dataPreprocessing import preprocessing

df_decisionTree = preprocessing(
    df=train_df,
    dealing_outlier=True,
    convert_No_Service_to_No=True,
    run_normalize=False,
    method='one_hot'
    )

# Decision Tree 학습 
from train_model import train_decisionTree
best_decision_model, X_test, y_test = train_decisionTree(
    df=df_decisionTree,
)

# 모델 평가 
from showTheResult import returnThePerformance_decisionTree,showTheDecisionTree

returnThePerformance_decisionTree(
    bestModel=best_decision_model,
    X_test=X_test,
    y_test=y_test,
    printResult=True
    )
showTheDecisionTree(
    model=best_decision_model,
    feature_names=df_decisionTree.columns,
    max_depth=3,
    fontsize=15
)

# one hot vs label
# No [phone, Internet] Service -> NO ? 
# 여부에 따른 전처리 -> 모델 훈련 -> 평가 비교 자동화 함수
# from AUTOMATIC import compare_performance_decisionTree_with_parameters
#
# compare_performance_decisionTree_with_parameters(df=train_df)


# ------------------------------
# K-Means
# ------------------------------
from train_model import elbow_method, profileCluster

train_df = df.copy()
df = preprocessing(
    df=train_df,
    dealing_outlier=True,
    convert_No_Service_to_No=True,
    run_normalize=True,
    method='one_hot'
)

elbow_method(df=df)
clustered_df, cluster_profile = profileCluster(df, n_cluster=3)

from showTheResult import plot_clusterBoxplots, numOfClient_clusters, show_cluster_with_pca, caculate_ChurnRate_cluster, evaluate_clustering, show_ThePairplot

plot_clusterBoxplots(clustered_df, features=['MonthlyCharges', 'tenure', 'TotalCharges'])
numOfClient_clusters(df=clustered_df)
show_cluster_with_pca(df=clustered_df)
caculate_ChurnRate_cluster(df=clustered_df)
evaluate_clustering(clustered_df)
show_ThePairplot(df=clustered_df, features=['MonthlyCharges', 'tenure', 'TotalCharges'])
