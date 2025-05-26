import pandas as pd
from load_dataSet import load_dataSet
from dataInspection import printInfo
from dataPreprocessing import preprocessing
from train_model import train_decisionTree, train_randomForest, elbow_method, profileCluster
from showTheResult import (
    returnThePerformance_decisionTree,showTheDecisionTree, find_bsetThreshold,
    plot_clusterBoxplots, numOfClient_clusters, show_cluster_with_pca, 
    caculate_ChurnRate_cluster, evaluate_clustering, show_ThePairplot
)
from AUTOMATIC import compare_performance_decisionTree_with_parameters
from typing import Literal

class DataScience:
    def __init__(self):
        self.original_df = None
        self.df = None
        self.model = None
        self.model_name: Literal['decisionTree','randomForest','kMeans'] = None
        self.X_test = None
        self.y_test = None
        self.clustered_df = None
        self.cluster_profile = None
        self.bestThreshold = .5
    
    def load_dataSet(self):
        self.original_df = load_dataSet()
    
    def print_statistical(self):
        printInfo(self.original_df)
    
    def preprocessing(self, **kwargs):
        self.df = preprocessing(
            df=self.original_df,
            **kwargs
        )
        return self.df
    
    def trainModel(self, 
                   model: Literal['decisionTree','randomForest','kMeans'] = 'decisionTree',
                   useSmoth = False,
                   test_size = 0.2,
                   n_cluster = 3):
        self.model_name = model
        if model == 'decisionTree':
            self.model, self.X_test, self.y_test = train_decisionTree(self.df,
                                                                      useSmoth=useSmoth,
                                                                      test_size=test_size)
        elif model == 'randomForest':
            self.model, self.X_test, self.y_test = train_randomForest(self.df,
                                                                      useSmoth=useSmoth,
                                                                      test_size=test_size)
        elif model == 'kMeans':
            elbow_method(self.df)
            self.clustered_df, self.cluster_profile = profileCluster(self.df, n_cluster=n_cluster)
            
    def evaluate(self,
                printResult = False,
                tuneThreshold = False):
        if tuneThreshold:
            self.bestThreshold = find_bsetThreshold(
                bestModel=self.model,
                X_test=self.X_test,
                y_test=self.y_test)
        return returnThePerformance_decisionTree(
            bestModel=self.model,
                X_test=self.X_test,
                y_test=self.y_test,
                printResult=printResult,
                threshold=self.bestThreshold
            )
        
    def showTree(self,
                max_depth=3,
                fontsize=15):
        if self.model_name == "decisionTree":
            showTheDecisionTree(
                model=self.model,
                feature_names=self.df.columns,
                max_depth=max_depth,
                fontsize=fontsize
            )
        else:
            print()
            
    def analysis_Kmeans(self):
        plot_clusterBoxplots(self.clustered_df, features=['MonthlyCharges', 'tenure', 'TotalCharges'])
        numOfClient_clusters(df=self.clustered_df)
        show_cluster_with_pca(df=self.clustered_df)
        caculate_ChurnRate_cluster(df=self.clustered_df)
        evaluate_clustering(self.clustered_df)
        show_ThePairplot(df=self.clustered_df, features=['MonthlyCharges', 'tenure', 'TotalCharges'])
    
    def find_best_parameter_decisionTree(self,
                                        param_grid={
                                            'convert_No_service_to_No': [True, False],
                                            'encoding_method': ['one_hot','label'],
                                            'selectBestFeatures': [True,False],
                                            'numOfBestFeatures':  [5,10,15,20,25],
                                            'useSmoth': [True, False],
                                        },
                                        showTheGraph = False):
        compare_performance_decisionTree_with_parameters(
            df=self.original_df,
            param_grid=param_grid,
            showTheGraph=showTheGraph
        )
        
