"""
그냥 편의를 위한 함수 코드
"""
import pandas as pd
from dataPreprocessing import preprocessing
from train_model import train_decisionTree
from showTheResult import returnThePerformance_decisionTree,showTheDecisionTree
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

def compare_performance_decisionTree_with_parameters(
    df,
    param_grid={
        'convert_No_service_to_No': [True,False],
        'encoding_method': ['one_hot','label']
        }
    ):
    
    results = []
    param_combination = ParameterGrid(param_grid)
    
    for param in param_combination:
        df_param = preprocessing(
            df = df.copy(),
            convert_No_Service_to_No = param['convert_No_service_to_No'],
            method = param['encoding_method']
        )
        
        best_model_param, X_test, y_test = train_decisionTree(df=df_param)
        
        roc_auc, precision, recall, f1 = returnThePerformance_decisionTree(
            bestModel = best_model_param,
            X_test = X_test,
            y_test = y_test
            )
        results.append({
            **param,  
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        })
    results =  pd.DataFrame(results)
    results['Setting'] = results.apply(
        lambda row: f"{row['encoding_method']}, convert={row['convert_No_service_to_No']}",
        axis=1
    )

    metrics = ['precision', 'recall', 'f1', 'roc_auc']
    titles = ['Precision', 'Recall', 'F1 Score', 'ROC-AUC']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        sns.barplot(
            x='Setting',
            y=metric,
            data=results,
            ax=axes[i],
            palette='Set2',
            hue='Setting'
        )
        axes[i].set_title(titles[i])
        axes[i].set_ylim(0, 1)
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=30)
        axes[i].grid(True, axis='y')

    plt.tight_layout()
    plt.show()

