"""
그냥 편의를 위한 함수 코드
"""
import pandas as pd
from dataPreprocessing import preprocessing
from train_model import train_decisionTree
from showTheResult import returnThePerformance_decisionTree, find_bsetThreshold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

def compare_performance_decisionTree_with_parameters(
    df,
    param_grid={
        'convert_No_service_to_No': [True, False],
        'encoding_method': ['one_hot', 'label'],
        'selectBestFeatures': [True,False],
        'numOfBestFeatures':  [5,10,15,20,25],
        'useSmoth': [True, False]
        },
    showTheGraph = False
    ):
    
    results = []
    param_combination = ParameterGrid(param_grid)
    
    for param in param_combination:
        df_param = preprocessing(
            df = df.copy(),
            convert_No_Service_to_No = param['convert_No_service_to_No'],
            method = param['encoding_method'],
            selectBestFeatures=param['selectBestFeatures'],
            numOfBestFeatures=param['numOfBestFeatures']
        )
        
        best_model_param, X_test, y_test = train_decisionTree(df=df_param,
                                                              useSmoth=param['useSmoth'])
        
        bestThreshold = find_bsetThreshold(bestModel=best_model_param, X_test=X_test, y_test=y_test)
        
        roc_auc, precision, recall, f1 = returnThePerformance_decisionTree(
            bestModel = best_model_param,
            X_test = X_test,
            y_test = y_test,
            threshold=bestThreshold
            )
        results.append({
            **param,  
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        })
        
    results = pd.DataFrame(results)
    results['Setting'] = results.apply(
        lambda row: f"{row['encoding_method']}, conv={row['convert_No_service_to_No']}, SMOTH={row['useSmoth']}, topN={row['numOfBestFeatures']}" 
        if row['selectBestFeatures'] else f"{row['encoding_method']}, conv={row['convert_No_service_to_No']}, SMOTH={row['useSmoth']}, all",
        axis=1
    )

    if showTheGraph:
        metrics = ['precision', 'recall', 'f1', 'roc_auc']
        titles = ['Precision', 'Recall', 'F1 Score', 'ROC-AUC']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            sns.barplot(
                x='Setting',
                y=metric,
                data=results,
                ax=ax,
                palette='Set3',
                hue='Setting'
            )
            ax.set_title(titles[i], fontsize=14)
            ax.set_ylim(0, 1)
            ax.set_xlabel('')
            ax.set_ylabel(metric.capitalize())
            ax.tick_params(axis='x', rotation=30)
            ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.2f}', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='bottom', fontsize=9)

        plt.suptitle("Decision Tree Performance with each params", fontsize=20)
        plt.tight_layout()
        plt.show()
    
    print("\nBest params by F1:\n")
    display_cols = ['Setting', 'precision', 'recall', 'f1', 'roc_auc']
    print(results.sort_values(by='f1', ascending=False)[display_cols].head(5))

    #results.to_csv("decision_tree_performance.csv", index=False, encoding='utf-8-sig')