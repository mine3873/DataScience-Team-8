#from DataScienceTermP import preprocessed_df

#df = preprocessed_df
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ------------------------------------
# Decision Tree
# ------------------------------------
def train_decisionTree(df, 
                        param_grid = {
                            'criterion': ['gini', 'entropy'],
                            'max_depth': [3, 5, 7, 10, None],
                            'min_samples_leaf': [1, 3, 5, 7, 10]
                            },
                        target='Churn'
                        ):
    """_summary_
        Given some parameters for training decision tree model, 
        train decision tree models with optional parameters and return the best model.
        
    Args:
        df (pd.DataFrame): 
            dataSet for training decision tree model.
        method (Literal['one_hot','label'], optional): 
            the method how to change the categoric value of dataSet to numeric data. Defaults to 'one_hot'.
        param_grid (dict, optional): 
            parameters for training decision tree model. among these decision model, the best performance model is selected. 
                Defaults to { 'criterion': ['gini', 'entropy'],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_leaf': [1, 5, 10] }.
        show_performance (bool, optional): 
            whether to print the performance of the best model. Defaults to False.
                the result values are ROC AUC, Precision, Recall, F1 Score
    """
    X = df.drop(target, axis=1)
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=1
    )
    
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

# ------------------------------------
# K-means clustering
# ------------------------------------

def elbow_method(
    df,
    KRange=range(2,6 + 1)
    ):
    X = df.copy()
    scores = []
    
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

def profileCluster(df, n_cluster=3):
    X = df.copy()
    
    model = KMeans(
        n_clusters=n_cluster,
        random_state=1,
        n_init=10
    )
    X['Cluster'] = model.fit_predict(X)
    
    profile = X.groupby('Cluster').mean(numeric_only=True)
    
    print("cluster Profiles: ")
    print(profile)
    
    return X, profile

