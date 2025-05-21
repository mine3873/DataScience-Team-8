
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------
# Decision Tree
# ------------------------------------
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
def returnThePerformance_decisionTree(bestModel, X_test, y_test, printResult = False, threshold = .5):
    """_summary_
        evaluate model.
    Args:
        bestModel (DecisionTreeClassifier): model
        X_test (arraylike): features for prediction
        y_test (arraylike): target for prediction
        printResult (bool, optional): whether print the result. Defaults to False.
        threshold (float, optional): threshold for predicted value, not used now. Defaults to .5.

    Returns:
        each scores from model.
    """
    y_pred = bestModel.predict(X_test)
    y_proba = bestModel.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    if printResult:
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        # -----------------------------
        # ROC AAUC:
        # how well the model predict churn or not
        # -----------------------------
        print(f"ROC-AUC: {roc_auc:.4f}")

        # -----------------------------
        # precision:
        # TP / (TP + FP), 
        # how many practically really truly churn from predicted churn with model.
        # -----------------------------
        print(f"Precision: {precision:.4f}")

        # -----------------------------
        # recall:
        # TP / (TP + FN), 
        # how many correctly prediction with model from practivally churn.
        # -----------------------------
        print(f"Recall: {recall:.4f}")

        # -----------------------------
        # F1 Score:
        # 2 * precision * Recall / (Precision + Recall), 
        # how many correctly prediction with model from practivally churn.
        # -----------------------------
        print(f"F1 Score: {f1:.4f}", )
    return roc_auc, precision, recall, f1


from sklearn.tree import plot_tree
def showTheDecisionTree(model,
                        feature_names,
                        className=['No','Yes'],
                        max_depth=10,
                        fontsize=20):
    plt.figure(figsize=(20,10))
    #-------------------------
    # each Node contain:
    # boolean expression -> ex. A <= .5
    # entropy
    # smaples : num of data touch to current node.
    # value : [a,b] -> [num of churn == 0, num of churn == 1]
    #-------------------------
    plot_tree(model,
              feature_names=feature_names,
              filled=True,
              rounded=True,
              max_depth=max_depth,
              fontsize=fontsize)
    
    plt.tight_layout()
    plt.show()

# ------------------------------------
# K-means clustering
# ------------------------------------

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def plot_clusterBoxplots(df, clusterFeature='Cluster', features=None):
    """_summary_
        print data statistic info in each feature
    Args:
        df (pd.DataFrame): data
        clusterFeature (str, optional): Cluster columns. Defaults to 'Cluster'.
        features (list, optional): feature list to check data info. Defaults to None.
    """
    if features is None:
        features = df.select_dtypes(include='number').drop(columns=[clusterFeature]).columns
    rows ,cols = 1, len(features)
    
    fig, axes = plt.subplots(nrows=rows,ncols=cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        sns.boxplot(x=clusterFeature, y=feature, data=df, ax=axes[idx])
        axes[idx].set_title(f"{feature}")
    
    plt.tight_layout()
    plt.show()
    
def numOfClient_clusters(df, clusterFeature='Cluster'):
    sns.countplot(x=clusterFeature, data=df)
    plt.title("num of clients in each cluster")
    plt.tight_layout()
    plt.show()


def show_cluster_with_pca(df, clusterFeature='Cluster'):
    X = df.copy().drop(columns=[clusterFeature])
    X_pca = PCA(n_components=2).fit_transform(X)
    
    plt.figure(figsize=(12,8))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df[clusterFeature], palette='Set2', s=50)
    plt.title("Customer Clusters")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()
    
def caculate_ChurnRate_cluster(
        df,
        clusterFeature='Cluster',
        churnFeature='Churn'
    ):
    churnRate = df.groupby(clusterFeature)[churnFeature].mean()
    
    plt.figure(figsize=(12,8))
    sns.barplot(x=churnRate.index, y=churnRate.values, palette="Set2", hue=churnRate.index)
    
    plt.tight_layout()
    plt.show()
    
    print("Churn rate:")
    for cluster, rate in churnRate.items():
        print(f"cluster-{cluster}: {rate:.3f}%")
        
def evaluate_clustering(df, cluster_col='Cluster'):
    X = df.drop(columns=[cluster_col])
    labels = df[cluster_col]

    score = silhouette_score(X, labels)
    print(f"Silhouette Score: {score:.4f}")

    return score

def show_ThePairplot(df, cluster_col='Cluster', features=None):  
    sns.pairplot(df[features + [cluster_col]], hue=cluster_col, palette='Set2')
    plt.suptitle("Pairplot by Cluster", y=1.02)
    plt.show()