import pandas as pd
import numpy as np
from typing import Literal

def preprocessing(
    df,
    dealing_outlier = False,
    showTheOutlierForFeautres = False,
    convert_No_Service_to_No = False,
    run_normalize = False,
    method: Literal['one_hot','label','NONE'] = 'one_hot',
    ):
    """_summary_
        data processing function
    Args:
        df (pd.DataFrame): 데이터
        dealing_outlier (bool, optional): whether to manage outlier. Defaults to False.
        showTheOutlierForFeautres (bool, optional): display boxPlot graph with outlier in each numeric feature. Defaults to False.
        convert_No_Service_to_No (bool, optional):whether to change No (phone, internet) service -> No. Defaults to False.
        run_normalize (bool, optional): whether to nomalize numeric data. Defaults to False.
        method (Literal[one_hot,label,NONE], optional): method to change categoric to numeric. Defaults to 'one_hot'. 'NONE' mean nothing do.

    Returns:
        pd.DataFrame: preprocessed dataFrame
    """
    
    df.drop(columns=['customerID'], inplace=True)
    # drop meaningless features.
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # change datatype of 'TotalCharges' into nemeric from Object.
    # param, errors, coerce mean:
    # invalid parsing will be set as NaN.
    # ref: https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html
    
    # 평균 달 지출 feature 추가
    df['AvgMonthlySpend'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
    df['AvgMonthlySpend']
    
    numericFeatures = df.select_dtypes(include=np.number).columns
    df[numericFeatures] = df[numericFeatures].fillna(value=df[numericFeatures].median())
    
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(bool)
    # change the datatype of 'SeniorCitizen, int64, to boolean
    
    
    
    if dealing_outlier:
        # -------------------------------------
        # Manage outlier data
        # -------------------------------------
        dealingOutlier(df, showTheOutlierForFeautres)
    
    if convert_No_Service_to_No:
        feature_with_phoneService = [
            'MultipleLines'
        ]
        
        features_with_internetService = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies"
        ]
        for column in feature_with_phoneService:
            df[column] = df[column].replace("No phone service","No")
        
        for column in features_with_internetService:
            df[column] = df[column].replace("No internet service","No")
    
    if run_normalize:
        # -------------------------------------
        # normalize data
        # -------------------------------------
        normalizeData(df=df)
    
    if method != 'NONE':
        df = categoricEncoding(df=df, method=method)
    
    return df


def dealingOutlier(df, showTheOutlierForFeautres):  
    
    numericFeatures = df.select_dtypes(include=np.number).columns.to_list()
    # list for feature names in which data is numeric
    
    if showTheOutlierForFeautres:
        """
            show mean, outlier, etc.. info for every numeric features.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        n_cols = len(numericFeatures)
        fig, axes = plt.subplots(ncols=n_cols, nrows=1, figsize=(4 * n_cols, 4))
        axes = axes.flatten()
        for idx, col in enumerate(numericFeatures):
            sns.boxplot(x=df[col], ax=axes[idx])
            axes[idx].set_title(f"{col}")
            axes[idx].grid(True)
        plt.tight_layout()
        plt.show()
    
    # change outlier data to median
    # outlier is not in between 
    for col in numericFeatures:
        Q1 = df[col].quantile(1/4)
        Q3 = df[col].quantile(3/4)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5 * IQR
        upperBound = Q3 + 1.5 * IQR

        median = df[col].median()
        df.loc[df[col] < lowerBound, col] = median
        df.loc[df[col] > upperBound, col] = median
    

def normalizeData(df):
    numericFeatures = df.select_dtypes(include=np.number).columns.to_list()
    
    from sklearn.preprocessing import StandardScaler
    df[numericFeatures] = StandardScaler().fit_transform(df[numericFeatures])
    

def dealingCategoric(df, showTheCategoricValueForFeatures, target='Churn'):
    df[target] = df[target].map({'Yes':1, 'No':0})
    
    categoricFeatures = df.select_dtypes(include='object').columns
    
    if showTheCategoricValueForFeatures:
        for col in categoricFeatures:
            print(f"[{col}]'s values:")
            print(df[col].unique())
            print("-"*40)
    
    categoricFeatures = categoricFeatures.to_list()
    df = pd.get_dummies(df, columns=categoricFeatures, drop_first=True)
    # change the values from categoric into numeric with one hot encoding method
    
    return df


def computeCorrelation(df, target='Churn'):
    X = df.drop(target, axis=1)
    y = df[target]
    
    # --------------------------------
    # univariate selection
    # --------------------------------
    from sklearn.feature_selection import SelectKBest
    selectModel = SelectKBest(k='all')
    selectModel.fit(X,y)
    
    univariateScore = pd.DataFrame({
        'Feature': X.columns,
        'F-Score': selectModel.scores_
    }).sort_values(by='F-Score', ascending=False)
    
    print("\nUnivariate Selection: ")
    print(univariateScore.head(10))
    
    # --------------------------------
    # corr heatmap
    # ---------------------------------
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(12,12))
    
    corrmat = df.corr(numeric_only=True)
    #cmap='coolwarm'
    sns.heatmap(corrmat[[target]].sort_values(by=target, ascending=False), annot=True, cmap='RdYlGn', linewidths=0.5)
    
    """
    corrmat = df.corr(numeric_only=True)
    top_corr_features = corrmat.index
    sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='RdYlGn')
    """
    plt.tight_layout()
    plt.show()


def categoricEncoding(df, method: Literal['one_hot','label'] = 'one_hot'):
    """_summary_
        change categoric to numeric with one hot encoding or label encoding
    Args:
        df (pd.DataFrame): data
        method (Literal[one_hot,label], optional): method. Defaults to 'one_hot'.

    Returns:
        pd.DataFrame: encoded DataFrame
    """
    copiedDf = df.copy()
    if method == 'one_hot':
        import pandas as pd
        
        copiedDf['Churn'] = copiedDf['Churn'].map({'Yes':1, 'No':0})
        categoricFeatures = copiedDf.select_dtypes(include='object').columns
        categoricFeatures = categoricFeatures.to_list()
        copiedDf = pd.get_dummies(copiedDf, columns=categoricFeatures, drop_first=True)
    elif method == 'label':
        from sklearn.preprocessing import LabelEncoder
        
        for col in copiedDf.select_dtypes(include='object').columns:
            copiedDf[col] = LabelEncoder().fit_transform(copiedDf[col])
    return copiedDf



if __name__ == "__main__":
    from load_dataSet import df
    p_df = preprocessing(df=df)
    
    