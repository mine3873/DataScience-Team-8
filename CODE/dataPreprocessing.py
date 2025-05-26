import pandas as pd
import numpy as np
from typing import Literal



def preprocessing(
    df,
    dealing_outlier = False,
    showTheOutlierForFeautres = False,
    convert_No_Service_to_No = False,
    run_normalize = False,
    selectBestFeatures = False,
    numOfBestFeatures = 10,
    showingCorr = False,
    method: Literal['one_hot','label','NONE'] = 'one_hot',
    ):
    """_summary_
        data processing function
    Args:
        df (pd.DataFrame): 학습에 사용될 데이터셋
        
        dealing_outlier (bool, optional): outlier 처리 여부. Defaults to False.
        
        showTheOutlierForFeautres (bool, optional): 숫자형 데이터 컬럼의 통계 정보-> boxplot 확인 여부. Defaults to False.
        
        convert_No_Service_to_No (bool, optional): 
            데이터셋 내에서 PhoneService, InternetService 컬럼의 데이터가 'NO'인 경우, 해당 열의 이와 관련된 데이터는 각각
            No phone service, No internet service 이와 같은 값을 가짐. 
            PhoneService, InternetService 의 값이 'YES' 인 경우에는 관련된 컬럼 데이터는 (YES, NO) 두 개의 범주로 나뉨.
            No phone service, No internet service 데이터 값을 NO로 값을 변경하여 데이터를 다룰 것인가? 여부.
            . Defaults to False.
            
        run_normalize (bool, optional): 숫자형 데이터 정규화 처리 여부. Defaults to False.
            Decision Tree에서는 정규화 여부는 상관없지만, K-Means clustering에서는 차이가 컸습니다. 
            
        method (Literal[ one_hot, label, NONE], optional): 모델 학습 위해서는 각 데이터를 숫자형 데이터로 바꿀 필요가 있다.
            범주형 데이터 -> 숫자형 데이터 변환 방식 선택, 
            'one hot encoding' OR 'label encoding' 
            . Defaults to 'one_hot'. 'NONE' mean nothing do.

    Returns:
        pd.DataFrame: 전처리된 데이터셋
    """
    df = df.copy()
    
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    df.drop(columns=['customerID'], inplace=True)
    # drop meaningless features.
    # 불필요한 컬럼 삭제 (고객 ID)
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # change datatype of 'TotalCharges' into nemeric from Object.
    # param, errors, coerce mean:
    # invalid parsing will be set as NaN.
    # ref: https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html
    # 'TotalCharges' 데이터 타입을 Object -> Numeric으로 변경.
    # errors='coerce' 로 설정한 경우, 만약 변경할 값이 '80.99'와 같은 숫자형이 아닌, '88.1asdf'와 같이 숫자 형태가 아닌 경우 -> NaN으로 변환
    
    # 평균 달 지출 feature 추가
    # 'tenure'가 0인 경우 -> 1로 설정하여 에러 방지.
    # 지금 생각하면 'tenure'가 0인 경우 -> df['AvgMonthlySpend'] = 0으로 설정? 
    df['AvgMonthlySpend'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
    
    # 숫자형 데이터 NaN값 처리. 여기선 각 컬럼의 median 값으로 설정.
    # 다른 방식으로도 한번 해봐서 비교해보겠습니다..
    numericFeatures = df.select_dtypes(include=np.number).columns
    df[numericFeatures] = df[numericFeatures].fillna(value=df[numericFeatures].median())
    
    
    if dealing_outlier:
        # -------------------------------------
        # Manage outlier data
        # -------------------------------------
        
        # numeric features 직접 명시
        numericFeatures = ['MonthlyCharges', 'TotalCharges', 'tenure', 'AvgMonthlySpend']
        dealingOutlier(df, numericFeatures, showTheOutlierForFeautres)
    
    # 값이 [YES, NO]와 같이 2개 뿐인 범주형 데이터 -> [1,0]
    # 더 많을 경우 encoding 적용
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
    df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
    df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
    
    if convert_No_Service_to_No:
        """
            EX) 'phoneService' == No인 경우, 
            'phoneService'와 관련된 컴럼들은 각 범주형 데이터를 가지지 않고 No Phone Service로 저장됨.
            | PhoneService | MultipleLines    |
            |--------------|------------------|
            | No           | No phone service |
            | Yes          | No               |
            | Yes          | YES              |
            
            'No phone service' -> 'No' 로 치환할 것인가? 여부 
        """
        features_with_phoneService = [
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
        for column in features_with_phoneService:
            df[column] = df[column].replace("No phone service", "No")
        
        for column in features_with_internetService:
            df[column] = df[column].replace("No internet service", "No")
    
    if run_normalize:
        # -------------------------------------
        # normalize data
        # -------------------------------------
        
        # numeric features 직접 명시, 
        # [1, 0] 인, 기존 Categoric도 정규화 처리된 오류 정정.
        numericFeatures = ['MonthlyCharges', 'TotalCharges', 'tenure', 'AvgMonthlySpend']
        normalizeData(df=df, numericFeatures=numericFeatures)
    
    if method != 'NONE':
        df = categoricEncoding(df=df, method=method)
    
    if selectBestFeatures:
        topFeatures = computeCorrelation(df=df, numOfFeatures=numOfBestFeatures, showingCorr = showingCorr)
        df = df[topFeatures + ['Churn']]
    
    return df

def dealingOutlier(df,
                numericFeatures = ['MonthlyCharges', 'TotalCharges', 'tenure', 'AvgMonthlySpend'],
                showTheOutlierForFeautres = False):  
    
    if showTheOutlierForFeautres:
        """
            show mean, outlier, etc.. info for every numeric features.
            숫자형 데이터 -> boxplot으로 표현.
            그냥 과정 중에 확인 용도로 사용.. 
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

def normalizeData(df,
                numericFeatures = ['MonthlyCharges', 'TotalCharges', 'tenure', 'AvgMonthlySpend']
                ):
    """_summary_
        숫자형 데이터 정규화 -> StandardScaler
        -> 이것도 지금생각하니 다른 방법으로도 해보고 결과 비교해보겠습니다.. 
    
    """
    from sklearn.preprocessing import StandardScaler
    df[numericFeatures] = StandardScaler().fit_transform(df[numericFeatures])

def computeCorrelation(df, target='Churn', numOfFeatures = 10, showingCorr = False):
    X = df.drop(target, axis=1)
    y = df[target]
    
    # --------------------------------
    # univariate selection
    # --------------------------------
    from sklearn.feature_selection import SelectKBest, f_classif
    selectModel = SelectKBest(score_func=f_classif, k='all')
    selectModel.fit(X,y)
    
    univariateScore = pd.DataFrame({
        'Feature': X.columns,
        'F-Score': selectModel.scores_
    }).sort_values(by='F-Score', ascending=False)
    
    if showingCorr:
        print("\nUnivariate Selection: ")
        print(univariateScore.head(numOfFeatures))
    
        # --------------------------------
        # corr heatmap
        # ---------------------------------
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12,12))
        
        corrmat = df.corr(numeric_only=True)
        #cmap='coolwarm'
        sns.heatmap(corrmat[[target]].sort_values(by=target, ascending=False),
                    annot=True,
                    cmap='RdYlGn',
                    linewidths=0.5)
        plt.title(f"corr with {target}")
        plt.tight_layout()
        plt.show()
    return univariateScore['Feature'].head(numOfFeatures).tolist()

def categoricEncoding(df, method: Literal['one_hot','label'] = 'one_hot'):
    """_summary_
        change categoric to numeric with one hot encoding or label encoding
        범주형 데이터 -> 숫자형 데이터로 변환
    Args:
        df (pd.DataFrame): 데이터셋
        method (Literal[one_hot,label], optional): 데이터 변환 방식 'one hot' OR 'label'. Defaults to 'one_hot'.

    Returns:
        pd.DataFrame: 변환 후 데이터셋
    """
    copiedDf = df.copy()
    if method == 'one_hot':
        #copiedDf['Churn'] = copiedDf['Churn'].map({'Yes':1, 'No':0})
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
    
    