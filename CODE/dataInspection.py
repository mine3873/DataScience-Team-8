import pandas as pd

def printInfo(df):
    """_summary_
        데이터셋의 정보 출력 함수
        - 데이터 내 feature 
        - 5 예시 데이터
        - 각 feature의 데이터 타입
    """
    print("the features of DataSet:")
    print(df.columns)
    
    print("\nfirst 5 rows of DataSet:")
    print(df.head())
    
    print("\nDataTypes of DataSet:")
    print(df.dtypes)
    # 'TotalCharges' feature is nemeric, float. 
    # but the data type of this, set Object.
    # So need to change into nemeric
    """
        'TotalCharges' 의 데이터는 숫자형 데이터의 형태이지만, 
        데이터셋 내에서는 Object로 설정되어있음.
        -> preprocessing 단계에서 변환
        
    """
    # 위와 같은 이유로 각 컬럼의 NaN 의 수는 0으로 출력. -> 문자열 'NaN'으로 인식되기 때문.
    # 
    print(f"Num Of NaN in each feature: \n{df.isna().sum()}")
    
    # 숫자형 데이터 통계적 정보 확인 
    print(f"Statistical Information: \n{df.describe()}")
    
    
    

if __name__ == "__main__":
    from load_dataSet import df
    printInfo(df=df)

