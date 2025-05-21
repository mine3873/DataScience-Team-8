import pandas as pd

def printInfo(df):
    """_summary_
        print the information of DataSet.
        - features
        - 5 rows for example
        - data type for each feature
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
    
    print(df.isna().sum())
    

if __name__ == "__main__":
    from load_dataSet import df
    printInfo(df=df)

