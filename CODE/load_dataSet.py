import pandas as pd

def load_dataSet():
    fileName = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(fileName)
    return df

if __name__ == "__main__":
    fileName = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(fileName)
    print()