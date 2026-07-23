import pandas as pd

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts features from the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing raw data.

    Returns:
        pd.DataFrame: DataFrame with extracted features.
    """
    
    df["iops_data_read_rolling_mean_5m"]=df['userDataReadIops'].rolling(window=5).mean()
    df["iops_data_read_rolling_std_5m"]=df['userDataReadIops'].rolling(window=5).std()
    df["iops_data_read_pct_change_1h"]=df['userDataReadIops'].pct_change(periods=60)


    df["iops_data_write_rolling_mean_5m"]=df['userDataWriteIops'].rolling(window=5).mean()
    df["iops_data_write_rolling_std_5m"]=df['userDataWriteIops'].rolling(window=5).std()
    df["iops_data_write_pct_change_1h"]=df['userDataWriteIops'].pct_change(periods=60)


    df["read_latency_rolling_mean_5m"]=df['readLatencyMs'].rolling(window=5).mean()
    df["read_latency_rolling_std_5m"]=df['readLatencyMs'].rolling(window=5).std()
    df["read_latency_pct_change_1h"]=df['readLatencyMs'].pct_change(periods=60)

    df["write_latency_rolling_mean_5m"]=df['writeLatencyMs'].rolling(window=5).mean()
    df["write_latency_rolling_std_5m"]=df['writeLatencyMs'].rolling(window=5).std()
    df["write_latency_pct_change_1h"]=df['writeLatencyMs'].pct_change(periods=60)
    return df