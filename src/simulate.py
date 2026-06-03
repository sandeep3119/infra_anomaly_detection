import numpy as np
import pandas as pd



def generate_normal_data(n_rows, seed):
    np.random.seed(seed)
    
    # Create timestamps with 1-minute intervals starting from 2026-01-01
    timestamps = pd.date_range(start='2026-01-01', periods=n_rows, freq='1min')
    
    # Extract hour of day for time-of-day pattern (0-23)
    hours = timestamps.hour + timestamps.minute / 60
    
    # Create time-of-day pattern (sinusoidal, peaks around noon)
    time_pattern = np.sin((hours - 6) * np.pi / 12)
    
    # Generate data for each column
    df = pd.DataFrame({
        'timestamp': timestamps,
        'userDataReadIops': np.clip(5000 + time_pattern * 1000 + np.random.normal(0, 200, n_rows), 0,None),
        'userDataWriteIops': np.clip(3000 + time_pattern * 600 + np.random.normal(0, 150, n_rows), 0,None),
        'readLatencyMs': np.clip(2 + np.random.normal(0, 0.5, n_rows), 0.1, None),
        'writeLatencyMs': np.clip(3 + np.random.normal(0, 0.5, n_rows), 0.1, None),
        'cpuPercent': np.clip(40 + np.random.normal(0, 5, n_rows),0,100),
        'memoryPercent': np.clip(60 + np.random.normal(0, 5, n_rows),0,100),
        'label': 0
    })
    
    return df


def inject_disk_degradation(df, used_indices):
    """
    Inject disk degradation anomaly: gradual IOPS decline over 500 rows.
    Label = 1
    """
    df = df.copy()
    n_rows = len(df)
    window_size = 500
    
    # Pick random start position, ensuring window fits within dataframe
    start_idx = 0
    end_idx = 0
    window_len = 0
    while True:
            start_idx = np.random.randint(0, max(1, n_rows - window_size))
            end_idx = min(start_idx + window_size, n_rows)
            if np.all(used_indices[start_idx:end_idx] == 0):
                used_indices[start_idx:end_idx] = 1
                break

    # Create gradual decline factor (0.2 to 1.0, decreasing over window)
    window_len = end_idx - start_idx
    decline = np.linspace(1.0, 0.2, window_len)
    
    # Apply degradation to IOPS
    df.loc[start_idx:end_idx-1, 'userDataReadIops'] *= decline
    df.loc[start_idx:end_idx-1, 'userDataWriteIops'] *= decline
    df.loc[start_idx:end_idx-1, 'label'] = 1
    
    return df,used_indices


def inject_io_error_burst(df,used_indices):
    """
    Inject IO error burst anomaly: sudden spike in write errors.
    Label = 2
    """
    df = df.copy()
    n_rows = len(df)
    window_size = np.random.randint(100, 201)  # Random window 100-200 rows
    
    # Pick random start position
    start_idx = 0
    end_idx = 0
    
    while True:
            start_idx = np.random.randint(0, max(1, n_rows - window_size))
            end_idx = min(start_idx + window_size, n_rows)
            if np.all(used_indices[start_idx:end_idx] == 0):
                used_indices[start_idx:end_idx] = 2
                break
    
    # Spike in write IOPS (errors manifested as high write activity)
    df.loc[start_idx:end_idx-1, 'userDataWriteIops'] *= 3.0
    df.loc[start_idx:end_idx-1, 'writeLatencyMs'] *= 5.0
    df.loc[start_idx:end_idx-1, 'label'] = 2
    
    return df,used_indices


def inject_latency_spike(df,used_indices):
    """
    Inject latency spike anomaly: sudden jump in read/write latency.
    Label = 3
    """
    df = df.copy()
    n_rows = len(df)
    window_size = np.random.randint(100, 201)  # Random window 100-200 rows
    
    # Pick random start position
    start_idx = 0
    end_idx = 0
    while True:
            start_idx = np.random.randint(0, max(1, n_rows - window_size))
            end_idx = min(start_idx + window_size, n_rows)
            if np.all(used_indices[start_idx:end_idx] == 0):
                used_indices[start_idx:end_idx] = 3
                break
    
    # Spike in latency
    df.loc[start_idx:end_idx-1, 'readLatencyMs'] += np.random.normal(10, 2, end_idx - start_idx)
    df.loc[start_idx:end_idx-1, 'writeLatencyMs'] += np.random.normal(15, 2, end_idx - start_idx)
    df.loc[start_idx:end_idx-1, 'label'] = 3
    
    return df, used_indices


def inject_node_failure(df, used_indices):
    """
    Inject node failure anomaly: metrics drop to near-zero.
    Label = 4
    """
    df = df.copy()
    n_rows = len(df)
    window_size = np.random.randint(100, 201)  # Random window 100-200 rows
    
    # Pick random start position
    start_idx = 0
    end_idx = 0
    while True:
            start_idx = np.random.randint(0, max(1, n_rows - window_size))
            end_idx = min(start_idx + window_size, n_rows)
            if np.all(used_indices[start_idx:end_idx] == 0):
                used_indices[start_idx:end_idx] = 4
                break
    
    # Drop metrics to near-zero
    df.loc[start_idx:end_idx-1, 'userDataReadIops'] = np.random.uniform(0, 50, end_idx - start_idx)
    df.loc[start_idx:end_idx-1, 'userDataWriteIops'] = np.random.uniform(0, 50, end_idx - start_idx)
    df.loc[start_idx:end_idx-1, 'cpuPercent'] = np.random.uniform(0, 5, end_idx - start_idx)
    df.loc[start_idx:end_idx-1, 'memoryPercent'] = np.random.uniform(0, 5, end_idx - start_idx)
    df.loc[start_idx:end_idx-1, 'label'] = 4
    
    return df,used_indices