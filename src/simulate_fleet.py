from src.simulate import generate_normal_data, inject_disk_degradation, inject_io_error_burst, inject_latency_spike, inject_node_failure
import numpy as np
import pandas as pd
import argparse


#Generate synthetic data for a fleet of devices
def generate_device(device_id, tier, n_rows, seed):
    """
    Generate synthetic data for a single device.
    """
    df = generate_normal_data(n_rows, seed)
    used = np.zeros(len(df), dtype=int)
    df,used = inject_disk_degradation(df, used, n=1)
    df,used = inject_io_error_burst(df, used, n=1)
    df,used = inject_latency_spike(df, used, n=1)
    df,used = inject_node_failure(df, used, n=1)
    df['device_id'] = device_id
    df['tier'] = tier
    return df

def build_fleet(n_devices, n_rows, out_path):
    """
    Generate synthetic data for a fleet of devices.
    """
    all_dfs = []
    for device_id in range(n_devices):
        d_id = f"node_{device_id:02d}"
        tier = "production" if device_id < n_devices * 0.3 else "internal"  # 30% production, 70% internal
        all_dfs.append(generate_device(d_id, tier, n_rows, seed=100+device_id)) #unique seed for each device to ensure different random data
    
    fleet_df = pd.concat(all_dfs, ignore_index=True)
    # Shuffle the rows to mimic real-world data distribution similar to underorderd data lake
    fleet_df = fleet_df.sample(frac=1, random_state=32).reset_index(drop=True)
    fleet_df.to_parquet(out_path, partition_cols=['device_id'], coerce_timestamps='us', allow_truncated_timestamps=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data for a fleet of devices.")
    parser.add_argument("--n_devices", type=int, default=10, help="Number of devices to simulate.")
    parser.add_argument("--n_rows", type=int, default=10000, help="Number of rows per device.")
    parser.add_argument("--out_path", type=str, default="data/fleet", help="Output path for the generated data.")
    
    args = parser.parse_args()
    
    build_fleet(args.n_devices, args.n_rows, args.out_path)