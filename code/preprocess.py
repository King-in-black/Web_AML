import pandas as pd
import numpy as np

# Mapping from MAC address to sensor ID
sensor_map = {
    'B0:B2:1C:8F:6B:C4': 1,
    'CC:7B:5C:AF:BD:F4': 2,
    '08:B6:1F:75:ED:80': 3,
    '08:D1:F9:DC:8F:80': 4
}


def process_single_csv(input_file):
    """
    Read a single CSV file and split it into multiple samples using blank lines.
    For each sample, group the data by timestamp and sensor MAC, fill missing data,
    and normalize the number of time steps to 20.

    Each time step contains 5 features: Lin_Acc_X, Lin_Acc_Y, Lin_Acc_Z, Roll, Pitch.
    Each processed sample is flattened into a 1D array of 400 features.
    """
    df = pd.read_csv(
        input_file,
        header=0,
        na_values=[''],
        keep_default_na=True,
        skip_blank_lines=False  # Keep blank lines as sample separators
    )

    # Split the CSV into samples based on blank lines
    samples = []
    current_sample = []
    for idx, row in df.iterrows():
        if pd.isna(row['Lin_Acc_X']) and pd.isna(row['Lin_Acc_Y']) and pd.isna(row['Lin_Acc_Z']):
            if current_sample:
                samples.append(pd.DataFrame(current_sample))
                current_sample = []
        else:
            current_sample.append(row)
    if current_sample:
        samples.append(pd.DataFrame(current_sample))

    processed_samples = []
    for sample_df in samples:
        # Sort timestamps
        timestamps = sorted(sample_df['Timestamp'].unique(),
                            key=lambda x: pd.to_datetime(x, format='%Y:%m:%d %H:%M:%S:%f'))

        # Initialize per-sensor dataframes
        sensor_data = {
            1: pd.DataFrame(index=timestamps, columns=['Lin_Acc_X', 'Lin_Acc_Y', 'Lin_Acc_Z', 'Roll', 'Pitch']),
            2: pd.DataFrame(index=timestamps, columns=['Lin_Acc_X', 'Lin_Acc_Y', 'Lin_Acc_Z', 'Roll', 'Pitch']),
            3: pd.DataFrame(index=timestamps, columns=['Lin_Acc_X', 'Lin_Acc_Y', 'Lin_Acc_Z', 'Roll', 'Pitch']),
            4: pd.DataFrame(index=timestamps, columns=['Lin_Acc_X', 'Lin_Acc_Y', 'Lin_Acc_Z', 'Roll', 'Pitch'])
        }

        # Fill in each sensor's data by timestamp
        groups = sample_df.groupby('Timestamp')
        for t, group in groups:
            for _, row in group.iterrows():
                mac = row['MAC']
                if mac in sensor_map:
                    sensor_id = sensor_map[mac]
                    sensor_data[sensor_id].loc[t] = row[['Lin_Acc_X', 'Lin_Acc_Y', 'Lin_Acc_Z', 'Roll', 'Pitch']].values

        # Convert to numeric and fill missing values
        for sensor_id in sensor_data:
            sensor_data[sensor_id] = sensor_data[sensor_id].apply(pd.to_numeric, errors='coerce')
            sensor_data[sensor_id] = sensor_data[sensor_id].interpolate(method='linear', limit_direction='both')
            sensor_data[sensor_id] = sensor_data[sensor_id].fillna(method='ffill').fillna(method='bfill')

        # Normalize to exactly 20 time steps
        num_timesteps = len(timestamps)
        if num_timesteps < 20:
            for sensor_id in sensor_data:
                last_row = sensor_data[sensor_id].iloc[-1]
                num_to_add = 20 - num_timesteps
                pad_index = [f"{timestamps[-1]}_pad{i}" for i in range(1, num_to_add + 1)]
                pad_df = pd.DataFrame([last_row] * num_to_add, index=pad_index,
                                      columns=['Lin_Acc_X', 'Lin_Acc_Y', 'Lin_Acc_Z', 'Roll', 'Pitch'])
                sensor_data[sensor_id] = pd.concat([sensor_data[sensor_id], pad_df])
        elif num_timesteps > 20:
            for sensor_id in sensor_data:
                sensor_data[sensor_id] = sensor_data[sensor_id].iloc[:20]

        # Flatten the sample into a 1D array of shape (4 × 20 × 5 = 400)
        sample_flat = []
        for sensor_id in [1, 2, 3, 4]:
            sample_flat.extend(sensor_data[sensor_id].values.flatten())
        processed_samples.append(sample_flat)

    # Build the output DataFrame
    col_names = []
    for sensor_id in [1, 2, 3, 4]:
        for t in range(1, 21):
            for axis in ['Lin_Acc_X', 'Lin_Acc_Y', 'Lin_Acc_Z', 'Roll', 'Pitch']:
                col_names.append(f"sensor{sensor_id}_{axis}_t{t}")
    output_df = pd.DataFrame(processed_samples, columns=col_names)
    return output_df


def normalize_dataset(df):
    """
    Normalize the dataset using z-score normalization.
    For each IMU (sensor), compute a single mean and std across all features and timesteps,
    and use them to normalize the corresponding columns.
    """
    df_norm = df.copy()
    sensor_means = {}
    sensor_stds = {}

    for sensor_id in [1, 2, 3, 4]:
        sensor_cols = [col for col in df.columns if col.startswith(f"sensor{sensor_id}_")]
        mean_val = df[sensor_cols].values.mean()
        std_val = df[sensor_cols].values.std()
        sensor_means[sensor_id] = mean_val
        sensor_stds[sensor_id] = std_val
        df_norm[sensor_cols] = (df[sensor_cols] - mean_val) / std_val

    return df_norm, sensor_means, sensor_stds


def process_multiple_csv(input_files, output_files):
    all_dfs = []
    sample_counts = []
    for file in input_files:
        processed_df = process_single_csv(file)
        sample_counts.append(len(processed_df))
        all_dfs.append(processed_df)

    # Combine all samples
    combined_df = pd.concat(all_dfs, ignore_index=True).dropna()

    # Normalize across IMUs
    normalized_combined_df, sensor_means, sensor_stds = normalize_dataset(combined_df)

    # Split back into separate files and save
    start = 0
    for i, count in enumerate(sample_counts):
        subset_df = normalized_combined_df.iloc[start:start + count]
        subset_df.to_csv(output_files[i], index=False)
        print(f"Saved processed file: {output_files[i]}")
        start += count

    return sensor_means, sensor_stds


if __name__ == '__main__':
    # Replace with your actual input file paths
    input_files = [
        'dataset/100_backward.csv',
        'dataset/100_forward.csv',
        'dataset/100_attack.csv',
        'dataset/100_stop.csv'
    ]

    # Output processed file paths
    output_files = [
        'dataset/100_backward_processed.csv',
        'dataset/100_forward_processed.csv',
        'dataset/100_attack_processed.csv',
        'dataset/100_stop_processed.csv'
    ]

    sensor_means, sensor_stds = process_multiple_csv(input_files, output_files)
    np.save('sensor_means.npy', sensor_means)
    np.save('sensor_stds.npy', sensor_stds)
