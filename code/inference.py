import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from models.LSTM import BiLSTMModel  # Make sure this module is importable

# Mapping from MAC address to sensor ID
sensor_map = {
    'B0:B2:1C:8F:6B:C4': 1,
    'CC:7B:5C:AF:BD:F4': 2,
    '08:B6:1F:75:ED:80': 3,
    '08:D1:F9:DC:8F:80': 4
}


def process_test_csv(csv_path):
    """
    Read a test CSV file and generate a numpy array of shape S×4×5,
    where S is the number of time steps, 4 is the number of sensors
    (ordered by sensor_map), and 5 features are:
    [Lin_Acc_X, Lin_Acc_Y, Lin_Acc_Z, Roll, Pitch].

    Missing sensor data at any time step will be filled using linear
    interpolation and forward/backward filling.
    """
    df = pd.read_csv(csv_path)
    df['Timestamp_dt'] = pd.to_datetime(df['Timestamp'], format='%Y:%m:%d %H:%M:%S:%f')
    df = df.sort_values('Timestamp_dt')

    timestamps = sorted(df['Timestamp_dt'].unique())
    features = ['Lin_Acc_X', 'Lin_Acc_Y', 'Lin_Acc_Z', 'Roll', 'Pitch']

    sensor_data = {}
    for sensor_id in [1, 2, 3, 4]:
        sensor_data[sensor_id] = pd.DataFrame(index=timestamps, columns=features)

    groups = df.groupby('Timestamp_dt')
    for t, group in groups:
        for _, row in group.iterrows():
            mac = row['MAC']
            if mac in sensor_map:
                sensor_id = sensor_map[mac]
                sensor_data[sensor_id].loc[t] = row[features].values

    for sensor_id in sensor_data:
        sensor_data[sensor_id] = sensor_data[sensor_id].apply(pd.to_numeric, errors='coerce')
        sensor_data[sensor_id] = sensor_data[sensor_id].interpolate(method='linear', limit_direction='both')
        sensor_data[sensor_id] = sensor_data[sensor_id].fillna(method='ffill').fillna(method='bfill')

    S = len(timestamps)
    data_array = np.zeros((S, 4, 5), dtype=np.float32)
    for i, sensor_id in enumerate([1, 2, 3, 4]):
        data_array[:, i, :] = sensor_data[sensor_id].values

    return data_array


def sliding_window(data_array, window_size=20):
    """
    Apply a sliding window to the input data_array (S×4×5),
    and return a tensor of shape (N, window_size, 4, 5),
    where N = S - window_size + 1.
    """
    S = data_array.shape[0]
    windows = []
    for start in range(0, S - window_size + 1):
        window = data_array[start:start + window_size, :, :]
        windows.append(window)
    return np.array(windows)


def load_np():
    """
    Load the saved sensor_means and sensor_stds (as .npy dicts).
    """
    sensor_means = np.load('sensor_means.npy', allow_pickle=True).item()
    sensor_stds = np.load('sensor_stds.npy', allow_pickle=True).item()
    return sensor_means, sensor_stds


def test_model(test_csv, model, device, window_size=20, output_txt="predictions.txt"):
    # Step 1: Process CSV to get data in shape (S, 4, 5)
    data_array = process_test_csv(test_csv)

    # Step 2: Load means and stds from training
    sensor_means, sensor_stds = load_np()

    # Step 3: Normalize per sensor
    for idx, sensor_id in enumerate([1, 2, 3, 4]):
        mean_val = sensor_means[sensor_id]
        std_val = sensor_stds[sensor_id]
        data_array[:, idx, :] = (data_array[:, idx, :] - mean_val) / std_val

    # Step 4: Generate sliding window input
    windows = sliding_window(data_array, window_size=window_size)

    # Step 5: Convert to torch tensor and move to device
    windows_tensor = torch.tensor(windows, dtype=torch.float32).to(device)

    # Step 6: Inference
    model.eval()
    with torch.no_grad():
        outputs = model(windows_tensor)
        preds = outputs.argmax(dim=1)

    # Step 7: Move predictions to CPU
    predictions = preds.cpu().numpy()

    # Step 8: Map label index to class name
    label_to_name = {0: "forward", 1: "backward", 2: "attack", 3: "stop"}
    pred_names = [label_to_name.get(int(pred), "unknown") for pred in predictions]

    # Step 9: Write to file
    with open(output_txt, "w") as f:
        for p in pred_names:
            f.write(str(p) + "\n")
    print(f"Predictions saved to {output_txt}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BiLSTMModel(sensor_embed_dim=60, hidden_size=64, num_layers=2, num_classes=4).to(device)

    sd = torch.load('bilstm_model.pth', weights_only=True)
    model.load_state_dict(sd)
    print("Model loaded successfully.")

    test_csv = "1234.csv"

    test_model(test_csv, model, device, window_size=20, output_txt="predictions_1234.txt")


if __name__ == "__main__":
    main()
