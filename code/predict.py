import serial
import csv
import time
import threading
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import os
import numpy as np
import pandas as pd
import torch.nn as nn

SERIAL_PORT = "COM3"
BAUD_RATE = 115200

exit_flag = False
# Data buffer for last 3 seconds (assuming 20Hz sampling rate)
data_buffer = deque(maxlen=240)

# List to store all recorded data
all_data_list = []

# Buffers for real-time visualization for each sensor
sensor_buffers = {}
for sensor_id in [1, 2, 3, 4]:
    sensor_buffers[sensor_id] = {
        'time': deque(maxlen=60),
        'lin_acc_x': deque(maxlen=60),
        'lin_acc_y': deque(maxlen=60),
        'lin_acc_z': deque(maxlen=60),
        'roll': deque(maxlen=60),
        'pitch': deque(maxlen=60)
    }

# Sensor MAC to ID mapping
sensor_map = {
    'B0:B2:1C:8F:6B:C4': 1,
    'CC:7B:5C:AF:BD:F4': 2,
    '08:B6:1F:75:ED:80': 3,
    '08:D1:F9:DC:8F:80': 4
}

try:
    mean_array = np.load("mean.npy") 
    std_array = np.load("std.npy")   
    mean_tensor = torch.tensor(mean_array, dtype=torch.float)
    std_tensor = torch.tensor(std_array, dtype=torch.float)
    print("Normalization parameters loaded.")
except Exception as e:
    print("Error loading normalization parameters:", e)
    mean_tensor = None
    std_tensor = None
    
# Listen for user input ("q" to exit)
def listen_for_commands():
    global exit_flag
    while True:
        user_input = input("Enter 'q' to exit: ").strip().lower()
        if user_input == "q":
            exit_flag = True
            break

exit_thread = threading.Thread(target=listen_for_commands, daemon=True)
exit_thread.start()

# Open serial port
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

# CSV file setup (optional)
csv_filename = "esp_now_imu_data.csv"
header = ["MAC", "Timestamp", "Lin_Acc_X", "Lin_Acc_Y", "Lin_Acc_Z", "Roll", "Pitch"]
try:
    with open(csv_filename, "x", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
except FileExistsError:
    pass

print("Waiting for ESP-NOW data...")

# Serial data reading thread
def read_serial_data():
    mac_address = "Unknown"
    timestamp = "Unknown"
    lin_acc_x = 0.0
    lin_acc_y = 0.0
    lin_acc_z = 0.0
    roll = 0.0
    pitch = 0.0
    while not exit_flag:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue

        # Parse data
        if "Device" in line:
            mac_address = line.split()[-1].strip(")")
            continue

        if "Timestamp" in line:
            timestamp = line.split(": ", 1)[1]
            continue

        if "Acceleration" in line:
            acc_data = line.split(": ")[1].replace("X=", "").replace("Y=", "").replace("Z=", "").split(", ")
            lin_acc_x = float(acc_data[0])
            lin_acc_y = float(acc_data[1])
            lin_acc_z = float(acc_data[2])
            continue

        if "Orientation" in line:
            angle_data = line.split(": ")[1].replace("Roll=", "").replace("Pitch=", "").split(", ")
            roll = float(angle_data[0])
            pitch = float(angle_data[1])

            # Store in global buffer
            current_time = time.time()
            data_buffer.append((current_time, [mac_address, timestamp, lin_acc_x, lin_acc_y, lin_acc_z, roll, pitch]))

            # Update Matplotlib buffer for real-time visualization
            if mac_address in sensor_map:
                sensor_id = sensor_map[mac_address]
                sensor_buffers[sensor_id]['time'].append(current_time)
                sensor_buffers[sensor_id]['lin_acc_x'].append(lin_acc_x)
                sensor_buffers[sensor_id]['lin_acc_y'].append(lin_acc_y)
                sensor_buffers[sensor_id]['lin_acc_z'].append(lin_acc_z)
                sensor_buffers[sensor_id]['roll'].append(roll)
                sensor_buffers[sensor_id]['pitch'].append(pitch)

serial_thread = threading.Thread(target=read_serial_data, daemon=True)
serial_thread.start()

# Matplotlib setup for live plotting
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 10))
fig.suptitle("4 IMU Live Data Collection", fontsize=16)

def update(frame):

    for idx, sensor_id in enumerate([1, 2, 3, 4]):

        ax_acc = axes[idx, 0]
        ax_acc.cla()  
        times = list(sensor_buffers[sensor_id]['time'])
        acc_x = list(sensor_buffers[sensor_id]['lin_acc_x'])
        acc_y = list(sensor_buffers[sensor_id]['lin_acc_y'])
        acc_z = list(sensor_buffers[sensor_id]['lin_acc_z'])
        ax_acc.plot(times, acc_x, label="Lin_Acc_X", color="r")
        ax_acc.plot(times, acc_y, label="Lin_Acc_Y", color="g")
        ax_acc.plot(times, acc_z, label="Lin_Acc_Z", color="b")
        ax_acc.legend(loc="upper right")
        ax_acc.set_ylabel("mg")
        ax_acc.set_title(f"IMU {sensor_id} ACC")

        ax_ori = axes[idx, 1]
        ax_ori.cla()
        roll = list(sensor_buffers[sensor_id]['roll'])
        pitch = list(sensor_buffers[sensor_id]['pitch'])
        ax_ori.plot(times, roll, label="Roll", color="c")
        ax_ori.plot(times, pitch, label="Pitch", color="m")
        ax_ori.legend(loc="upper right")
        ax_ori.set_ylabel("°")
        ax_ori.set_title(f"IMU {sensor_id} Roll & Pitch")

        if idx == 3:
            ax_ori.set_xlabel("Time")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

ani = animation.FuncAnimation(fig, update, interval=200, cache_frame_data=False)

plt.show()

class BiLSTMModel(nn.Module):
    def __init__(self, sensor_embed_dim=60, hidden_size=64, num_layers=2, num_classes=4):

        super(BiLSTMModel, self).__init__()
        self.sensor_embed_dim = sensor_embed_dim
        
        self.sensor_embed = nn.Linear(5, sensor_embed_dim)
        self.sensor_attn = nn.MultiheadAttention(embed_dim=sensor_embed_dim, num_heads=3, batch_first=True)
        
        lstm_input_dim = 4 * sensor_embed_dim
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):

        bs, seq_len, n, d_prime = x.shape  
        x = x.reshape(bs * seq_len * n, d_prime) 
        x = self.sensor_embed(x)               
        x = x.reshape(bs, seq_len, n, self.sensor_embed_dim)
        
        x = x.reshape(bs * seq_len, n, self.sensor_embed_dim)
        attn_output, _ = self.sensor_attn(x, x, x) 
        attn_output = attn_output.reshape(bs, seq_len, n, self.sensor_embed_dim)
        
        attn_output = attn_output.reshape(bs, seq_len, n * self.sensor_embed_dim)
        
        lstm_out, _ = self.lstm(attn_output)
        pooled = torch.mean(lstm_out, dim=1)  
        
        logits = self.fc(pooled)             
        return logits

model = BiLSTMModel(sensor_embed_dim=60, hidden_size=64, num_layers=2, num_classes=4)
model_save_path = "bilstm_model.pth"
if os.path.exists(model_save_path):
    try:
        best_model_state = torch.load(model_save_path, map_location=torch.device('cpu'))
        model.load_state_dict(best_model_state)
        print(f"Loaded model from {model_save_path}")
    except Exception as e:
        print("Failed to load best_model_state, trying model.state_dict()...", e)
        model.load_state_dict(torch.load(model_save_path))
        print(f"Loaded trained model from {model_save_path}")
else:
    print("Model file not found!")
model.eval()


def process_realtime_sample(data_list):
    sensor_data = {1: [], 2: [], 3: [], 4: []}
    for row in data_list:
        mac = row[0]
        if mac in sensor_map:
            sensor_id = sensor_map[mac]
            sensor_data[sensor_id].append({
                'Lin_Acc_X': row[2],
                'Lin_Acc_Y': row[3],
                'Lin_Acc_Z': row[4],
                'Roll': row[5],
                'Pitch': row[6]
            })
    processed_samples = {}
    for sensor_id in [1, 2, 3, 4]:
        df = pd.DataFrame(sensor_data[sensor_id])
        if df.empty:
            df = pd.DataFrame([[0, 0, 0, 0, 0]] * 20, columns=['Lin_Acc_X', 'Lin_Acc_Y', 'Lin_Acc_Z', 'Roll', 'Pitch'])
        else:
            num_rows = len(df)
            if num_rows < 20:
                pad_df = pd.DataFrame([df.iloc[-1].values.tolist()] * (20 - num_rows),
                                      columns=df.columns)
                df = pd.concat([df, pad_df], ignore_index=True)
            elif num_rows > 20:
                df = df.iloc[:20]
        processed_samples[sensor_id] = df
    sample_flat = []
    for sensor_id in [1, 2, 3, 4]:
        sample_flat.extend(processed_samples[sensor_id].values.flatten().tolist())
    return sample_flat  

def predict_sample_from_tensor(sample_tensor):
    input_tensor = sample_tensor.permute(1, 0, 2)
    input_tensor = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        label_map = {0: "forward", 1: "backward", 2: "attack", 3: "stop"}
    return label_map.get(pred_class, "Unknown")


def predict_sample_from_vector(sample_vector):

    sample_tensor = torch.tensor(sample_vector, dtype=torch.float).view(4, 20, 5)
    if mean_tensor is not None and std_tensor is not None:
        sample_tensor = (sample_tensor - mean_tensor.view(4,1,1)) / std_tensor.view(4,1,1)
    return predict_sample_from_tensor(sample_tensor)


def predict_data():
    while not exit_flag:
        current_time = time.time()
        start_time = current_time - 2
        end_time = current_time
        filtered_data = [row[1] for row in data_buffer if start_time <= row[0] <= end_time]
        if filtered_data:
            sample_vector = process_realtime_sample(filtered_data)
            prediction = predict_sample_from_vector(sample_vector)
            print(f"predict result: {prediction}")
        else:
            print("missing data, skipping prediction")
        time.sleep(0.1)

predict_thread = threading.Thread(target=predict_data, daemon=True)
predict_thread.start()


# Show Matplotlib real-time visualization
plt.show()

# Save remaining data before exiting
if data_buffer:
    with open(csv_filename, "a", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([])
        csv_writer.writerows([row[1] for row in data_buffer])
    print(f"\nSaved {len(data_buffer)} records to {csv_filename} before exiting.")

ser.close()
print("\nProgram exited, data saved to", csv_filename)
