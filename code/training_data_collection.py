import serial
import csv
import time
import threading
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Serial configuration
SERIAL_PORT = "COM3"
BAUD_RATE = 115200

# Exit & Save Flags
exit_flag = False
save_flag = False
save_time = None

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


# Listen for user input
def listen_for_commands():
    global exit_flag, save_flag, save_time
    while True:
        user_input = input("Enter 'q' to exit, press 'Enter' to save data: ").strip().lower()
        if user_input == "q":
            exit_flag = True
            break
        elif user_input == "":
            save_flag = True
            save_time = time.time()

# Start user input listener thread
exit_thread = threading.Thread(target=listen_for_commands, daemon=True)
exit_thread.start()

# Open serial port
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

# CSV file setup
csv_filename = "esp_now_imu_data.csv"
header = ["MAC", "Timestamp", "Lin_Acc_X", "Lin_Acc_Y", "Lin_Acc_Z", "Roll", "Pitch"]

# Create CSV file if it does not exist
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
    global save_flag

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

            # Store in data buffer
            current_time = time.time()
            data_buffer.append((current_time, [mac_address, timestamp, lin_acc_x, lin_acc_y, lin_acc_z, roll, pitch]))

            if mac_address in sensor_map:
                sensor_id = sensor_map[mac_address]
                sensor_buffers[sensor_id]['time'].append(current_time)
                sensor_buffers[sensor_id]['lin_acc_x'].append(lin_acc_x)
                sensor_buffers[sensor_id]['lin_acc_y'].append(lin_acc_y)
                sensor_buffers[sensor_id]['lin_acc_z'].append(lin_acc_z)
                sensor_buffers[sensor_id]['roll'].append(roll)
                sensor_buffers[sensor_id]['pitch'].append(pitch)
            print(f"Device MAC: {mac_address} | Time: {timestamp} | X={lin_acc_x}mg, Y={lin_acc_y}mg, Z={lin_acc_z}mg | Roll={roll}° Pitch={pitch}°")

# Start serial data reading thread
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

# Save last 1 second before and after the trigger
def save_data():
    global save_flag, save_time
    while not exit_flag:
        if save_flag:
            save_flag = False  
            start_time = save_time - 1
            end_time = save_time + 1

            print(f" Waiting for future 1-second data to enter buffer...")
            time.sleep(1.5)  # Ensure future data enters the buffer

            filtered_data = [row[1] for row in data_buffer if start_time <= row[0] <= end_time]

            if not filtered_data:
                print(" No data found in the time window, retrying after 0.5s...")
                time.sleep(0.5)  # Wait another 0.5s to ensure data entry
                filtered_data = [row[1] for row in data_buffer if start_time <= row[0] <= end_time]

            if filtered_data:
                with open(csv_filename, "a", newline="", encoding="utf-8") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([])  
                    csv_writer.writerows(filtered_data)

                print(f" Saved {len(filtered_data)} records (Time Window: {start_time:.2f}s ~ {end_time:.2f}s) to {csv_filename}!")
            else:
                print(" No data found in the time window, not saved!")

        time.sleep(0.5)

# Start data saving thread
save_thread = threading.Thread(target=save_data, daemon=True)
save_thread.start()

# Display Matplotlib real-time visualization
plt.show()

# Save remaining data before exiting
if data_buffer:
    with open(csv_filename, "a", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([])
        csv_writer.writerows([row[1] for row in data_buffer])
    print(f"\nSaved {len(data_buffer)} records to {csv_filename} before exiting.")

# Close serial port
ser.close()
print("\nProgram exited, data saved to", csv_filename)
