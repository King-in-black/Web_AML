import serial
import csv
import time
import threading
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Serial port configuration
SERIAL_PORT = "COM3"
BAUD_RATE = 115200

# Exit flag
exit_flag = False

# Buffer for recent 3 seconds of data (assuming 20Hz sampling rate)
data_buffer = deque(maxlen=240)

# Full data list (for CSV saving)
all_data_list = []  # Each item: [MAC, Timestamp, Lin_Acc_X, Lin_Acc_Y, Lin_Acc_Z, Roll, Pitch]

# Buffers for matplotlib real-time plotting
time_buffer = deque(maxlen=60)
lin_acc_x_buffer = deque(maxlen=60)
lin_acc_y_buffer = deque(maxlen=60)
lin_acc_z_buffer = deque(maxlen=60)
roll_buffer = deque(maxlen=60)
pitch_buffer = deque(maxlen=60)

# MAC address to sensor ID mapping (consistent with offline processing)
sensor_map = {
    'B0:B2:1C:8F:6B:C4': 1,
    'CC:7B:5C:AF:BD:F4': 2,
    '08:B6:1F:75:ED:80': 3,
    '08:D1:F9:DC:8F:80': 4
}

# Thread: listen for user command (press 'q' to quit)
def listen_for_commands():
    global exit_flag
    while True:
        user_input = input("Type 'q' to quit: ").strip().lower()
        if user_input == "q":
            exit_flag = True
            break

exit_thread = threading.Thread(target=listen_for_commands, daemon=True)
exit_thread.start()

# Open serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

# Create CSV file for storing all data
csv_filename = "esp_now_imu_data.csv"
header = ["MAC", "Timestamp", "Lin_Acc_X", "Lin_Acc_Y", "Lin_Acc_Z", "Roll", "Pitch"]

try:
    with open(csv_filename, "x", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
except FileExistsError:
    pass

print("Waiting for ESP-NOW data...")

# Thread: read serial data
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

        if "📡 Device" in line:
            mac_address = line.split()[-1].strip(")")
            continue

        if "🕒 Timestamp" in line:
            timestamp = line.split(": ", 1)[1]
            continue

        if "📊 Acceleration" in line:
            acc_data = line.split(": ")[1].replace("X=", "").replace("Y=", "").replace("Z=", "").split(", ")
            lin_acc_x = float(acc_data[0])
            lin_acc_y = float(acc_data[1])
            lin_acc_z = float(acc_data[2])
            continue

        if "🔄 Orientation" in line:
            angle_data = line.split(": ")[1].replace("Roll=", "").replace("Pitch=", "").split(", ")
            roll = float(angle_data[0])
            pitch = float(angle_data[1])
            current_time = time.time()
            sample = [mac_address, timestamp, lin_acc_x, lin_acc_y, lin_acc_z, roll, pitch]
            data_buffer.append((current_time, sample))
            all_data_list.append(sample)
            time_buffer.append(current_time)
            lin_acc_x_buffer.append(lin_acc_x)
            lin_acc_y_buffer.append(lin_acc_y)
            lin_acc_z_buffer.append(lin_acc_z)
            roll_buffer.append(roll)
            pitch_buffer.append(pitch)
            print(f"MAC: {mac_address} | Time: {timestamp} | X={lin_acc_x}mg, Y={lin_acc_y}mg, Z={lin_acc_z}mg | Roll={roll}° Pitch={pitch}°")

serial_thread = threading.Thread(target=read_serial_data, daemon=True)
serial_thread.start()

# Thread: periodically save collected data
def save_all_data():
    while not exit_flag:
        if all_data_list:
            with open(csv_filename, "a", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                for row in all_data_list:
                    csv_writer.writerow(row)
            all_data_list.clear()
        time.sleep(1)

save_thread = threading.Thread(target=save_all_data, daemon=True)
save_thread.start()

# Real-time plot update function
def update_plot(frame):
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.plot(list(time_buffer), list(lin_acc_x_buffer), label="Lin_Acc_X", color="r")
    plt.plot(list(time_buffer), list(lin_acc_y_buffer), label="Lin_Acc_Y", color="g")
    plt.plot(list(time_buffer), list(lin_acc_z_buffer), label="Lin_Acc_Z", color="b")
    plt.legend(loc="upper right")
    plt.ylabel("Linear Acc (mg)")
    plt.title("IMU Linear Acceleration")

    plt.subplot(2, 1, 2)
    plt.plot(list(time_buffer), list(roll_buffer), label="Roll", color="c")
    plt.plot(list(time_buffer), list(pitch_buffer), label="Pitch", color="m")
    plt.legend(loc="upper right")
    plt.ylabel("Angle (°)")
    plt.xlabel("Time")
    plt.title("IMU Orientation (Roll & Pitch)")

    plt.tight_layout()

fig = plt.figure()
ani = animation.FuncAnimation(fig, update_plot, interval=200)
plt.show()

# Final saving on exit
if data_buffer:
    with open(csv_filename, "a", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([])
        csv_writer.writerows([row[1] for row in data_buffer])
    print(f"\n📁 Saved {len(data_buffer)} remaining records to {csv_filename}.")

ser.close()
print("\nProgram exited. Data saved to", csv_filename)
