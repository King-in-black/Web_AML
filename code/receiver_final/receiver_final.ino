#include <WiFi.h>
#include <esp_now.h>
#include "esp_timer.h"
#include <sys/time.h>
#include <ctime>
#include <cstdio>

#define MAX_DEVICES 10
#define BUFFER_SIZE 20  // Each device stores 20 data entries to prevent packet loss

// IMU Data Structure (Yaw removed)
typedef struct __attribute__((packed)) {
    uint64_t timestamp;  // UTC timestamp in microseconds
    double lin_acc_x;
    double lin_acc_y;
    double lin_acc_z;
    double roll;
    double pitch;
} IMUData;

// Device Data Buffer
struct DeviceBuffer {
    uint8_t mac[6];
    IMUData buffer[BUFFER_SIZE];
    int head;
    int tail;
    bool isFull;
};

DeviceBuffer devices[MAX_DEVICES];
int deviceCount = 0;
esp_timer_handle_t bufferReadTimer;

// Convert UTC Timestamp to YYYY:MM:DD HH:MM:SS:ms
void formatTimestamp(uint64_t timestamp, char *buffer) {
    time_t seconds = timestamp / 1000000;
    int milliseconds = (timestamp % 1000000) / 1000;
    struct tm timeinfo;
    gmtime_r(&seconds, &timeinfo);
    snprintf(buffer, 30, "%04d:%02d:%02d %02d:%02d:%02d:%03d",
             timeinfo.tm_year + 1900, timeinfo.tm_mon + 1, timeinfo.tm_mday,
             timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec, milliseconds);
}

// Find or Register a Device
int getDeviceID(const uint8_t *mac) {
    for (int i = 0; i < deviceCount; i++) {
        if (memcmp(mac, devices[i].mac, 6) == 0) {
            return i;
        }
    }

    if (deviceCount < MAX_DEVICES) {
        memcpy(devices[deviceCount].mac, mac, 6);
        devices[deviceCount].head = 0;
        devices[deviceCount].tail = 0;
        devices[deviceCount].isFull = false;
        return deviceCount++;
    } else {
        Serial.println("Maximum device limit reached. Cannot register a new device.");
        return -1;
    }
}

// Store Data in Device Buffer
void storeData(int deviceID, IMUData *imuData) {
    DeviceBuffer *dev = &devices[deviceID];
    dev->buffer[dev->head] = *imuData;
    dev->head = (dev->head + 1) % BUFFER_SIZE;
    if (dev->isFull) {
        dev->tail = (dev->tail + 1) % BUFFER_SIZE;  // Overwrite old data
    }
    dev->isFull = (dev->head == dev->tail);
}

// ESP-NOW Data Receive Callback
void OnDataRecv(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len) {
    if (len != sizeof(IMUData)) {
        Serial.printf("Data size mismatch: Expected %d bytes, received %d bytes\n", sizeof(IMUData), len);
        return;
    }

    IMUData imuData;
    memcpy(&imuData, incomingData, sizeof(imuData));

    int deviceID = getDeviceID(info->src_addr);
    if (deviceID == -1) return;

    storeData(deviceID, &imuData);
}

// Read and Print Buffered Data (Periodic Execution)
void readBufferedData(void* arg) {
    for (int i = 0; i < deviceCount; i++) {
        DeviceBuffer *dev = &devices[i];
        while (dev->head != dev->tail || dev->isFull) {
            IMUData *data = &dev->buffer[dev->tail];
            char formattedTime[30];
            formatTimestamp(data->timestamp, formattedTime);

            Serial.printf("\nDevice %d (MAC: %02X:%02X:%02X:%02X:%02X:%02X)\n",
                          i, dev->mac[0], dev->mac[1], dev->mac[2], dev->mac[3], dev->mac[4], dev->mac[5]);
            Serial.printf("Timestamp: %s\n", formattedTime);
            Serial.printf("Acceleration (mg): X=%.2f, Y=%.2f, Z=%.2f\n",
                          data->lin_acc_x, data->lin_acc_y, data->lin_acc_z);
            Serial.printf("Orientation (°): Roll=%.2f, Pitch=%.2f\n",
                          data->roll, data->pitch);  // Yaw removed
            dev->tail = (dev->tail + 1) % BUFFER_SIZE;
            dev->isFull = false;
        }
    }
}

// Initialize ESP-NOW
void setup() {
    Serial.begin(115200);
    WiFi.mode(WIFI_STA);

    if (esp_now_init() != ESP_OK) {
        Serial.println("Failed to initialize ESP-NOW");
        return;
    }

    esp_now_register_recv_cb(OnDataRecv);
    Serial.println("Receiver ready. Waiting for IMU data...");

    // Create Timer to Read Buffer Every 10ms
    const esp_timer_create_args_t buffer_timer_args = {
        .callback = &readBufferedData,
        .arg = NULL,
        .dispatch_method = ESP_TIMER_TASK,
        .name = "buffer_read_timer"
    };
    esp_timer_create(&buffer_timer_args, &bufferReadTimer);
    esp_timer_start_periodic(bufferReadTimer, 1000); // Read every 10ms
}

void loop() {
    delay(100);  // Yield CPU to reduce power consumption
}
