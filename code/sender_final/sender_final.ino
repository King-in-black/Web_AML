#include "AK09918.h"
#include "ICM20600.h"
#include <Wire.h>
#include <math.h>
#include <WiFi.h>
#include <esp_now.h>
#include "esp_timer.h"
#include "esp_eap_client.h"
#include "esp_netif.h"
#include "freertos/queue.h>
#include <sys/time.h>

#define WIFI_SSID "eduroam"
#define WIFI_USER "zz4321@ic.ac.uk"
#define WIFI_PASS "*********"
#define NTP_SERVER "pool.ntp.org"
#define GMT_OFFSET_SEC 0
#define DAYLIGHT_OFFSET_SEC 0

uint8_t receiverMAC[] = {0xB0, 0xB2, 0x1C, 0xAB, 0x19, 0x9C};

typedef struct {
    uint64_t timestamp;
    double lin_acc_x;
    double lin_acc_y;
    double lin_acc_z;
    double roll;
    double pitch;
} IMUData;

IMUData imuData;
ICM20600 icm20600(true);
QueueHandle_t imuQueue;
uint64_t baseTime = 0;
esp_now_peer_info_t peerInfo;

volatile bool ackReceived = false;
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
    ackReceived = (status == ESP_NOW_SEND_SUCCESS);
    if (status == ESP_NOW_SEND_SUCCESS) {
        Serial.println("(ACK received)");
    } else {
        Serial.println("(No ACK)");
    }
}

uint64_t getUTCMicroseconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + (uint64_t)tv.tv_usec;
}

void collectIMUData() {
    imuData.timestamp = getUTCMicroseconds();

    int16_t acc_x = icm20600.getAccelerationX();
    int16_t acc_y = icm20600.getAccelerationY();
    int16_t acc_z = icm20600.getAccelerationZ();

    if (acc_x == 0 && acc_y == 0 && acc_z == 0) {
        Serial.println("no data");
        return;
    }

    double roll = atan2(-acc_y, acc_z);
    double pitch = atan2(-acc_x, sqrt(acc_y * acc_y + acc_z * acc_z));

    double gravity_x = -sin(pitch) * 1000;
    double gravity_y = cos(pitch) * sin(roll) * 1000;
    double gravity_z = cos(pitch) * cos(roll) * 1000;

    imuData.lin_acc_x = acc_x - gravity_x;
    imuData.lin_acc_y = acc_y + gravity_y;
    imuData.lin_acc_z = acc_z - gravity_z;
    imuData.roll = roll;
    imuData.pitch = pitch;

    Serial.printf("IMU Data | Time: %llu µs\n", imuData.timestamp);
    Serial.printf("Raw Acceleration (mg) | X=%d, Y=%d, Z=%d\n", acc_x, acc_y, acc_z);
    Serial.printf("Calculated Angles (rad) | Roll=%.6f, Pitch=%.6f\n", imuData.roll, imuData.pitch);
}

void sendIMUData() {
    ackReceived = false;
    esp_now_send(receiverMAC, (uint8_t *)&imuData, sizeof(IMUData));
}

void imuTimerCallback(void* arg) {
    uint8_t signal = 1;
    xQueueSend(imuQueue, &signal, 0);
}

void connectToEduroam() {
    WiFi.disconnect(true);
    WiFi.mode(WIFI_STA);
    esp_netif_init();

    esp_eap_client_set_identity((uint8_t *)WIFI_USER, strlen(WIFI_USER));
    esp_eap_client_set_username((uint8_t *)WIFI_USER, strlen(WIFI_USER));
    esp_eap_client_set_password((uint8_t *)WIFI_PASS, strlen(WIFI_PASS));

    esp_wifi_sta_enterprise_enable();
    WiFi.begin(WIFI_SSID);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    configTime(GMT_OFFSET_SEC, DAYLIGHT_OFFSET_SEC, NTP_SERVER);
    delay(2000);
    baseTime = getUTCMicroseconds();
    Serial.printf("UTC base time set: %llu µs\n", baseTime);
}

void setup() {
    Serial.begin(115200);
    Wire.begin();
    connectToEduroam();

    imuQueue = xQueueCreate(10, sizeof(uint8_t));
    if (imuQueue == NULL) {
        Serial.println("failed create task");
        return;
    }

    if (esp_now_init() != ESP_OK) {
        Serial.println("failed initialization");
        return;
    }
    esp_now_register_send_cb(OnDataSent);

    memcpy(peerInfo.peer_addr, receiverMAC, 6);
    peerInfo.channel = 0;
    peerInfo.encrypt = false;
    esp_now_add_peer(&peerInfo);

    icm20600.initialize();

    esp_timer_handle_t imuTimer;
    const esp_timer_create_args_t imu_timer_args = {
        .callback = &imuTimerCallback,
        .arg = NULL,
        .dispatch_method = ESP_TIMER_TASK,
        .name = "imu_timer"
    };
    esp_timer_create(&imu_timer_args, &imuTimer);
    esp_timer_start_periodic(imuTimer, 100000);

    Serial.println("initialization success");
}

void loop() {
    uint8_t signal;
    if (xQueueReceive(imuQueue, &signal, portMAX_DELAY)) {
        uint64_t now = getUTCMicroseconds();

        uint64_t nextSlot = ((now / 100000) + 1) * 100000;
        while (getUTCMicroseconds() < nextSlot) {
            ;
        }

        collectIMUData();
        sendIMUData();
    }
}

