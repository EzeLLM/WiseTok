#include <WiFi.h>
#include <PubSubClient.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>
#include <DHT.h>

#define DHTPIN 4
#define DHTTYPE DHT22
#define BUTTON_PIN 5
#define LED_PIN 2
#define SENSOR_PIN 34

const char* ssid = "WiFi_Network";
const char* password = "password";
const char* mqtt_server = "broker.mqtt.com";
const int mqtt_port = 1883;
const char* mqtt_user = "user";
const char* mqtt_pass = "pass";

DHT dht(DHTPIN, DHTTYPE);
WiFiClient espClient;
PubSubClient client(espClient);

volatile bool button_pressed = false;
SemaphoreHandle_t data_semaphore;
TaskHandle_t sensor_task_handle;

void IRAM_ATTR buttonISR() {
    button_pressed = true;
    digitalWrite(LED_PIN, HIGH);
}

void sensorTask(void* pvParameter) {
    while (1) {
        float humidity = dht.readHumidity();
        float temperature = dht.readTemperature();

        if (isnan(humidity) || isnan(temperature)) {
            Serial.println("Failed to read from DHT sensor");
            vTaskDelay(pdMS_TO_TICKS(2000));
            continue;
        }

        if (xSemaphoreTake(data_semaphore, portMAX_DELAY) == pdTRUE) {
            char payload[128];
            snprintf(payload, sizeof(payload),
                    "{\"temp\": %.2f, \"humidity\": %.2f}",
                    temperature, humidity);

            if (client.connected()) {
                client.publish("sensor/data", payload);
                Serial.print("Published: ");
                Serial.println(payload);
            }

            xSemaphoreGive(data_semaphore);
        }

        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}

void setup() {
    Serial.begin(115200);
    delay(100);

    pinMode(BUTTON_PIN, INPUT_PULLUP);
    pinMode(LED_PIN, OUTPUT);
    pinMode(SENSOR_PIN, INPUT);

    digitalWrite(LED_PIN, LOW);

    attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), buttonISR, FALLING);

    dht.begin();

    data_semaphore = xSemaphoreCreateMutex();

    Serial.println("\nStarting WiFi connection...");
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWiFi connected");
        Serial.print("IP address: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("\nFailed to connect to WiFi");
    }

    client.setServer(mqtt_server, mqtt_port);
    client.setCallback(mqttCallback);

    xTaskCreatePinnedToCore(
        sensorTask,
        "SensorTask",
        4096,
        NULL,
        1,
        &sensor_task_handle,
        1
    );

    Serial.println("Setup complete");
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
    Serial.print("Message arrived on topic: ");
    Serial.println(topic);
    Serial.print("Payload: ");
    for (unsigned int i = 0; i < length; i++) {
        Serial.print((char)payload[i]);
    }
    Serial.println();
}

void reconnectMQTT() {
    while (!client.connected()) {
        Serial.print("Attempting MQTT connection...");
        if (client.connect("ESP32Client", mqtt_user, mqtt_pass)) {
            Serial.println("connected");
            client.subscribe("control/led");
        } else {
            Serial.print("failed, rc=");
            Serial.print(client.state());
            Serial.println(" try again in 5 seconds");
            delay(5000);
        }
    }
}

void loop() {
    if (!client.connected()) {
        reconnectMQTT();
    }
    client.loop();

    if (button_pressed) {
        Serial.println("Button pressed!");
        button_pressed = false;

        int adc_value = analogRead(SENSOR_PIN);
        Serial.print("ADC value: ");
        Serial.println(adc_value);

        delay(200);
        digitalWrite(LED_PIN, LOW);
    }

    int raw_analog = analogRead(SENSOR_PIN);
    float voltage = (raw_analog / 4095.0) * 3.3;

    if (xSemaphoreTake(data_semaphore, (TickType_t)10) == pdTRUE) {
        char status[100];
        snprintf(status, sizeof(status),
                "Sensor: %d, Voltage: %.2fV",
                raw_analog, voltage);
        xSemaphoreGive(data_semaphore);
    }

    delay(100);
}
