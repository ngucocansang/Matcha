#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <math.h>

Adafruit_MPU6050 mpu;

float pitch = 0;
unsigned long last_time;
unsigned long start_time;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10); // Chờ serial mở

  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) delay(10);
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  last_time = micros();
  start_time = millis();  // Lưu thời điểm bắt đầu 15 giây
}

void loop() {
  // dừng sau 15 giây
  if (millis() - start_time > 15000) {
    Serial.println("END of 15s");
    while (1) {}
  }

  // ----- Đọc IMU -----
  unsigned long now = micros();
  float dt = (now - last_time) / 1e6; // thời gian giữa 2 lần đo
  last_time = now;

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // ----- Tính pitch (deg) -----
  float accel_pitch = atan2(a.acceleration.x, sqrt(a.acceleration.y * a.acceleration.y + a.acceleration.z * a.acceleration.z)) * 180.0 / M_PI;
  
  // Gyro_y deg/s
  float gyro_y = g.gyro.y;

  // ----- Low-pass + complementary filter để pitch mượt -----
  pitch = 0.98f * (pitch + gyro_y * dt) + 0.02f * accel_pitch;

  // ----- Gửi pitch + pitch_rate sang Python -----
  Serial.print(pitch);
  Serial.print(",");
  Serial.println(gyro_y);

  // ----- Nhận action từ Python -----
  if (Serial.available()) {
    float action = Serial.readStringUntil('\n').toFloat();
    // TODO: map action -> motor driver
    // ví dụ: setMotor(action * 255);
  }

  delay(10); // ~100Hz
}
