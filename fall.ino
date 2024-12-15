#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

const int FALL_THRESHOLD = 30000;  

void setup() {
  Serial.begin(9600);
  Wire.begin();
  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed!");
    while (1);
  }
  Serial.println("MPU6050 connected!");
}

void loop() {
  int16_t ax, ay, az;
  int16_t gx, gy, gz;

  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  long magnitude = (long)ax * ax + (long)ay * ay + (long)az * az;

  Serial.print("Accel: ");
  Serial.print(ax); Serial.print(" ");
  Serial.print(ay); Serial.print(" ");
  Serial.print(az); Serial.print(" | Magnitude: ");
  Serial.println(magnitude);

  if (magnitude > FALL_THRESHOLD) {
    Serial.println("Fall detected!");
    delay(1000); 
  }

  delay(200); 
}
