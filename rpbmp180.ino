#include <Wire.h>
#include <Adafruit_BMP085.h>
#include "model.h"

// Minimal TensorFlow Lite includes


#include <Chirale_TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

Adafruit_BMP085 bmp;

constexpr int kTensorArenaSize = 30 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

const int seq_length = 15;
const int features = 3;
float history[seq_length][features] = {0};
int history_index = 0;
bool history_full = false;

const float data_min[3] = {1006.86, 28.90, 44.05};
const float data_max[3] = {1007.97, 30.90, 53.17};
float scaler_min[3];
float scaler_scale[3];

// Track sensor status and last good values
bool sensor_connected = true;
unsigned long last_successful_reading = 0;
const unsigned long SENSOR_TIMEOUT = 5000;

// Last known good values for fallback
float last_pressure = 1007.0;
float last_temperature = 30.0;
float last_altitude = 48.0;
bool has_valid_history = false;

// For trend-based prediction
float pressure_trend = 0;
float temperature_trend = 0;
float altitude_trend = 0;

float normalize(int idx, float val) {
  return (val - scaler_min[idx]) / scaler_scale[idx];
}

float denormalize(int idx, float val) {
  return val * scaler_scale[idx] + scaler_min[idx];
}

bool readBMP180(float &pressure, float &temperature, float &altitude) {
  pressure = bmp.readPressure() / 100.0;
  temperature = bmp.readTemperature();
  altitude = bmp.readAltitude(101325);
  
  bool valid = !isnan(pressure) && (pressure > 900) && (pressure < 1100) &&
               !isnan(temperature) && (temperature > -40) && (temperature < 85) &&
               !isnan(altitude) && (altitude > -100) && (altitude < 10000);
  
  if (valid) {
    // Update trends based on changes
    if (has_valid_history) {
      pressure_trend = (pressure - last_pressure) * 0.1 + pressure_trend * 0.9;
      temperature_trend = (temperature - last_temperature) * 0.1 + temperature_trend * 0.9;
      altitude_trend = (altitude - last_altitude) * 0.1 + altitude_trend * 0.9;
    }
    
    last_successful_reading = millis();
    last_pressure = pressure;
    last_temperature = temperature;
    last_altitude = altitude;
    has_valid_history = true;
  }
  
  return valid;
}

void checkSensorStatus() {
  unsigned long current_time = millis();
  
  if (sensor_connected) {
    if (current_time - last_successful_reading > SENSOR_TIMEOUT) {
      sensor_connected = false;
    }
  } else {
    float pressure, temperature, altitude;
    if (readBMP180(pressure, temperature, altitude)) {
      sensor_connected = true;
    }
  }
}

// Improved prediction with trends and natural variation
void simplePrediction(float &pressure, float &temperature, float &altitude) {
  // Base prediction on last values with trends
  pressure = last_pressure + pressure_trend + (random(-5, 5) / 100.0);
  temperature = last_temperature + temperature_trend + (random(-3, 3) / 100.0);
  altitude = last_altitude + altitude_trend + (random(-2, 2) / 100.0);
  
  // Reasonable bounds based on your actual data range
  pressure = constrain(pressure, 1000.0, 1020.0);
  temperature = constrain(temperature, 25.0, 35.0);
  altitude = constrain(altitude, 30.0, 80.0);
}

bool runModelPrediction(float &pressure, float &temperature, float &altitude) {
  if (!history_full) {
    return false;
  }
  
  // Fill input tensor
  for (int i = 0; i < seq_length; i++) {
    for (int j = 0; j < features; j++) {
      input->data.f[i * features + j] = history[(history_index + i) % seq_length][j];
    }
  }

  // Run prediction
  TfLiteStatus invoke_status = interpreter->Invoke();
  
  if (invoke_status == kTfLiteOk) {
    if (!isnan(output->data.f[0]) && !isnan(output->data.f[1]) && !isnan(output->data.f[2])) {
      pressure = denormalize(0, output->data.f[0]);
      temperature = denormalize(1, output->data.f[1]);
      altitude = denormalize(2, output->data.f[2]);
      return true;
    }
  }
  
  return false;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }
  
  Wire.begin(21, 22);
  Wire.setClock(100000);
  
  // Initialize BMP180
  bool sensor_initialized = false;
  for (int i = 0; i < 3; i++) {
    if (bmp.begin()) {
      sensor_initialized = true;
      break;
    }
    delay(1000);
  }
  
  if (!sensor_initialized) {
    sensor_connected = false;
  } else {
    sensor_connected = true;
    last_successful_reading = millis();
  }

  // Initialize scaler parameters
  for (int i = 0; i < features; i++) {
    scaler_min[i] = data_min[i];
    scaler_scale[i] = data_max[i] - data_min[i];
  }

  // Setup TensorFlow Lite model
  const tflite::Model* model = tflite::GetModel(bmp180_model_tflite);
  
  if (model->version() == TFLITE_SCHEMA_VERSION) {
    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddMul();

    static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
      
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() == kTfLiteOk) {
      input = interpreter->input(0);
      output = interpreter->output(0);
    }
  }
}

void loop() {
  checkSensorStatus();
  
  float pressure, temperature, altitude;
  
  if (sensor_connected && readBMP180(pressure, temperature, altitude)) {
    // Store normalized values
    history[history_index][0] = normalize(0, pressure);
    history[history_index][1] = normalize(1, temperature);
    history[history_index][2] = normalize(2, altitude);
    
    history_index = (history_index + 1) % seq_length;
    if (history_index == 0 && !history_full) {
      history_full = true;
    }
    
    Serial.println("Real time:");
    Serial.print("pressure = ");
    Serial.println(pressure, 2);
    Serial.print("temperature = ");
    Serial.println(temperature, 2);
    Serial.print("altitude = ");
    Serial.println(altitude, 2);
    
  } else {
    // Prediction mode
    bool prediction_success = false;
    
    // Try TensorFlow model first
    if (interpreter != nullptr) {
      prediction_success = runModelPrediction(pressure, temperature, altitude);
    }
    
    // If model fails, use improved fallback
    if (!prediction_success && has_valid_history) {
      simplePrediction(pressure, temperature, altitude);
      prediction_success = true;
    }
    
    if (prediction_success) {
      Serial.println("Prediction:");
      Serial.print("pressure = ");
      Serial.println(pressure, 2);
      Serial.print("temperature = ");
      Serial.println(temperature, 2);
      Serial.print("altitude = ");
      Serial.println(altitude, 2);
    } else {
      Serial.println("System offline");
      Serial.print("pressure = ");
      Serial.println(last_pressure, 2);
      Serial.print("temperature = ");
      Serial.println(last_temperature, 2);
      Serial.print("altitude = ");
      Serial.println(last_altitude, 2);
    }
  }
  
  Serial.println("-------------------");
  delay(2000);
}