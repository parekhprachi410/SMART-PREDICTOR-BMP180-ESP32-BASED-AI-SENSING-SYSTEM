#include <stdio.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"

static const char *TAG = "BMP180_MODEL";

// Scaler parameters from your training
const float data_min[3] = {1006.86, 28.90, 44.05};
const float data_max[3] = {1007.97, 30.90, 53.17};

// TensorFlow Lite variables
static tflite::MicroInterpreter* interpreter = NULL;
static TfLiteTensor* input = NULL;
static TfLiteTensor* output = NULL;

// Model sequence parameters
#define SEQ_LENGTH 15
#define NUM_FEATURES 3
#define INPUT_SIZE (SEQ_LENGTH * NUM_FEATURES)
#define OUTPUT_SIZE 3

// Circular buffer for sensor data
float sensor_buffer[SEQ_LENGTH][NUM_FEATURES];
int buffer_index = 0;

// Normalize function
void normalize_data(float* raw_input, float* normalized_output) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        normalized_output[i] = (raw_input[i] - data_min[i]) / (data_max[i] - data_min[i]);
    }
}

// Denormalize function  
void denormalize_data(float* normalized_input, float* raw_output) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        raw_output[i] = normalized_input[i] * (data_max[i] - data_min[i]) + data_min[i];
    }
}

// Initialize TensorFlow Lite
esp_err_t model_init(void) {
    ESP_LOGI(TAG, "Initializing TensorFlow Lite model...");
    
    // Map the model into a usable data structure
    const tflite::Model* model = tflite::GetModel(bmp180_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model provided is schema version %ld not equal to supported version %d",
                model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_FAIL;
    }
    
    // Create resolver with all operations
    static tflite::AllOpsResolver resolver;
    
    // Create interpreter
    static const int tensor_arena_size = 50 * 1024;
    static uint8_t* tensor_arena = (uint8_t*) heap_caps_malloc(tensor_arena_size, MALLOC_CAP_8BIT);
    
    if (tensor_arena == NULL) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena");
        return ESP_FAIL;
    }
    
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, tensor_arena_size);
    interpreter = &static_interpreter;
    
    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return ESP_FAIL;
    }
    
    // Get input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    ESP_LOGI(TAG, "Model initialized successfully");
    ESP_LOGI(TAG, "Input dimensions: %d", input->dims->data[1]);
    ESP_LOGI(TAG, "Output dimensions: %d", output->dims->data[1]);
    
    // Initialize sensor buffer with zeros
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            sensor_buffer[i][j] = 0.0f;
        }
    }
    
    return ESP_OK;
}

// Add new sensor data to buffer
void add_sensor_data(float pressure, float temperature, float altitude) {
    float new_data[NUM_FEATURES] = {pressure, temperature, altitude};
    float normalized_data[NUM_FEATURES];
    
    normalize_data(new_data, normalized_data);
    
    // Add to buffer
    for (int i = 0; i < NUM_FEATURES; i++) {
        sensor_buffer[buffer_index][i] = normalized_data[i];
    }
    
    buffer_index = (buffer_index + 1) % SEQ_LENGTH;
}

// Run prediction
esp_err_t model_predict(float* result_pressure, float* result_temperature, float* result_altitude) {
    if (interpreter == NULL) {
        ESP_LOGE(TAG, "Model not initialized");
        return ESP_FAIL;
    }
    
    // Prepare input data from buffer
    float input_data[INPUT_SIZE];
    int input_idx = 0;
    
    for (int i = 0; i < SEQ_LENGTH; i++) {
        int idx = (buffer_index + i) % SEQ_LENGTH;
        for (int j = 0; j < NUM_FEATURES; j++) {
            input_data[input_idx++] = sensor_buffer[idx][j];
        }
    }
    
    // Copy input data to tensor
    for (int i = 0; i < INPUT_SIZE; i++) {
        input->data.f[i] = input_data[i];
    }
    
    // Run inference
    int64_t start_time = esp_timer_get_time();
    TfLiteStatus invoke_status = interpreter->Invoke();
    int64_t inference_time = esp_timer_get_time() - start_time;
    
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return ESP_FAIL;
    }
    
    // Denormalize output
    float normalized_output[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        normalized_output[i] = output->data.f[i];
    }
    
    float raw_output[OUTPUT_SIZE];
    denormalize_data(normalized_output, raw_output);
    
    *result_pressure = raw_output[0];
    *result_temperature = raw_output[1];
    *result_altitude = raw_output[2];
    
    ESP_LOGI(TAG, "Inference time: %lld us", inference_time);
    
    return ESP_OK;
}

// Simulate sensor data (we'll use fake data for testing)
void read_sensor_data(float* pressure, float* temperature, float* altitude) {
    // Simulate realistic sensor readings
    *pressure = 1007.0f + ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    *temperature = 29.5f + ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    *altitude = 48.0f + ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
}

void app_main(void) {
    ESP_LOGI(TAG, "Starting BMP180 Sensor Prediction Model");
    
    // Initialize model
    if (model_init() != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize model");
        return;
    }
    
    // Main loop
    int cycle_count = 0;
    while (1) {
        // Read sensor data
        float pressure, temperature, altitude;
        read_sensor_data(&pressure, &temperature, &altitude);
        
        ESP_LOGI(TAG, "Sensor Reading - P: %.2f hPa, T: %.2f C, A: %.2f m", 
                pressure, temperature, altitude);
        
        // Add to buffer
        add_sensor_data(pressure, temperature, altitude);
        
        // Only predict after we have enough data
        if (cycle_count >= SEQ_LENGTH) {
            float pred_pressure, pred_temperature, pred_altitude;
            
            if (model_predict(&pred_pressure, &pred_temperature, &pred_altitude) == ESP_OK) {
                ESP_LOGI(TAG, "PREDICTION - P: %.2f hPa, T: %.2f C, A: %.2f m", 
                        pred_pressure, pred_temperature, pred_altitude);
            }
        } else {
            ESP_LOGI(TAG, "Collecting data... (%d/%d)", cycle_count + 1, SEQ_LENGTH);
        }
        
        cycle_count++;
        vTaskDelay(2000 / portTICK_PERIOD_MS);  // Wait 2 seconds
    }
}