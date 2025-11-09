import serial
import time
import csv

# Replace 'COM3' with your ESP32's serial port (check Device Manager for the correct COM port)
ser = serial.Serial('COM5', 115200, timeout=1)
time.sleep(2)  # Wait for connection

data = []
start_time = time.time()
duration = 3600  # Collect for 30 minutes (1800 seconds); adjust as needed (e.g., 600 for 10 mins)

print("Collecting data... Press Ctrl+C to stop early.")

try:
    while time.time() - start_time < duration:
        line = ser.readline().decode('utf-8').strip()
        if line and not line.startswith("timestamp"):  # Skip header if repeated
            parts = line.split(',')
            if len(parts) == 4:
                try:
                    timestamp, pressure, temperature, altitude = map(float, parts)
                    data.append([timestamp, pressure, temperature, altitude])
                    print(f"Collected: {line}")
                except ValueError:
                    print(f"Skipped invalid line: {line}")
except KeyboardInterrupt:
    print("Stopped early by user.")

ser.close()

# Save to CSV
if data:
    with open('bmp180_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'pressure', 'temperature', 'altitude'])
        writer.writerows(data)
    print(f"Data saved to bmp180_data.csv ({len(data)} rows)")
else:
    print("No data collected. Check ESP32 connection and serial port.")
