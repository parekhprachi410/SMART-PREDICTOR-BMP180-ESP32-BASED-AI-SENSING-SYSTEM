import os

def tflite_to_c_array(tflite_path, output_path):
    """Convert .tflite file to C array"""
    
    # Read the .tflite file
    with open(tflite_path, 'rb') as f:
        tflite_data = f.read()
    
    # Generate C array
    array_name = os.path.basename(tflite_path).replace('.', '_')
    
    with open(output_path, 'w') as f:
        # Write the array declaration
        f.write(f"// Model data for {os.path.basename(tflite_path)}\n")
        f.write(f"// Generated from {tflite_path}\n")
        f.write(f"// Total size: {len(tflite_data)} bytes\n\n")
        
        f.write(f"const unsigned char {array_name}[] = {{\n")
        
        # Write bytes in hex format
        for i, byte in enumerate(tflite_data):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{byte:02x}")
            if i < len(tflite_data) - 1:
                f.write(", ")
            if (i + 1) % 12 == 0:
                f.write("\n")
        
        f.write("\n};\n")
        f.write(f"const unsigned int {array_name}_len = {len(tflite_data)};\n")
    
    print(f"✅ Converted {tflite_path} to {output_path}")
    print(f"✅ Array name: {array_name}")
    print(f"✅ Array length: {len(tflite_data)} bytes")

if __name__ == "__main__":
    tflite_to_c_array('bmp180_model.tflite', 'model.h')