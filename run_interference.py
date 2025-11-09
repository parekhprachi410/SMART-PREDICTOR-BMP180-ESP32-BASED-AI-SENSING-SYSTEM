import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='bmp180_model.tflite')
interpreter.allocate_tensors()

inp_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]
print('Input details:', inp_det)
print('Output details:', out_det)

# Replace this with real preprocessed data (45 floats)
sample_input = np.zeros((1,45), dtype=np.float32)

# If you have scaler_params.npy (means & scales), load and apply before inference:
# scaler = np.load('scaler_params.npy', allow_pickle=True).item()
# sample_input = (raw_input - scaler['mean']) / scaler['scale']

interpreter.set_tensor(inp_det['index'], sample_input)
interpreter.invoke()
output = interpreter.get_tensor(out_det['index'])
print('Output:', output)
