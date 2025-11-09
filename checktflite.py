import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path='bmp180_model.tflite')
interpreter.allocate_tensors()
print('Inputs:', interpreter.get_input_details())
print('Outputs:', interpreter.get_output_details())
# Check quantization
td = interpreter.get_tensor_details()
qcount = sum(1 for t in td if t.get('quantization') and t.get('quantization')[0] != 0)
print('Total tensors:', len(td), 'Quantized tensors with scale!=0:', qcount)
