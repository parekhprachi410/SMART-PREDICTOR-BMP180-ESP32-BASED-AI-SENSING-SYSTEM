import numpy as np
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path='bmp180_model.tflite')
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
sample_input = np.zeros(inp['shape'], dtype=inp['dtype'])
interpreter.set_tensor(inp['index'], sample_input)
interpreter.invoke()
print(interpreter.get_tensor(out['index']))
