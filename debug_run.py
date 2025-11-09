# debug_run.py
import os, sys, traceback, numpy as np, pandas as pd

print("Starting debug_run.py")

try:
    print("\n1) Check files exist:")
    for name in ('bmp180_model.tflite', 'bmp180_data.csv', 'scaler_params.npy'):
        print(" ", name, "->", os.path.exists(name))
except Exception:
    print("Failed file existence check")
    traceback.print_exc()

try:
    print("\n2) Load scaler_params.npy")
    sc = np.load('scaler_params.npy', allow_pickle=True).item()
    print("  type:", type(sc))
    print("  keys:", list(sc.keys()))
    for k,v in sc.items():
        print("   ", k, "->", v)
except Exception:
    print("Failed loading scaler")
    traceback.print_exc()

try:
    print("\n3) Read CSV and show tail(15)")
    df = pd.read_csv('bmp180_data.csv')
    print("  shape:", df.shape)
    print(df.tail(15).to_string(index=False))
except Exception:
    print("Failed reading CSV")
    traceback.print_exc()

try:
    print("\n4) Prepare last15 flat input")
    expected = ['pressure','temperature','altitude']
    for c in expected:
        if c not in df.columns:
            raise RuntimeError(f"Missing column: {c}")
    if len(df) < 15:
        raise RuntimeError("Not enough rows")
    last = df[expected].iloc[-15:].values
    flat = last.flatten().reshape(1, -1).astype(np.float32)
    print("  flat shape:", flat.shape)
    print("  first 12 values:", flat[0,:12].tolist())
except Exception:
    print("Failed preparing flat")
    traceback.print_exc()

try:
    print("\n5) Scale using min/max from scaler")
    mn = np.array(sc['data_min'], dtype=np.float32)
    mx = np.array(sc['data_max'], dtype=np.float32)
    print("  min:", mn, "max:", mx)
    repeats = flat.shape[1] // 3
    mn_rep = np.tile(mn, repeats)
    mx_rep = np.tile(mx, repeats)
    scaled = (flat - mn_rep) / (mx_rep - mn_rep + 1e-12)
    print("  scaled shape:", scaled.shape)
    print("  first 12 scaled values:", scaled[0,:12].tolist())
except Exception:
    print("Failed scaling")
    traceback.print_exc()

try:
    print("\n6) Load TFLite model and interpreter")
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path='bmp180_model.tflite')
    print("  Interpreter created")
    print("  Allocating tensors...")
    interpreter.allocate_tensors()
    print("  Allocated")
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    print("  input detail:", inp_det)
    print("  output detail:", out_det)
except Exception:
    print("Failed loading/interpreting model")
    traceback.print_exc()

try:
    print("\n7) Set tensor and invoke")
    # ensure dtype
    arr = scaled.astype(inp_det['dtype'])
    print("  trying to set tensor with shape", arr.shape, "dtype", arr.dtype)
    interpreter.set_tensor(inp_det['index'], arr)
    interpreter.invoke()
    out = interpreter.get_tensor(out_det['index'])
    print("  raw output:", out)
    # inverse transform outputs
    inv = (out * (mx - mn)) + mn
    print("  inverse output (real units):", inv)
except Exception:
    print("Failed during inference")
    traceback.print_exc()

print("\nFinished debug_run.py")
