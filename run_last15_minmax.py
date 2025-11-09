# run_last15_minmax.py
import numpy as np
import pandas as pd
import tensorflow as tf
import os, sys

MODEL = 'bmp180_model.tflite'
CSV = 'bmp180_data.csv'
SCALER = 'scaler_params.npy'
TIMESTEPS = 15   # 15*3 = 45 inputs

def load_minmax(path):
    d = np.load(path, allow_pickle=True).item()
    # expect keys data_min, data_max
    mn = np.array(d['data_min'], dtype=np.float32)
    mx = np.array(d['data_max'], dtype=np.float32)
    return mn, mx

def prepare_last15_flat(csv_path, timesteps=15):
    df = pd.read_csv(csv_path)
    expected = ['pressure','temperature','altitude']
    for c in expected:
        if c not in df.columns:
            raise SystemExit(f"CSV missing column {c}. Available: {df.columns.tolist()}")
    if len(df) < timesteps:
        raise SystemExit(f"CSV has only {len(df)} rows but need {timesteps}.")
    last = df[expected].iloc[-timesteps:].values  # shape (timesteps, 3)
    flat = last.flatten()   # row-major: t0_p,t0_t,t0_a, t1_p,...
    return flat.reshape(1, -1).astype(np.float32), df

def minmax_scale_flat(x_flat, mn, mx):
    # x_flat shape (1, timesteps*3)
    repeats = x_flat.shape[1] // 3
    mn_rep = np.tile(mn, repeats).astype(np.float32)
    mx_rep = np.tile(mx, repeats).astype(np.float32)
    scaled = (x_flat - mn_rep) / (mx_rep - mn_rep + 1e-12)
    return scaled, (mn_rep, mx_rep)

def inverse_minmax_outputs(y_scaled, mn, mx):
    # y_scaled shape (1,3)
    return (y_scaled * (mx - mn)) + mn

def main():
    if not os.path.exists(MODEL):
        print("Model file not found:", MODEL); return
    if not os.path.exists(CSV):
        print("CSV not found:", CSV); return
    if not os.path.exists(SCALER):
        print("Scaler file not found:", SCALER); return

    mn, mx = load_minmax(SCALER)
    print("Loaded min:", mn, "max:", mx)

    x_flat, df = prepare_last15_flat(CSV, TIMESTEPS)
    print("Prepared input flat shape:", x_flat.shape)

    x_scaled, (mn_rep, mx_rep) = minmax_scale_flat(x_flat, mn, mx)
    # optional: clamp to [0,1]
    x_scaled = np.clip(x_scaled, 0.0, 1.0).astype(np.float32)

    # Load model and run
    interpreter = tf.lite.Interpreter(model_path=MODEL)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    print("Model input expected shape:", inp_det['shape'], "dtype:", inp_det['dtype'])
    print("Model output shape:", out_det['shape'], "dtype:", out_det['dtype'])

    # Ensure shapes match
    try:
        inp_arr = x_scaled.astype(inp_det['dtype'])
        # if model expects a different first-dim (like -1), shapes OK
        interpreter.set_tensor(inp_det['index'], inp_arr)
    except Exception as e:
        print("Failed to set tensor:", e)
        print("Provided input shape:", x_scaled.shape)
        return

    interpreter.invoke()
    out_scaled = interpreter.get_tensor(out_det['index'])
    print("Model raw (scaled) output:", out_scaled)

    # Inverse transform outputs assuming same per-channel min/max order
    inv = inverse_minmax_outputs(out_scaled, mn, mx)
