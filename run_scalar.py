# run_with_scaler.py
import numpy as np
import pandas as pd
import tensorflow as tf
import os

MODEL = 'bmp180_model.tflite'
CSV = 'bmp180_data.csv'       # your dataset file
SCALER_NPY = 'scaler_params.npy'  # you mentioned this exists
# If you used separate input/output scalers, adapt names or structure.

def load_scaler(path):
    """Load scaler info saved as numpy file. Accepts dict-like formats:
       {'mean':..., 'scale':...} or {'min':..., 'max':...} or sklearn scaler saved .get_params() style.
       Returns a callable for transform and inverse_transform.
    """
    stuff = np.load(path, allow_pickle=True).item()
    # Try common formats
    if 'mean' in stuff and 'scale' in stuff:
        mean = np.array(stuff['mean'])
        scale = np.array(stuff['scale'])
        def inverse(y): return (y * scale) + mean
        def forward(x): return (x - mean) / scale
        return forward, inverse
    if 'min' in stuff and 'max' in stuff:
        mn = np.array(stuff['min'])
        mx = np.array(stuff['max'])
        def inverse(y): return y * (mx - mn) + mn
        def forward(x): return (x - mn) / (mx - mn)
        return forward, inverse
    # If it's a sklearn scaler (saved attributes)
    if 'scale_' in stuff and 'mean_' in stuff:
        mean = np.array(stuff['mean_'])
        scale = np.array(stuff['scale_'])
        def inverse(y): return (y * scale) + mean
        def forward(x): return (x - mean) / scale
        return forward, inverse
    # Fallback: try dictionary with 'input'/'output' subdicts
    if 'input' in stuff or 'output' in stuff:
        return stuff  # caller will handle
    raise ValueError("Unknown scaler format in {}".format(path))

def main():
    # Load model
    interpreter = tf.lite.Interpreter(model_path=MODEL)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    print("Model input shape:", inp_det['shape'], "dtype:", inp_det['dtype'])
    print("Model output shape:", out_det['shape'], "dtype:", out_det['dtype'])

    # Load scaler params
    if os.path.exists(SCALER_NPY):
        try:
            forward_inv = load_scaler(SCALER_NPY)
        except Exception as e:
            print("Couldn't parse scaler file:", e)
            forward_inv = None
    else:
        forward_inv = None
        print("No scaler_params.npy found; will run without inverse transform.")

    # Load a sample row from CSV as raw sensor features (first row by default)
    df = pd.read_csv(CSV)
    print("CSV columns:", df.columns.tolist())
    # Adjust column selection to the exact 45-feature input your model expects.
    # For example, if df already contains only the 45 features in the correct order:
    sample = df.iloc[0].values.astype(np.float32)     # first row
    if sample.shape[0] != 45:
        raise SystemExit(f"Sample length {sample.shape[0]} != 45. Adjust your CSV or feature selection.")
    sample = sample.reshape(1,45)

    # If forward_inv is a tuple (forward, inverse)
    if isinstance(forward_inv, tuple):
        forward, inverse = forward_inv
        scaled_input = forward(sample).astype(np.float32)
    else:
        # if scaler file contains separate input/output dicts:
        if isinstance(forward_inv, dict) and 'input' in forward_inv:
            inp = forward_inv['input']
            if 'mean' in inp and 'scale' in inp:
                mean = np.array(inp['mean']); scale = np.array(inp['scale'])
                scaled_input = ((sample - mean) / scale).astype(np.float32)
            else:
                print("Unknown input scaler structure; using raw sample instead.")
                scaled_input = sample.astype(np.float32)
        else:
            print("No usable scaler loaded; using raw sample (may be incorrect).")
            scaled_input = sample.astype(np.float32)

    # Run inference
    interpreter.set_tensor(inp_det['index'], scaled_input)
    interpreter.invoke()
    out = interpreter.get_tensor(out_det['index'])
    print("Model raw output:", out)

    # Inverse transform output if possible
    if isinstance(forward_inv, tuple):
        _, inverse = forward_inv
        real_output = inverse(out)
        print("Inverse-transformed output (real units):", real_output)
    elif isinstance(forward_inv, dict) and 'output' in forward_inv:
        outsc = forward_inv['output']
        if 'mean' in outsc and 'scale' in outsc:
            mean = np.array(outsc['mean']); scale = np.array(outsc['scale'])
            real_output = (out * scale) + mean
            print("Inverse-transformed output (real units):", real_output)
        else:
            print("Output scaler format unknown; raw outputs shown above.")
    else:
        print("No output scaler available, raw model outputs shown.")

if __name__ == '__main__':
    main()
