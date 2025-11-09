# inspect_scaler.py
import numpy as np, pprint, os
path = 'scaler_params.npy'
if not os.path.exists(path):
    print(path, "not found")
else:
    obj = np.load(path, allow_pickle=True)
    try:
        item = obj.item()
    except Exception:
        item = obj
    print("Type:", type(item))
    pprint.pprint(item)
    # If values are arrays, show shapes
    if isinstance(item, dict):
        for k,v in item.items():
            try:
                import numpy as _np
                print(k, "-> type:", type(v), "shape:", getattr(_np.array(v), 'shape', None))
            except Exception:
                pass
