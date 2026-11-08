# Reference models — paper workflow

These are the supplied 20 x 20 reference artifacts; the notebooks do not overwrite them.

| File | Purpose |
| --- | --- |
| `best_model_512x512x256.pth` | Selected MLP state dictionary; eight input descriptors, one target |
| `model_area_al4eotnet.pth` | Dense spatial state dictionary with architecture metadata |
| `norm_params_tur.npz` | Training-only target min/max and denominator |
| `model_turbidity_area.onnx` | Fixed input `[1, 8, 20, 20]`, output `[1, 1, 20, 20]`, opset 11 |
| `myriad_fp16/model_turbidity_area.xml` and `.bin` | Legacy OpenVINO FP16 IR for device execution |

Both MLP and spatial outputs are in normalized target units. To obtain FNU,
apply `prediction * denom + min` using the matching NPZ file, then clamp only
the lower bound to zero. There is no upper clipping at 140 FNU.

The spatial network applies the MLP mapping through 1 x 1 convolutions. It
contains 401,417 parameters, including the fixed depthwise input layer.
The float checkpoint, ONNX and IR files are separate runtime artifacts. Use
paired inputs when measuring numerical differences between them.

New training artifacts are written to `outputs/training/`. Keep the selected MLP,
spatial checkpoint and target-normalization file from the same run together.
