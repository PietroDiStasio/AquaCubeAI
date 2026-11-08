"""Original MLP and its parameter-sharing 1 x 1 convolutional reformulation."""

from pathlib import Path
import torch
import torch.nn as nn


class RegressNN(nn.Module):

    def __init__(self, input_size, hidden_layers, output_channels):
        super(RegressNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.BatchNorm1d(hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.25))
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.BatchNorm1d(hidden_layers[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.25))
        layers.append(nn.Linear(hidden_layers[-1], output_channels))
        layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def get_depthwise_conv_layer(window_size, in_channels):
    conv = nn.Conv2d(
        in_channels,
        in_channels,
        window_size,
        stride=window_size,
        padding=0,
        dilation=1,
        groups=in_channels,
        bias=False,
    )
    weight = nn.Parameter(
        1 / window_size * 1 / window_size * torch.ones([in_channels, 1, window_size, window_size]),
        requires_grad=False,
    )
    conv.load_state_dict({"weight": weight}, strict=False)
    conv.weight.requires_grad = False
    return conv


def get_intermediate_conv_layer(weight, bias):
    conv = nn.Conv2d(
        weight.shape[1], weight.shape[0], 1, stride=1, padding=0, dilation=1, groups=1, bias=True
    )
    weight_unsqueezed = nn.Parameter(weight.unsqueeze(-1).unsqueeze(-1), requires_grad=False)
    bias_repeated = nn.Parameter(bias, requires_grad=False)
    conv.load_state_dict({"weight": weight_unsqueezed, "bias": bias_repeated}, strict=False)
    conv.weight.requires_grad = False
    conv.bias.requires_grad = False
    return conv


def get_RegressNN_model(
    input_size, hidden_layers, output_channels, checkpoint_path="best_model.pth"
):
    model_regress = RegressNN(input_size, hidden_layers, output_channels)
    model_regress.eval()
    checkpoint_load = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_regress.load_state_dict(checkpoint_load)
    return model_regress


def get_BatchNorm2D(weight, bias, running_mean, running_var):
    batch_norm_2d = nn.BatchNorm2d(weight.shape, track_running_stats=True)
    batch_norm_2d.load_state_dict(
        {"weight": weight, "bias": bias, "running_mean": running_mean, "running_var": running_var},
        strict=False,
    )
    batch_norm_2d.weight.requires_grad = False
    batch_norm_2d.bias.requires_grad = False
    batch_norm_2d.eval()
    return batch_norm_2d


class AL4EOTNet(nn.Module):

    def __init__(self, window_size, in_channels, hidden_layers, output_channels, checkpoint_path):
        super(AL4EOTNet, self).__init__()
        depthwise_average_conv = get_depthwise_conv_layer(window_size, in_channels)
        regress_nn = get_RegressNN_model(
            in_channels, hidden_layers, output_channels, checkpoint_path
        )
        layers = [depthwise_average_conv]
        for n in range(len(regress_nn.model)):
            if isinstance(regress_nn.model[n], nn.Linear):
                layers.append(
                    get_intermediate_conv_layer(
                        regress_nn.model[n].weight, regress_nn.model[n].bias
                    )
                )
            elif isinstance(regress_nn.model[n], nn.BatchNorm1d):
                layers.append(
                    get_BatchNorm2D(
                        regress_nn.model[n].weight,
                        regress_nn.model[n].bias,
                        regress_nn.model[n].running_mean,
                        regress_nn.model[n].running_var,
                    )
                )
            elif isinstance(regress_nn.model[n], nn.ReLU):
                layers.append(nn.ReLU())
            else:
                pass
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def load_area_model(model_dir, device="cpu"):
    """Load the supplied spatial state dictionary using its architecture metadata."""
    model_dir = Path(model_dir)
    ckpt = torch.load(model_dir / "model_area_al4eotnet.pth", map_location="cpu", weights_only=True)
    if int(ckpt.get("patch_hw", 20)) != 20 or int(ckpt["window_size"]) != 1:
        raise ValueError("Expected the paper's 20 x 20 dense spatial checkpoint.")
    checkpoint_name = Path(
        str(ckpt.get("checkpoint_path_mlp", "best_model_512x512x256.pth")).replace("\\", "/")
    ).name
    model = AL4EOTNet(
        ckpt["window_size"],
        ckpt["in_channels"],
        list(ckpt["hidden_layers"]),
        ckpt["output_channels"],
        model_dir / checkpoint_name,
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device).eval()


def save_spatial_model(checkpoint_path, hidden_layers, output_dir, patch_hw=20, pca_enabled=False):
    """Wrap trained weights and export the fixed-shape opset-11 ONNX model."""
    if pca_enabled:
        raise ValueError(
            "Dense export requires original bands; disable PCA for the paper workflow."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = AL4EOTNet(1, 8, list(hidden_layers), 1, checkpoint_path).cpu().eval()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "window_size": 1,
            "in_channels": 8,
            "output_channels": 1,
            "hidden_layers": tuple(hidden_layers),
            "checkpoint_path_mlp": Path(checkpoint_path).name,
            "patch_hw": patch_hw,
        },
        output_dir / "model_area_al4eotnet.pth",
    )
    dummy = torch.randn(1, 8, patch_hw, patch_hw, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output_dir / "model_turbidity_area.onnx"),
        input_names=["input"],
        output_names=["y_pred"],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(output_dir / "model_turbidity_area.onnx"), providers=["CPUExecutionProvider"]
    )
    shape = session.run(None, {"input": dummy.numpy()})[0].shape
    if shape != (1, 1, patch_hw, patch_hw):
        raise RuntimeError(f"Unexpected ONNX output: {shape}")
    return model
