from __future__ import annotations

from .unet import UNet
from .unetpp import UNetPlusPlus


def build_model(config: dict) -> UNet | UNetPlusPlus:
    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    name = model_cfg["name"]
    kwargs = {"num_classes": dataset_cfg["num_classes"], "in_channels": model_cfg.get("in_channels", 3), "channels": tuple(model_cfg.get("channels", [64, 128, 256, 512]))}
    if name == "unet":
        return UNet(**kwargs)
    if name == "unetpp":
        kwargs["channels"] = tuple(model_cfg.get("channels", [64, 128, 256, 512, 1024]))
        return UNetPlusPlus(**kwargs, deep_supervision=model_cfg.get("deep_supervision", False))
    raise ValueError(f"Unknown model: {name}")
