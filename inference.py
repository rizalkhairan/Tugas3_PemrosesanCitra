from functools import lru_cache
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torchvision.transforms.functional import normalize, to_tensor

MODEL_PATH = Path(__file__).resolve().parent / "model" / "segmentation-model.pt"
# Row index is the training class ID.
PALETTE = np.array([
    (128, 128, 128),  # Sky
    (128, 0, 0),      # Building
    (192, 192, 128),  # Pole
    (128, 64, 128),   # Road
    (0, 0, 192),      # Pavement
    (128, 128, 0),    # Tree
    (192, 128, 128),  # SignSymbol
    (64, 64, 128),    # Fence
    (64, 0, 128),     # Car
    (64, 64, 0),      # Pedestrian
    (0, 128, 192),    # Bicyclist
], dtype=np.uint8)


@lru_cache(maxsize=1)
def _load_model() -> tuple[torch.nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None, in_channels=3,
        classes=len(PALETTE), activation=None,
    )
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    return model.to(device).eval(), device


@torch.inference_mode()
def segment(image: np.ndarray) -> np.ndarray:
    pixels = np.asarray(image)
    if pixels.dtype != np.uint8:
        raise ValueError("Expected a uint8 image.")
    rgb = Image.fromarray(pixels).convert("RGB")
    tensor = normalize(to_tensor(rgb), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    model, device = _load_model()
    batch = tensor.unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logits = model(batch)
    prediction = logits.argmax(dim=1)[0].cpu().numpy()
    return PALETTE[prediction]
