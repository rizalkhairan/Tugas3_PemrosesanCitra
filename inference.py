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
    (64, 128, 64),      # Animal
    (192, 0, 128),      # Archway
    (0, 128, 192),      # Bicyclist
    (0, 128, 64),       # Bridge
    (128, 0, 0),        # Building
    (64, 0, 128),       # Car
    (64, 0, 192),       # CartLuggagePram
    (192, 128, 64),     # Child
    (192, 192, 128),    # Column_Pole
    (64, 64, 128),      # Fence
    (128, 0, 192),      # LaneMkgsDriv
    (192, 0, 64),       # LaneMkgsNonDriv
    (128, 128, 64),     # Misc_Text
    (192, 0, 192),      # MotorcycleScooter
    (128, 64, 64),      # OtherMoving
    (64, 192, 128),     # ParkingBlock
    (64, 64, 0),        # Pedestrian
    (128, 64, 128),     # Road
    (128, 128, 192),    # RoadShoulder
    (0, 0, 192),        # Sidewalk
    (192, 128, 128),    # SignSymbol
    (128, 128, 128),    # Sky
    (64, 128, 192),     # SUVPickupTruck
    (0, 0, 64),         # TrafficCone
    (0, 64, 64),        # TrafficLight
    (192, 64, 128),     # Train
    (128, 128, 0),      # Tree
    (192, 128, 192),    # Truck_Bus
    (64, 0, 64),        # Tunnel
    (192, 192, 0),      # VegetationMisc
    (64, 192, 0),       # Wall
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
