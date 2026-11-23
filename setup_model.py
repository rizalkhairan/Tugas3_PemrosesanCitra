from pathlib import Path
from tempfile import TemporaryDirectory

MODEL_ID = "16GsadY7nYgXxbBNEgUfkPuWZDDmViK_F"
MODEL_PATH = Path(__file__).resolve().parent / "model" / "segmentation-model.pt"


def main() -> None:
    if MODEL_PATH.is_file():
        print(f"Already installed: {MODEL_PATH}")
        return

    import gdown
    import torch

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=MODEL_PATH.parent) as folder:
        downloaded = Path(folder) / MODEL_PATH.name
        if gdown.download(id=MODEL_ID, output=str(downloaded)) is None:
            raise RuntimeError("Could not download the model from Google Drive.")

        checkpoint = torch.load(downloaded, map_location="meta", weights_only=True)
        weights = checkpoint.get("model", checkpoint)
        head = weights.get("segmentation_head.0.weight")
        if head is None or head.shape[0] != 31:
            raise ValueError("Expected a 31-class CamVid checkpoint.")
        downloaded.replace(MODEL_PATH)

    print(f"Installed: {MODEL_PATH}")


if __name__ == "__main__":
    main()
