# Tugas3_PemrosesanCitra

# Setup
This app uses a deep learning model for segmentation. To download the model run
```
python setup_model.py
```

# Requirements
This app is tested on MATLAB 2026a. The following python modules are also required
- torch
- torchvision
- segmentation-models-pytorch
- numpy
- Pillow
- gdown (to download the model)
```
uv pip install torch torchvision segmentation-models-pytorch==0.5.0 numpy Pillow gdown
```