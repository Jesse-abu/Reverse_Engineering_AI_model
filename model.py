import torch
from ultralytics import YOLO

model_obj = YOLO("runs\detect\\train14\\weights\\last.pt")

torch.save(model_obj.model.state_dict(), "weights_only.pt")

model_obj.train(data='data.yaml', epochs=40, device='cpu')