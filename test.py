import torch
import torchvision
import torchaudio

print(torch.__version__) # 2.0.0
print(torchvision.__version__) #0.15.1
print(torchaudio.__version__) #2.0.1

print(torch.cuda.is_available())

"""
if __name__ == '__main__:
    from ultralytics import YOLO
    model = YOLO('yolov8n.yaml')
"""