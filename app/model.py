import torch
from torchvision.models import mobilenet_v3_small


print("--------------------------------")
print("ORAL-SENSE EDGE")
print("MobileNetV3 Model Test")
print("--------------------------------")


model = mobilenet_v3_small(weights="DEFAULT")

model.eval()

print("MobileNetV3 successfully loaded!")

print("Model ready for AI processing.")