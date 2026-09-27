import torch
import cv2

from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from torch import nn


# ==========================================
# ORAL-SENSE EDGE
# REAL IMAGE PREDICTION
# ==========================================

print("--------------------------------")
print("ORAL-SENSE EDGE")
print("AI SCREENING PREDICTION")
print("--------------------------------")


# ------------------------------------------
# Settings
# ------------------------------------------

IMAGE_PATH = "results/current_image.jpg"

MODEL_PATH = "model/oral_sense_mobilenetv3.pth"


# ------------------------------------------
# Device
# ------------------------------------------

device = torch.device("cpu")

print("Device:", device)


# ------------------------------------------
# Load image
# ------------------------------------------

image = cv2.imread(IMAGE_PATH)


if image is None:

    print()
    print("ERROR: Image nahi mili.")
    print("Expected image:")
    print(IMAGE_PATH)

    exit()


print("Image loaded successfully!")


# ------------------------------------------
# Convert OpenCV image to RGB
# ------------------------------------------

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ------------------------------------------
# Image transformation
# ------------------------------------------

transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


image_tensor = transform(image)


# Add batch dimension

image_tensor = image_tensor.unsqueeze(0)


image_tensor = image_tensor.to(device)


# ------------------------------------------
# Load MobileNetV3
# ------------------------------------------

model = mobilenet_v3_small(
    weights=None
)


# ------------------------------------------
# Two classes
# ------------------------------------------

number_of_classes = 2


model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    number_of_classes
)


# ------------------------------------------
# Load trained model
# ------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


# Get class names

classes = checkpoint["classes"]


model = model.to(device)

model.eval()


# ------------------------------------------
# Prediction
# ------------------------------------------

with torch.no_grad():

    outputs = model(
        image_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    predicted_index = torch.argmax(
        probabilities,
        dim=1
    ).item()


# ------------------------------------------
# Result
# ------------------------------------------

predicted_class = classes[
    predicted_index
]

probability = probabilities[
    0,
    predicted_index
].item() * 100


print()
print("--------------------------------")
print("SCREENING RESULT")
print("--------------------------------")

print(
    "Prediction:",
    predicted_class
)

print(
    "Model probability:",
    f"{probability:.2f}%"
)

print("--------------------------------")

print()
print("Note:")
print(
    "This is an AI screening prototype, "
    "not a medical diagnosis."
)