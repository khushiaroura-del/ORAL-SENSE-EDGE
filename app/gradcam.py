import torch
import cv2
import numpy as np

from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from torch import nn


# ==========================================
# ORAL-SENSE EDGE
# GRAD-CAM EXPLAINABLE AI
# ==========================================

print("--------------------------------")
print("ORAL-SENSE EDGE")
print("GRAD-CAM EXPLANATION")
print("--------------------------------")


IMAGE_PATH = "results/current_image.jpg"
MODEL_PATH = "model/oral_sense_mobilenetv3.pth"
OUTPUT_PATH = "results/gradcam_result.jpg"


# ==========================================
# DEVICE
# ==========================================

device = torch.device("cpu")

print("Device:", device)


# ==========================================
# LOAD IMAGE
# ==========================================

original_image = cv2.imread(IMAGE_PATH)

if original_image is None:
    print("ERROR: Image nahi mili.")
    print("Expected:", IMAGE_PATH)
    exit()

print("Image loaded successfully!")


# ==========================================
# PREPARE IMAGE
# ==========================================

rgb_image = cv2.cvtColor(
    original_image,
    cv2.COLOR_BGR2RGB
)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensor = transform(
    rgb_image
).unsqueeze(0)

input_tensor = input_tensor.to(device)


# ==========================================
# LOAD MODEL
# ==========================================

model = mobilenet_v3_small(
    weights=None
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

classes = checkpoint["classes"]

model = model.to(device)

model.eval()


# ==========================================
# FIND LAST CONVOLUTIONAL LAYER
# ==========================================

target_layer = model.features[-1]

activations = None
gradients = None


# ==========================================
# FORWARD HOOK
# ==========================================

def forward_hook(
    module,
    input,
    output
):

    global activations

    activations = output


# ==========================================
# BACKWARD HOOK
# ==========================================

def backward_hook(
    module,
    grad_input,
    grad_output
):

    global gradients

    gradients = grad_output[0]


target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)


# ==========================================
# FORWARD PASS
# ==========================================

output = model(
    input_tensor
)

probabilities = torch.softmax(
    output,
    dim=1
)

predicted_class = torch.argmax(
    probabilities,
    dim=1
).item()

probability = probabilities[
    0,
    predicted_class
].item() * 100


print()
print("Prediction:", classes[predicted_class])

print(
    "Model probability:",
    f"{probability:.2f}%"
)


# ==========================================
# BACKWARD PASS
# ==========================================

model.zero_grad()

score = output[
    0,
    predicted_class
]

score.backward()


# ==========================================
# GRAD-CAM CALCULATION
# ==========================================

activation = activations[0]

gradient = gradients[0]


weights = gradient.mean(
    dim=(1, 2)
)


cam = torch.zeros(
    activation.shape[1:],
    dtype=torch.float32
)


for i in range(
    activation.shape[0]
):

    cam += (
        weights[i]
        * activation[i]
    )


cam = torch.relu(cam)


# ==========================================
# NORMALIZE HEATMAP
# ==========================================

cam = cam.detach().numpy()

cam -= cam.min()

if cam.max() != 0:

    cam /= cam.max()


# ==========================================
# RESIZE HEATMAP
# ==========================================

height, width = original_image.shape[:2]

heatmap = cv2.resize(
    cam,
    (width, height)
)


heatmap = np.uint8(
    255 * heatmap
)


heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)


# ==========================================
# CREATE OVERLAY
# ==========================================

overlay = cv2.addWeighted(
    original_image,
    0.6,
    heatmap,
    0.4,
    0
)


# ==========================================
# SAVE RESULT
# ==========================================

cv2.imwrite(
    OUTPUT_PATH,
    overlay
)


print()
print("--------------------------------")
print("GRAD-CAM COMPLETE")
print("--------------------------------")

print(
    "Heatmap saved at:"
)

print(OUTPUT_PATH)

print()
print(
    "Note: Heatmap shows regions "
    "that influenced the model prediction."
)