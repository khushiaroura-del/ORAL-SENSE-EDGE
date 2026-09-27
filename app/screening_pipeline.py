import os
import cv2
import torch
import numpy as np

from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from quality_check import check_image_quality


# ==========================================
# ORAL-SENSE EDGE
# UNIFIED SCREENING PIPELINE
# ==========================================

IMAGE_PATH = "results/current_image.jpg"
MODEL_PATH = "model/oral_sense_mobilenetv3.pth"
OUTPUT_PATH = "results/screening_gradcam.jpg"


print("--------------------------------")
print("ORAL-SENSE EDGE")
print("UNIFIED SCREENING PIPELINE")
print("--------------------------------")


# ==========================================
# 1. LOAD IMAGE
# ==========================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Image nahi mili.")
    print("Expected:", IMAGE_PATH)
    exit()

print("Image loaded successfully!")


# ==========================================
# 2. IMAGE QUALITY CHECK
# ==========================================

quality, brightness, sharpness = check_image_quality(image)

print()
print("IMAGE QUALITY")
print("--------------------------------")
print("Result:", quality)
print("Brightness:", round(brightness, 2))
print("Sharpness:", round(sharpness, 2))


# ==========================================
# STOP IF IMAGE IS NOT GOOD
# ==========================================

if quality != "GOOD IMAGE":

    print()
    print("--------------------------------")
    print("IMAGE NOT SUITABLE")
    print("--------------------------------")

    print("Please take another clearer image.")

    exit()


print()
print("Image quality is good.")
print("Continuing to AI screening...")


# ==========================================
# 3. PREPARE IMAGE
# ==========================================

device = torch.device("cpu")

original_image = image.copy()

rgb_image = cv2.cvtColor(
    image,
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


# ==========================================
# 4. LOAD MOBILEV3 MODEL
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
# 5. GRAD-CAM HOOKS
# ==========================================

target_layer = model.features[-1]

activations = None
gradients = None


def forward_hook(module, input, output):

    global activations

    activations = output


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
# 6. AI PREDICTION
# ==========================================

output = model(input_tensor)

probabilities = torch.softmax(
    output,
    dim=1
)


predicted_index = torch.argmax(
    probabilities,
    dim=1
).item()


predicted_class = classes[
    predicted_index
]


probability = probabilities[
    0,
    predicted_index
].item() * 100


print()
print("--------------------------------")
print("AI SCREENING RESULT")
print("--------------------------------")

print(
    "Predicted class:",
    predicted_class
)

print(
    "Model probability:",
    f"{probability:.2f}%"
)


# ==========================================
# 7. GRAD-CAM
# ==========================================

model.zero_grad()


score = output[
    0,
    predicted_index
]


score.backward()


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


cam = cam.detach().numpy()


cam -= cam.min()


if cam.max() != 0:

    cam /= cam.max()


# ==========================================
# 8. CREATE HEATMAP
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
# 9. OVERLAY
# ==========================================

overlay = cv2.addWeighted(
    original_image,
    0.6,
    heatmap,
    0.4,
    0
)


# ==========================================
# 10. SAVE RESULT
# ==========================================

os.makedirs(
    "results",
    exist_ok=True
)


cv2.imwrite(
    OUTPUT_PATH,
    overlay
)


# ==========================================
# FINAL REPORT
# ==========================================

print()
print("--------------------------------")
print("SCREENING PIPELINE COMPLETE")
print("--------------------------------")

print("Image quality:", quality)

print(
    "Predicted class:",
    predicted_class
)

print(
    "Model probability:",
    f"{probability:.2f}%"
)

print()
print("Grad-CAM saved at:")
print(OUTPUT_PATH)

print()
print("--------------------------------")
print("IMPORTANT")
print("--------------------------------")

print(
    "This is an AI screening prototype, "
    "not a medical diagnosis."
)

print(
    "Grad-CAM shows regions that influenced "
    "the model prediction; it does not prove "
    "the presence or location of a lesion."
)