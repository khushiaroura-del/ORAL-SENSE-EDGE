import torch
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small
from torch import nn
from torch.utils.data import DataLoader


print("--------------------------------")
print("ORAL-SENSE EDGE")
print("TEST SET EVALUATION")
print("--------------------------------")


# Device
device = torch.device("cpu")

print("Device:", device)


# --------------------------------
# Image transformation
# --------------------------------

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------
# Load test dataset
# --------------------------------

test_dataset = datasets.ImageFolder(
    "dataset/test",
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


print()
print("Classes:", test_dataset.classes)
print("Test images:", len(test_dataset))


# --------------------------------
# Load MobileNetV3
# --------------------------------

model = mobilenet_v3_small(
    weights=None
)

number_of_classes = len(
    test_dataset.classes
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    number_of_classes
)


# --------------------------------
# Load trained model
# --------------------------------

checkpoint = torch.load(
    "model/oral_sense_mobilenetv3.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


# --------------------------------
# Test
# --------------------------------

correct = 0
total = 0

confusion_matrix = torch.zeros(
    number_of_classes,
    number_of_classes,
    dtype=torch.int64
)


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()


        for true_label, predicted_label in zip(
            labels,
            predictions
        ):

            confusion_matrix[
                true_label,
                predicted_label
            ] += 1


# --------------------------------
# Results
# --------------------------------

accuracy = (
    correct / total
) * 100


print()
print("--------------------------------")
print("TEST RESULTS")
print("--------------------------------")

print(
    "Correct:",
    correct
)

print(
    "Total:",
    total
)

print(
    "Test Accuracy:",
    f"{accuracy:.2f}%"
)


print()
print("Classes:")
print(test_dataset.classes)


print()
print("Confusion Matrix:")
print(confusion_matrix)