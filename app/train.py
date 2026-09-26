import os
import copy
import torch

from torchvision import datasets, transforms
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

from torch import nn, optim
from torch.utils.data import DataLoader


# ==========================================
# ORAL-SENSE EDGE
# MobileNetV3 Training
# ==========================================

print("--------------------------------")
print("ORAL-SENSE EDGE")
print("MobileNetV3 TRAINING")
print("--------------------------------")


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# DATA PATHS
# ==========================================

TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"


# ==========================================
# IMAGE TRANSFORMS
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


validation_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# LOAD DATASET
# ==========================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    VALIDATION_DIR,
    transform=validation_transform
)


print()
print("Classes:")
print(train_dataset.classes)

print()
print("Training images:",
      len(train_dataset))

print("Validation images:",
      len(validation_dataset))


# ==========================================
# DATA LOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# ==========================================
# LOAD PRETRAINED MOBILENETV3
# ==========================================

weights = MobileNet_V3_Small_Weights.DEFAULT

model = mobilenet_v3_small(
    weights=weights
)


# ==========================================
# REPLACE CLASSIFIER
# ==========================================

number_of_classes = len(
    train_dataset.classes
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    number_of_classes
)


model = model.to(device)


# ==========================================
# LOSS + OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


# ==========================================
# TRAINING SETTINGS
# ==========================================

epochs = 10

best_validation_accuracy = 0.0

best_model_weights = copy.deepcopy(
    model.state_dict()
)


# ==========================================
# TRAINING LOOP
# ==========================================

for epoch in range(epochs):

    print()
    print(
        f"Epoch {epoch + 1}/{epochs}"
    )

    print("----------------------------")


    # ------------------------------
    # TRAIN
    # ------------------------------

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        _, predictions = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()


    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    ) * 100


    # ------------------------------
    # VALIDATION
    # ------------------------------

    model.eval()

    validation_correct = 0

    validation_total = 0


    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(images)


            _, predictions = torch.max(
                outputs,
                1
            )


            validation_total += (
                labels.size(0)
            )


            validation_correct += (
                predictions == labels
            ).sum().item()


    validation_accuracy = (
        validation_correct
        / validation_total
    ) * 100


    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Validation Accuracy: "
        f"{validation_accuracy:.2f}%"
    )


    # ------------------------------
    # SAVE BEST MODEL
    # ------------------------------

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = (
            validation_accuracy
        )

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        print(
            "New best model found!"
        )


# ==========================================
# SAVE MODEL
# ==========================================

model.load_state_dict(
    best_model_weights
)


os.makedirs(
    "model",
    exist_ok=True
)


model_path = (
    "model/oral_sense_mobilenetv3.pth"
)


torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "classes":
            train_dataset.classes
    },
    model_path
)


print()
print("--------------------------------")
print("TRAINING COMPLETE")
print("--------------------------------")

print(
    "Best Validation Accuracy:",
    f"{best_validation_accuracy:.2f}%"
)

print()
print(
    "Model saved at:"
)

print(model_path)