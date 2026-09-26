import os
import random
import shutil


# ==============================
# ORAL-SENSE EDGE
# DATASET PREPARATION
# ==============================

# Dataset ka actual location
SOURCE = "Oral Images Dataset/original_data"

# Hamari 2 classes
CLASSES = [
    "benign_lesions",
    "malignant_lesions"
]

# Dataset split
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Same result har baar mile
random.seed(42)


def get_images(folder):

    images = []

    for file in os.listdir(folder):

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            images.append(file)

    return images


print("--------------------------------")
print("ORAL-SENSE EDGE")
print("DATASET PREPARATION")
print("--------------------------------")
print()


# Check source folder
if not os.path.exists(SOURCE):

    print("ERROR:")
    print("Dataset folder nahi mila.")
    print()
    print("Expected location:")
    print(SOURCE)

    exit()


# Process each class
for class_name in CLASSES:

    source_folder = os.path.join(
        SOURCE,
        class_name
    )

    print("Processing:", class_name)

    # Check class folder
    if not os.path.exists(source_folder):

        print("Folder nahi mila:")
        print(source_folder)
        print()

        continue

    # Get images
    images = get_images(source_folder)

    print("Total images:", len(images))

    # Shuffle
    random.shuffle(images)

    total = len(images)

    # Calculate split points
    train_end = int(total * TRAIN_RATIO)

    validation_end = (
        train_end
        + int(total * VALIDATION_RATIO)
    )

    # Create splits
    split_images = {

        "train": images[:train_end],

        "validation": images[
            train_end:validation_end
        ],

        "test": images[
            validation_end:
        ]
    }

    # Copy images
    for split, files in split_images.items():

        destination_folder = os.path.join(
            "dataset",
            split,
            class_name
        )

        # Create destination folder
        os.makedirs(
            destination_folder,
            exist_ok=True
        )

        for file in files:

            source_path = os.path.join(
                source_folder,
                file
            )

            destination_path = os.path.join(
                destination_folder,
                file
            )

            shutil.copy2(
                source_path,
                destination_path
            )

        print(
            split,
            "=",
            len(files),
            "images"
        )

    print()


print("--------------------------------")
print("DATASET PREPARATION COMPLETE!")
print("--------------------------------")
print()
print("Train / Validation / Test")
print("folders ready.")