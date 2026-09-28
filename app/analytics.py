import os
from screening_history import get_history


def get_analytics():

    history = get_history()

    total = len(history)

    benign = 0
    malignant = 0
    good_images = 0
    follow_up = 0

    for record in history:

        result = record.get(
            "predicted_class",
            ""
        ).lower()

        quality = record.get(
            "image_quality",
            ""
        ).upper()

        if "benign" in result:
            benign += 1

        if "malignant" in result:
            malignant += 1
            follow_up += 1

        if quality == "GOOD IMAGE":
            good_images += 1

    return {
        "total": total,
        "benign": benign,
        "malignant": malignant,
        "good_images": good_images,
        "follow_up": follow_up
    }


if __name__ == "__main__":

    print("=" * 55)
    print("ORAL-SENSE EDGE")
    print("SCREENING ANALYTICS")
    print("=" * 55)

    data = get_analytics()

    print("\nTotal Screenings:", data["total"])
    print("Benign Results:", data["benign"])
    print("Malignant Results:", data["malignant"])
    print("Good Images:", data["good_images"])
    print("Follow-ups:", data["follow_up"])