import os
import json
from datetime import datetime


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

HISTORY_FILE = os.path.join(
    BASE_DIR,
    "results",
    "screening_history.json"
)


def load_history():

    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:
        return []


def save_screening(
    image_name,
    predicted_class,
    probability,
    quality
):

    history = load_history()

    record = {
        "date_time": datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        ),

        "image_name": image_name,

        "predicted_class": predicted_class,

        "model_score": round(
            probability,
            2
        ),

        "image_quality": quality
    }

    history.append(
        record
    )

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    return record


def get_history():

    return load_history()


if __name__ == "__main__":

    print("=" * 55)
    print("ORAL-SENSE EDGE")
    print("SCREENING HISTORY")
    print("=" * 55)

    record = save_screening(
        "test_image.jpg",
        "benign_lesions",
        97.48,
        "GOOD IMAGE"
    )

    print("\nNew screening saved:")
    print(record)

    print("\nTotal screening records:")

    history = get_history()

    print(
        len(history)
    )