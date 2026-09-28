import os
import json
import glob


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

HISTORY_FILE = os.path.join(
    RESULTS_DIR,
    "screening_history.json"
)


def get_privacy_status():
    """
    Returns the privacy configuration used by ORAL-SENSE EDGE.
    """

    return {
        "processing": "LOCAL / OFFLINE",
        "cloud_upload": "DISABLED",
        "storage": "LOCAL ONLY",
        "user_control": "ENABLED"
    }


def clear_local_data():
    """
    Deletes locally generated screening data.

    Model files and application source code are NOT deleted.
    """

    deleted_files = []

    # Delete screening history
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
            deleted_files.append(HISTORY_FILE)
        except Exception:
            pass

    # Delete generated result images
    patterns = [
        "current_image.jpg",
        "gradcam_result.jpg",
        "screening_gradcam.jpg",
        "classic_gradcam_*.jpg",
        "screening_gradcam_*.jpg",
        "monitoring/*.jpg",
        "monitoring/*.png"
    ]

    for pattern in patterns:

        full_pattern = os.path.join(
            RESULTS_DIR,
            pattern
        )

        for file_path in glob.glob(
            full_pattern
        ):

            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted_files.append(file_path)

            except Exception:
                pass

    return deleted_files


def privacy_summary():
    """
    Human-readable privacy summary.
    """

    return (
        "ORAL-SENSE EDGE processes screening images "
        "locally on the device.\n\n"
        "No automatic cloud upload is performed.\n"
        "Screening history is stored locally.\n"
        "Generated local data can be cleared by the user.\n\n"
        "This privacy design is intended for the current "
        "prototype and does not guarantee security against "
        "all operating-system or device-level threats."
    )


if __name__ == "__main__":

    print("=" * 45)
    print("ORAL-SENSE EDGE - PRIVACY CENTER")
    print("=" * 45)

    status = get_privacy_status()

    for key, value in status.items():
        print(
            f"{key.upper():20}: {value}"
        )

    print("\n")
    print(privacy_summary())