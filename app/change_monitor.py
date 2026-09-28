import os
import cv2
import numpy as np


# ============================================================
# ORAL-SENSE EDGE
# LESION CHANGE MONITORING
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

MONITOR_DIR = os.path.join(
    RESULTS_DIR,
    "monitoring"
)

os.makedirs(
    MONITOR_DIR,
    exist_ok=True
)


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    image = cv2.resize(
        image,
        (600, 450)
    )

    return image


# ============================================================
# VISUAL DIFFERENCE
# ============================================================

def calculate_difference(
    image1,
    image2
):

    gray1 = cv2.cvtColor(
        image1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        image2,
        cv2.COLOR_BGR2GRAY
    )

    # Reduce small camera/noise differences
    gray1 = cv2.GaussianBlur(
        gray1,
        (5, 5),
        0
    )

    gray2 = cv2.GaussianBlur(
        gray2,
        (5, 5),
        0
    )

    difference = cv2.absdiff(
        gray1,
        gray2
    )

    # Threshold
    _, threshold = cv2.threshold(
        difference,
        25,
        255,
        cv2.THRESH_BINARY
    )

    # Remove tiny noise
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        kernel
    )

    changed_pixels = np.sum(
        threshold > 0
    )

    total_pixels = threshold.size

    change_percentage = (
        changed_pixels /
        total_pixels
    ) * 100

    return (
        difference,
        threshold,
        change_percentage
    )


# ============================================================
# CREATE VISUAL REPORT
# ============================================================

def create_monitoring_report(
    image1_path,
    image2_path
):

    image1 = prepare_image(
        image1_path
    )

    image2 = prepare_image(
        image2_path
    )

    (
        difference,
        threshold,
        change_percentage
    ) = calculate_difference(
        image1,
        image2
    )


    # --------------------------------------------------------
    # Difference heatmap
    # --------------------------------------------------------

    heatmap = cv2.applyColorMap(
        difference,
        cv2.COLORMAP_JET
    )


    # --------------------------------------------------------
    # Changed-area overlay
    # --------------------------------------------------------

    overlay = image2.copy()

    overlay[
        threshold > 0
    ] = (
        0,
        0,
        255
    )

    overlay = cv2.addWeighted(
        image2,
        0.75,
        overlay,
        0.25,
        0
    )


    # --------------------------------------------------------
    # Side-by-side comparison
    # --------------------------------------------------------

    comparison = np.hstack(
        (
            image1,
            image2
        )
    )


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    comparison_path = os.path.join(
        MONITOR_DIR,
        "visit_comparison.jpg"
    )

    heatmap_path = os.path.join(
        MONITOR_DIR,
        "change_heatmap.jpg"
    )

    overlay_path = os.path.join(
        MONITOR_DIR,
        "changed_area_overlay.jpg"
    )


    cv2.imwrite(
        comparison_path,
        comparison
    )

    cv2.imwrite(
        heatmap_path,
        heatmap
    )

    cv2.imwrite(
        overlay_path,
        overlay
    )


    # --------------------------------------------------------
    # Monitoring interpretation
    # --------------------------------------------------------

    if change_percentage < 5:

        status = "LOW VISUAL CHANGE"

    elif change_percentage < 15:

        status = "MODERATE VISUAL CHANGE"

    else:

        status = "HIGH VISUAL CHANGE"


    return {

        "change_percentage":
            change_percentage,

        "status":
            status,

        "comparison_path":
            comparison_path,

        "heatmap_path":
            heatmap_path,

        "overlay_path":
            overlay_path

    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 55)
    print("ORAL-SENSE EDGE")
    print("LESION CHANGE MONITORING")
    print("=" * 55)

    print()
    print(
        "This module compares two images visually."
    )

    print(
        "It does NOT diagnose cancer progression."
    )

    print()

    visit1 = input(
        "Enter Visit 1 image path: "
    ).strip().strip('"')

    visit2 = input(
        "Enter Visit 2 image path: "
    ).strip().strip('"')


    if not os.path.exists(
        visit1
    ):

        print(
            "Visit 1 image not found."
        )

        exit()


    if not os.path.exists(
        visit2
    ):

        print(
            "Visit 2 image not found."
        )

        exit()


    result = create_monitoring_report(
        visit1,
        visit2
    )


    print()
    print("=" * 55)

    print(
        f"Visual change: "
        f"{result['change_percentage']:.2f}%"
    )

    print(
        f"Monitoring status: "
        f"{result['status']}"
    )

    print()

    print(
        "Comparison saved:"
    )

    print(
        result["comparison_path"]
    )

    print()

    print(
        "Heatmap saved:"
    )

    print(
        result["heatmap_path"]
    )

    print()

    print(
        "Changed-area overlay saved:"
    )

    print(
        result["overlay_path"]
    )

    print("=" * 55)