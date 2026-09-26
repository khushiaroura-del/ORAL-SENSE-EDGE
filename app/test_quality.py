import cv2
from quality_check import check_image_quality

image = cv2.imread("results/current_image.jpg")

if image is None:
    print("Image nahi mili.")
else:
    result, brightness, blur = check_image_quality(image)

    print("--------------------------------")
    print("ORAL-SENSE EDGE")
    print("IMAGE QUALITY CHECK")
    print("--------------------------------")

    print("Result:", result)
    print("Brightness:", round(brightness, 2))
    print("Sharpness:", round(blur, 2))