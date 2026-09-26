import cv2


def check_image_quality(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = gray.mean()

    blur_value = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    if brightness < 50:
        return "TOO DARK", brightness, blur_value

    if brightness > 210:
        return "TOO BRIGHT", brightness, blur_value

    if blur_value < 50:
        return "TOO BLURRY", brightness, blur_value

    return "GOOD IMAGE", brightness, blur_value