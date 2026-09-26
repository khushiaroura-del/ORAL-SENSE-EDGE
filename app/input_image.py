import cv2
import tkinter as tk
from tkinter import filedialog
import os

SAVE_PATH = "results/current_image.jpg"


def capture_from_camera():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Camera open nahi ho raha.")
        return

    print("Camera started.")
    print("S = Photo lena")
    print("Q = Camera band karna")

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera se image nahi mil rahi.")
            break

        cv2.imshow("ORAL-SENSE EDGE - Webcam", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):

            os.makedirs("results", exist_ok=True)

            cv2.imwrite(SAVE_PATH, frame)

            print("Photo successfully saved!")
            print("Location:", SAVE_PATH)

            break

        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


def upload_image():

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png"),
            ("All Files", "*.*")
        ]
    )

    root.destroy()

    if not file_path:
        print("Koi image select nahi ki gayi.")
        return

    image = cv2.imread(file_path)

    if image is None:
        print("Image open nahi ho paayi.")
        return

    os.makedirs("results", exist_ok=True)

    cv2.imwrite(SAVE_PATH, image)

    print("Image successfully uploaded!")
    print("Location:", SAVE_PATH)


print("==============================")
print("       ORAL-SENSE EDGE")
print("==============================")

print()
print("1 = Use Webcam")
print("2 = Upload Existing Image")

choice = input("Choose option (1/2): ")

if choice == "1":

    capture_from_camera()

elif choice == "2":

    upload_image()

else:

    print("Invalid option.")