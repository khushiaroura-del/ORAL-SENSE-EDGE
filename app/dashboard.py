import os
import time
import cv2
import torch
import numpy as np
import customtkinter as ctk

from PIL import Image
from tkinter import filedialog, messagebox

from torchvision import models, transforms


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "oral_sense_mobilenetv3.pth"
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")

CURRENT_IMAGE = os.path.join(
    RESULTS_DIR,
    "current_image.jpg"
)

GRADCAM_IMAGE = os.path.join(
    RESULTS_DIR,
    "dashboard_gradcam.jpg"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# 3. LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# Get class names from saved model
if "classes" in checkpoint:
    CLASS_NAMES = checkpoint["classes"]
else:
    CLASS_NAMES = [
        "benign_lesions",
        "malignant_lesions"
    ]


# Create MobileNetV3 Small
model = models.mobilenet_v3_small(
    weights=None
)

# Change final classifier
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features,
    len(CLASS_NAMES)
)


# Load weights
if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
elif "state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["state_dict"]
    )
else:
    model.load_state_dict(checkpoint)


model = model.to(DEVICE)
model.eval()


# ============================================================
# 4. IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 5. IMAGE QUALITY
# ============================================================

def check_image_quality(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    brightness = float(gray.mean())

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    if brightness < 50:
        quality = "TOO DARK"

    elif brightness > 210:
        quality = "TOO BRIGHT"

    elif sharpness < 50:
        quality = "TOO BLURRY"

    else:
        quality = "GOOD IMAGE"

    return quality, brightness, sharpness


# ============================================================
# 6. MODEL PREDICTION
# ============================================================

def predict_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    start_time = time.perf_counter()

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        probability, predicted_index = torch.max(
            probabilities,
            1
        )

    end_time = time.perf_counter()

    inference_time = (
        end_time - start_time
    ) * 1000

    predicted_index = int(
        predicted_index.item()
    )

    probability = float(
        probability.item() * 100
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    return (
        predicted_class,
        probability,
        inference_time,
        predicted_index
    )


# ============================================================
# 7. GRAD-CAM
# ============================================================

def generate_gradcam(
    image_path,
    predicted_index
):

    original = cv2.imread(
        image_path
    )

    if original is None:
        return False

    rgb = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        rgb
    )

    input_tensor = transform(
        pil_image
    ).unsqueeze(0).to(DEVICE)

    activations = []
    gradients = []

    target_layer = model.features[-1]

    def forward_hook(
        module,
        input,
        output
    ):
        activations.append(
            output.detach()
        )

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):
        gradients.append(
            grad_output[0].detach()
        )

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:

        model.zero_grad()

        output = model(
            input_tensor
        )

        score = output[
            0,
            predicted_index
        ]

        score.backward()

        if len(activations) == 0:
            return False

        if len(gradients) == 0:
            return False

        activation = activations[0]

        gradient = gradients[0]

        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activation
        ).sum(
            dim=1
        ).squeeze()

        cam = torch.relu(
            cam
        )

        cam = cam.cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        cam = cv2.resize(
            cam,
            (
                original.shape[1],
                original.shape[0]
            )
        )

        heatmap = np.uint8(
            255 * cam
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            original,
            0.55,
            heatmap,
            0.45,
            0
        )

        cv2.imwrite(
            GRADCAM_IMAGE,
            overlay
        )

        return True

    except Exception as e:

        print(
            "Grad-CAM Error:",
            e
        )

        return False

    finally:

        forward_handle.remove()
        backward_handle.remove()


# ============================================================
# 8. DASHBOARD
# ============================================================

class OralSenseDashboard(ctk.CTk):

    def __init__(self):

        super().__init__()

        self.title(
            "ORAL-SENSE EDGE | AI Screening"
        )

        self.geometry(
            "1450x900"
        )

        self.minsize(
            1100,
            700
        )

        ctk.set_appearance_mode(
            "dark"
        )

        ctk.set_default_color_theme(
            "blue"
        )

        self.build_ui()


    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        header = ctk.CTkFrame(
            self,
            corner_radius=0
        )

        header.pack(
            fill="x"
        )

        title = ctk.CTkLabel(
            header,
            text="◉ ORAL-SENSE EDGE",
            font=(
                "Arial",
                28,
                "bold"
            )
        )

        title.pack(
            side="left",
            padx=25,
            pady=(18, 2)
        )

        subtitle = ctk.CTkLabel(
            header,
            text="UNIFIED AI SCREENING  •  LOCAL AI  •  PRIVACY-FIRST",
            font=(
                "Arial",
                13
            )
        )

        subtitle.pack(
            side="left",
            padx=10,
            pady=(20, 2)
        )


        # ----------------------------------------------------
        # MAIN
        # ----------------------------------------------------

        main = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )

        main.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=15
        )

        main.grid_columnconfigure(
            0,
            weight=1
        )

        main.grid_columnconfigure(
            1,
            weight=1
        )

        main.grid_rowconfigure(
            0,
            weight=1
        )


        # ====================================================
        # LEFT SIDE
        # ====================================================

        left = ctk.CTkFrame(
            main,
            corner_radius=18
        )

        left.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 10)
        )


        left_title = ctk.CTkLabel(
            left,
            text="ORIGINAL IMAGE",
            font=(
                "Arial",
                18,
                "bold"
            )
        )

        left_title.pack(
            pady=(20, 10)
        )


        self.original_image_label = ctk.CTkLabel(
            left,
            text="No image selected",
            width=500,
            height=400
        )

        self.original_image_label.pack(
            padx=20,
            pady=10,
            expand=True
        )


        # Buttons
        button_frame = ctk.CTkFrame(
            left,
            fg_color="transparent"
        )

        button_frame.pack(
            pady=15
        )


        capture_button = ctk.CTkButton(
            button_frame,
            text="📷 CAPTURE",
            width=150,
            height=40,
            command=self.capture_image
        )

        capture_button.grid(
            row=0,
            column=0,
            padx=5
        )


        upload_button = ctk.CTkButton(
            button_frame,
            text="📁 UPLOAD",
            width=150,
            height=40,
            command=self.upload_image
        )

        upload_button.grid(
            row=0,
            column=1,
            padx=5
        )


        run_button = ctk.CTkButton(
            left,
            text="▶  RUN AI SCREENING",
            width=330,
            height=48,
            font=(
                "Arial",
                15,
                "bold"
            ),
            command=self.run_analysis
        )

        run_button.pack(
            pady=(5, 20)
        )


        # ====================================================
        # RIGHT SIDE
        # ====================================================

        right = ctk.CTkFrame(
            main,
            corner_radius=18
        )

        right.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(10, 0)
        )


        result_title = ctk.CTkLabel(
            right,
            text="AI RESULT",
            font=(
                "Arial",
                18,
                "bold"
            )
        )

        result_title.pack(
            pady=(18, 5)
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        self.prediction_label = ctk.CTkLabel(
            right,
            text="Waiting for analysis",
            font=(
                "Arial",
                26,
                "bold"
            )
        )

        self.prediction_label.pack(
            pady=(5, 12)
        )


        # ====================================================
        # MODEL SCORE - IMPORTANT
        # ====================================================

        score_frame = ctk.CTkFrame(
            right,
            corner_radius=14
        )

        score_frame.pack(
            fill="x",
            padx=20,
            pady=5
        )


        score_title = ctk.CTkLabel(
            score_frame,
            text="MODEL SCORE",
            font=(
                "Arial",
                13,
                "bold"
            )
        )

        score_title.pack(
            pady=(10, 0)
        )


        self.score_box = ctk.CTkLabel(
            score_frame,
            text="--",
            font=(
                "Arial",
                32,
                "bold"
            )
        )

        self.score_box.pack(
            pady=(0, 10)
        )


        # ====================================================
        # LIVE DATA
        # ====================================================

        live_title = ctk.CTkLabel(
            right,
            text="LIVE MODEL DATA",
            font=(
                "Arial",
                15,
                "bold"
            )
        )

        live_title.pack(
            pady=(12, 5)
        )


        data_frame = ctk.CTkFrame(
            right,
            fg_color="transparent"
        )

        data_frame.pack(
            fill="x",
            padx=15
        )

        data_frame.grid_columnconfigure(
            0,
            weight=1
        )

        data_frame.grid_columnconfigure(
            1,
            weight=1
        )


        self.quality_box = self.create_data_box(
            data_frame,
            "IMAGE QUALITY",
            "--",
            0,
            0
        )


        self.brightness_box = self.create_data_box(
            data_frame,
            "BRIGHTNESS",
            "--",
            0,
            1
        )


        self.sharpness_box = self.create_data_box(
            data_frame,
            "SHARPNESS",
            "--",
            1,
            0
        )


        self.inference_box = self.create_data_box(
            data_frame,
            "INFERENCE TIME",
            "--",
            1,
            1
        )


        self.model_box = self.create_data_box(
            data_frame,
            "MODEL",
            "MobileNetV3",
            2,
            0
        )


        self.device_box = self.create_data_box(
            data_frame,
            "DEVICE",
            "CPU",
            2,
            1
        )


        # ====================================================
        # GRAD-CAM
        # ====================================================

        grad_title = ctk.CTkLabel(
            right,
            text="AI ATTENTION MAP • GRAD-CAM",
            font=(
                "Arial",
                14,
                "bold"
            )
        )

        grad_title.pack(
            pady=(12, 3)
        )


        self.gradcam_label = ctk.CTkLabel(
            right,
            text="Run AI Screening to generate",
            width=420,
            height=180
        )

        self.gradcam_label.pack(
            padx=20,
            pady=5
        )


        self.gradcam_status = ctk.CTkLabel(
            right,
            text="● Waiting",
            font=(
                "Arial",
                11
            )
        )

        self.gradcam_status.pack(
            pady=2
        )


        # ====================================================
        # STATUS
        # ====================================================

        self.live_status = ctk.CTkLabel(
            self,
            text="Ready • Select an image to begin",
            font=(
                "Arial",
                12
            )
        )

        self.live_status.pack(
            pady=(0, 5)
        )


        disclaimer = ctk.CTkLabel(
            self,
            text=(
                "Screening assistance only — not a medical diagnosis. "
                "Model score is an AI inference score, not clinical certainty."
            ),
            font=(
                "Arial",
                11
            )
        )

        disclaimer.pack(
            pady=(0, 10)
        )


    # ========================================================
    # DATA BOX
    # ========================================================

    def create_data_box(
        self,
        parent,
        title,
        value,
        row,
        column
    ):

        frame = ctk.CTkFrame(
            parent,
            corner_radius=10
        )

        frame.grid(
            row=row,
            column=column,
            sticky="ew",
            padx=5,
            pady=4
        )


        title_label = ctk.CTkLabel(
            frame,
            text=title,
            font=(
                "Arial",
                10,
                "bold"
            )
        )

        title_label.pack(
            pady=(6, 0)
        )


        value_label = ctk.CTkLabel(
            frame,
            text=value,
            font=(
                "Arial",
                15,
                "bold"
            )
        )

        value_label.pack(
            pady=(0, 7)
        )


        return value_label


    # ========================================================
    # DISPLAY IMAGE
    # ========================================================

    def display_image(
        self,
        image_path,
        label,
        width,
        height
    ):

        if not os.path.exists(
            image_path
        ):
            return


        image = Image.open(
            image_path
        ).convert("RGB")


        image.thumbnail(
            (
                width,
                height
            )
        )


        ctk_image = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=image.size
        )


        label.configure(
            image=ctk_image,
            text=""
        )


        label.image = ctk_image


    # ========================================================
    # CAPTURE IMAGE
    # ========================================================

    def capture_image(self):

        camera = cv2.VideoCapture(
            0
        )


        if not camera.isOpened():

            messagebox.showerror(
                "Camera Error",
                "Camera could not be opened."
            )

            return


        messagebox.showinfo(
            "Camera",
            "Press SPACE to capture.\nPress Q to cancel."
        )


        captured = False


        while True:

            ret, frame = camera.read()

            if not ret:
                break


            cv2.imshow(
                "ORAL-SENSE EDGE CAMERA",
                frame
            )


            key = cv2.waitKey(
                1
            ) & 0xFF


            if key == ord(" "):

                cv2.imwrite(
                    CURRENT_IMAGE,
                    frame
                )

                captured = True

                break


            if key == ord("q"):

                break


        camera.release()

        cv2.destroyAllWindows()


        if captured:

            self.display_image(
                CURRENT_IMAGE,
                self.original_image_label,
                500,
                400
            )

            self.live_status.configure(
                text="Image captured successfully • Ready for AI screening"
            )


    # ========================================================
    # UPLOAD IMAGE
    # ========================================================

    def upload_image(self):

        file_path = filedialog.askopenfilename(
            title="Select Oral Image",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp"
                )
            ]
        )


        if not file_path:
            return


        image = cv2.imread(
            file_path
        )


        if image is None:

            messagebox.showerror(
                "Image Error",
                "Could not read this image."
            )

            return


        cv2.imwrite(
            CURRENT_IMAGE,
            image
        )


        self.display_image(
            CURRENT_IMAGE,
            self.original_image_label,
            500,
            400
        )


        self.live_status.configure(
            text="Image uploaded successfully • Ready for AI screening"
        )


    # ========================================================
    # RESET
    # ========================================================

    def reset_results(self):

        self.prediction_label.configure(
            text="Waiting for analysis"
        )

        self.score_box.configure(
            text="--"
        )

        self.quality_box.configure(
            text="--"
        )

        self.brightness_box.configure(
            text="--"
        )

        self.sharpness_box.configure(
            text="--"
        )

        self.inference_box.configure(
            text="--"
        )

        self.gradcam_label.configure(
            image=None,
            text="Run AI Screening to generate"
        )

        self.gradcam_status.configure(
            text="● Waiting"
        )


    # ========================================================
    # RUN AI ANALYSIS
    # ========================================================

    def run_analysis(self):

        if not os.path.exists(
            CURRENT_IMAGE
        ):

            messagebox.showwarning(
                "No Image",
                "Please capture or upload an image first."
            )

            return


        self.live_status.configure(
            text="AI analysis running..."
        )

        self.update()


        total_start = time.perf_counter()


        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(
            CURRENT_IMAGE
        )


        if image is None:

            messagebox.showerror(
                "Error",
                "Could not read current image."
            )

            return


        # ----------------------------------------------------
        # Quality
        # ----------------------------------------------------

        quality, brightness, sharpness = check_image_quality(
            image
        )


        self.quality_box.configure(
            text=quality
        )

        self.brightness_box.configure(
            text=f"{brightness:.2f}"
        )

        self.sharpness_box.configure(
            text=f"{sharpness:.2f}"
        )


        # ----------------------------------------------------
        # WARNING ONLY
        # ----------------------------------------------------

        if quality != "GOOD IMAGE":

            self.live_status.configure(
                text=(
                    f"Warning: {quality} • "
                    "AI will still analyze the image"
                )
            )

            self.update()


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        try:

            (
                predicted_class,
                probability,
                inference_time,
                predicted_index

            ) = predict_image(
                CURRENT_IMAGE
            )


        except Exception as e:

            messagebox.showerror(
                "Model Error",
                str(e)
            )

            self.live_status.configure(
                text="Model inference failed"
            )

            return


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        self.prediction_label.configure(
            text=predicted_class
        )


        # ====================================================
        # IMPORTANT MODEL SCORE
        # ====================================================

        self.score_box.configure(
            text=f"{probability:.2f}%"
        )


        self.inference_box.configure(
            text=f"{inference_time:.1f} ms"
        )


        self.model_box.configure(
            text="MobileNetV3"
        )


        self.device_box.configure(
            text="CPU"
        )


        self.update()


        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        self.gradcam_status.configure(
            text="● Generating attention map..."
        )

        self.update()


        gradcam_success = generate_gradcam(
            CURRENT_IMAGE,
            predicted_index
        )


        if gradcam_success:

            self.display_image(
                GRADCAM_IMAGE,
                self.gradcam_label,
                420,
                180
            )

            self.gradcam_status.configure(
                text="● AI attention map generated"
            )

        else:

            self.gradcam_label.configure(
                image=None,
                text="Grad-CAM unavailable"
            )

            self.gradcam_status.configure(
                text="● Grad-CAM failed"
            )


        # ----------------------------------------------------
        # TOTAL TIME
        # ----------------------------------------------------

        total_time = (
            time.perf_counter()
            - total_start
        ) * 1000


        self.live_status.configure(
            text=(
                f"Analysis complete • "
                f"Total processing time: {total_time:.1f} ms"
            )
        )


        print("\n==============================")
        print("ORAL-SENSE EDGE AI RESULT")
        print("==============================")
        print(
            "Prediction:",
            predicted_class
        )
        print(
            "Model Score:",
            f"{probability:.2f}%"
        )
        print(
            "Image Quality:",
            quality
        )
        print(
            "Brightness:",
            f"{brightness:.2f}"
        )
        print(
            "Sharpness:",
            f"{sharpness:.2f}"
        )
        print(
            "Inference Time:",
            f"{inference_time:.1f} ms"
        )
        print(
            "Grad-CAM:",
            "Generated" if gradcam_success else "Failed"
        )
        print("==============================\n")


# ============================================================
# 9. START APPLICATION
# ============================================================

if __name__ == "__main__":

    app = OralSenseDashboard()

    app.mainloop()