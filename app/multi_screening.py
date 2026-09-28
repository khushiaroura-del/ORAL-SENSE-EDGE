import os
import time
import cv2
import torch
import torch.nn as nn
import customtkinter as ctk
import numpy as np

from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from torchvision import models, transforms


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "oral_sense_mobilenetv3.pth"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "multi_screening"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cpu")


# =========================================================
# THEME
# =========================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG = "#080B12"
CARD = "#111722"
CARD_2 = "#151C28"
BORDER = "#263244"
TEXT = "#F4F7FB"
MUTED = "#8C98AA"
CYAN = "#36D9FF"
GREEN = "#35E69A"
ORANGE = "#FFB84D"
RED = "#FF5C70"
PURPLE = "#9B7BFF"


# =========================================================
# TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# IMAGE QUALITY
# =========================================================

def check_image_quality(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    brightness = gray.mean()

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    warnings = []

    if brightness < 50:
        warnings.append("TOO DARK")

    elif brightness > 210:
        warnings.append("TOO BRIGHT")

    if sharpness < 50:
        warnings.append("TOO BLURRY")

    if not warnings:
        quality = "GOOD IMAGE"
    else:
        quality = " + ".join(warnings)

    return quality, brightness, sharpness


# =========================================================
# MODEL
# =========================================================

def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    classes = checkpoint.get(
        "classes",
        ["benign_lesions", "malignant_lesions"]
    )

    model = models.mobilenet_v3_small(
        weights=None
    )

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(classes)
    )

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

    model.to(DEVICE)
    model.eval()

    return model, classes


# =========================================================
# PREDICTION
# =========================================================

def predict_image(model, classes, image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    tensor = transform(rgb)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(DEVICE)

    start = time.time()

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        probability, prediction = torch.max(
            probabilities,
            1
        )

    inference_time = (
        time.time() - start
    ) * 1000

    predicted_class = classes[
        prediction.item()
    ]

    score = probability.item() * 100

    return (
        predicted_class,
        score,
        inference_time
    )


# =========================================================
# GRAD CAM
# =========================================================

class GradCAM:

    def __init__(self, model):

        self.model = model
        self.activations = None
        self.gradients = None

        self.target_layer = model.features[-1]

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )

    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    def generate(self, image):

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        tensor = transform(rgb)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(DEVICE)

        self.model.zero_grad()

        output = self.model(tensor)

        prediction = output.argmax(
            dim=1
        )

        score = output[
            0,
            prediction
        ]

        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(dim=1).squeeze()

        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()

        if cam.max() != 0:

            cam = (
                cam - cam.min()
            ) / (
                cam.max() - cam.min()
            )

        cam = cv2.resize(
            cam,
            (
                image.shape[1],
                image.shape[0]
            )
        )

        heatmap = np.uint8(
            cam * 255
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            image,
            0.55,
            heatmap,
            0.45,
            0
        )

        return overlay

    def close(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# =========================================================
# DASHBOARD
# =========================================================

class MultiScreeningApp(ctk.CTk):

    def __init__(self):

        super().__init__()

        self.title(
            "ORAL-SENSE EDGE • Multi-Image AI Screening"
        )

        self.geometry("1380x850")
        self.minsize(1100, 700)

        self.configure(
            fg_color=BG
        )

        self.model, self.classes = load_model()

        self.gradcam = GradCAM(
            self.model
        )

        self.image_paths = []

        self.analyzed_count = 0

        self.create_dashboard()

    # =====================================================
    # HEADER
    # =====================================================

    def create_dashboard(self):

        # HEADER
        header = ctk.CTkFrame(
            self,
            fg_color=BG
        )

        header.pack(
            fill="x",
            padx=28,
            pady=(22, 8)
        )

        left = ctk.CTkFrame(
            header,
            fg_color="transparent"
        )

        left.pack(
            side="left"
        )

        logo = ctk.CTkLabel(
            left,
            text="◉",
            text_color=CYAN,
            font=("Arial", 30, "bold")
        )

        logo.pack(
            side="left",
            padx=(0, 10)
        )

        title_frame = ctk.CTkFrame(
            left,
            fg_color="transparent"
        )

        title_frame.pack(
            side="left"
        )

        ctk.CTkLabel(
            title_frame,
            text="ORAL-SENSE EDGE",
            text_color=TEXT,
            font=("Arial", 27, "bold")
        ).pack(
            anchor="w"
        )

        ctk.CTkLabel(
            title_frame,
            text="MULTI-IMAGE AI SCREENING  •  LOCAL INFERENCE",
            text_color=MUTED,
            font=("Arial", 11, "bold")
        ).pack(
            anchor="w"
        )

        # STATUS
        status = ctk.CTkFrame(
            header,
            fg_color=CARD,
            corner_radius=18,
            border_width=1,
            border_color=BORDER
        )

        status.pack(
            side="right",
            padx=5
        )

        ctk.CTkLabel(
            status,
            text="●",
            text_color=GREEN,
            font=("Arial", 14, "bold")
        ).pack(
            side="left",
            padx=(14, 5),
            pady=10
        )

        ctk.CTkLabel(
            status,
            text="AI ENGINE ONLINE",
            text_color=TEXT,
            font=("Arial", 11, "bold")
        ).pack(
            side="left",
            padx=(0, 14)
        )

        # DIVIDER
        ctk.CTkFrame(
            self,
            height=1,
            fg_color=BORDER
        ).pack(
            fill="x",
            padx=28,
            pady=(5, 16)
        )

        # STATS
        self.create_stats()

        # CONTROLS
        self.create_controls()

        # RESULTS
        self.scroll_frame = ctk.CTkScrollableFrame(
            self,
            label_text="  SCREENING WORKSPACE",
            label_text_color=TEXT,
            label_font=("Arial", 14, "bold"),
            fg_color=BG,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=CYAN
        )

        self.scroll_frame.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=(5, 20)
        )

        self.show_empty_state()

        # FOOTER
        footer = ctk.CTkLabel(
            self,
            text=(
                "LOCAL AI   •   PRIVACY-FIRST   •   "
                "OFFLINE INFERENCE   •   SCREENING ASSISTANCE ONLY"
            ),
            text_color="#657286",
            font=("Arial", 10, "bold")
        )

        footer.pack(
            pady=(0, 12)
        )

    # =====================================================
    # STATS
    # =====================================================

    def create_stats(self):

        stats = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )

        stats.pack(
            fill="x",
            padx=25,
            pady=(0, 12)
        )

        self.stat_images = self.make_stat(
            stats,
            "IMAGES",
            "0",
            CYAN
        )

        self.stat_analyzed = self.make_stat(
            stats,
            "ANALYZED",
            "0",
            GREEN
        )

        self.stat_model = self.make_stat(
            stats,
            "MODEL",
            "MobileNetV3",
            PURPLE
        )

        self.stat_device = self.make_stat(
            stats,
            "DEVICE",
            "CPU",
            ORANGE
        )

    def make_stat(
        self,
        parent,
        title,
        value,
        accent
    ):

        card = ctk.CTkFrame(
            parent,
            fg_color=CARD,
            corner_radius=16,
            border_width=1,
            border_color=BORDER
        )

        card.pack(
            side="left",
            fill="x",
            expand=True,
            padx=5
        )

        ctk.CTkLabel(
            card,
            text=title,
            text_color=MUTED,
            font=("Arial", 10, "bold")
        ).pack(
            anchor="w",
            padx=16,
            pady=(12, 0)
        )

        label = ctk.CTkLabel(
            card,
            text=value,
            text_color=accent,
            font=("Arial", 19, "bold")
        )

        label.pack(
            anchor="w",
            padx=16,
            pady=(2, 12)
        )

        return label

    # =====================================================
    # CONTROLS
    # =====================================================

    def create_controls(self):

        controls = ctk.CTkFrame(
            self,
            fg_color=CARD,
            corner_radius=18,
            border_width=1,
            border_color=BORDER
        )

        controls.pack(
            fill="x",
            padx=30,
            pady=(0, 14)
        )

        self.add_button = ctk.CTkButton(
            controls,
            text="＋  ADD IMAGES",
            height=42,
            width=150,
            corner_radius=12,
            fg_color=CARD_2,
            hover_color="#202B3D",
            border_width=1,
            border_color=BORDER,
            text_color=TEXT,
            font=("Arial", 12, "bold"),
            command=self.add_images
        )

        self.add_button.pack(
            side="left",
            padx=(12, 8),
            pady=12
        )

        self.analyze_button = ctk.CTkButton(
            controls,
            text="▶  ANALYZE ALL",
            height=42,
            width=160,
            corner_radius=12,
            fg_color=CYAN,
            hover_color="#20BFE5",
            text_color="#061018",
            font=("Arial", 12, "bold"),
            command=self.analyze_all
        )

        self.analyze_button.pack(
            side="left",
            padx=8,
            pady=12
        )

        self.clear_button = ctk.CTkButton(
            controls,
            text="↻  CLEAR",
            height=42,
            width=110,
            corner_radius=12,
            fg_color="#21151A",
            hover_color="#351A21",
            text_color=RED,
            border_width=1,
            border_color="#5B2731",
            font=("Arial", 12, "bold"),
            command=self.clear_all
        )

        self.clear_button.pack(
            side="left",
            padx=8,
            pady=12
        )

        self.count_label = ctk.CTkLabel(
            controls,
            text="0 images ready",
            text_color=MUTED,
            font=("Arial", 11, "bold")
        )

        self.count_label.pack(
            side="right",
            padx=18
        )

    # =====================================================
    # EMPTY STATE
    # =====================================================

    def show_empty_state(self):

        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        empty = ctk.CTkFrame(
            self.scroll_frame,
            fg_color=CARD,
            corner_radius=22,
            border_width=1,
            border_color=BORDER
        )

        empty.pack(
            fill="x",
            padx=10,
            pady=30
        )

        ctk.CTkLabel(
            empty,
            text="◎",
            text_color=CYAN,
            font=("Arial", 50)
        ).pack(
            pady=(35, 5)
        )

        ctk.CTkLabel(
            empty,
            text="READY FOR SCREENING",
            text_color=TEXT,
            font=("Arial", 19, "bold")
        ).pack()

        ctk.CTkLabel(
            empty,
            text=(
                "Add one or more oral images to begin "
                "local AI analysis."
            ),
            text_color=MUTED,
            font=("Arial", 12)
        ).pack(
            pady=(5, 35)
        )

    # =====================================================
    # ADD IMAGES
    # =====================================================

    def add_images(self):

        files = filedialog.askopenfilenames(
            title="Select Oral Images",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp *.webp"
                )
            ]
        )

        if not files:
            return

        for path in files:

            if path not in self.image_paths:

                self.image_paths.append(path)

        self.update_stats()
        self.show_selected_images()

    # =====================================================
    # SELECTED IMAGES
    # =====================================================

    def show_selected_images(self):

        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        for index, path in enumerate(
            self.image_paths
        ):

            card = ctk.CTkFrame(
                self.scroll_frame,
                fg_color=CARD,
                corner_radius=18,
                border_width=1,
                border_color=BORDER
            )

            card.pack(
                fill="x",
                padx=8,
                pady=6
            )

            try:

                image = Image.open(path)
                image.thumbnail((120, 90))

                preview = ImageTk.PhotoImage(image)

                label = ctk.CTkLabel(
                    card,
                    text="",
                    image=preview
                )

                label.image = preview

                label.pack(
                    side="left",
                    padx=15,
                    pady=12
                )

            except Exception:
                pass

            info = ctk.CTkFrame(
                card,
                fg_color="transparent"
            )

            info.pack(
                side="left",
                fill="x",
                expand=True,
                padx=10
            )

            ctk.CTkLabel(
                info,
                text=f"IMAGE {index + 1:02d}",
                text_color=CYAN,
                font=("Arial", 11, "bold")
            ).pack(
                anchor="w"
            )

            ctk.CTkLabel(
                info,
                text=os.path.basename(path),
                text_color=TEXT,
                font=("Arial", 13, "bold")
            ).pack(
                anchor="w",
                pady=(2, 3)
            )

            ctk.CTkLabel(
                info,
                text="READY FOR AI ANALYSIS",
                text_color=MUTED,
                font=("Arial", 10)
            ).pack(
                anchor="w"
            )

    # =====================================================
    # ANALYZE ALL
    # =====================================================

    def analyze_all(self):

        if not self.image_paths:

            messagebox.showwarning(
                "No Images",
                "Please add at least one image first."
            )

            return

        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        self.analyzed_count = 0

        self.count_label.configure(
            text=f"ANALYZING {len(self.image_paths)} IMAGES..."
        )

        self.analyze_button.configure(
            state="disabled",
            text="⏳  ANALYZING..."
        )

        self.update()

        for index, path in enumerate(
            self.image_paths
        ):

            self.analyze_single(
                index,
                path
            )

            self.analyzed_count += 1

            self.stat_analyzed.configure(
                text=str(
                    self.analyzed_count
                )
            )

            self.update()

        self.analyze_button.configure(
            state="normal",
            text="▶  ANALYZE ALL"
        )

        self.count_label.configure(
            text=f"{len(self.image_paths)} images analyzed"
        )

    # =====================================================
    # SINGLE IMAGE
    # =====================================================

    def analyze_single(
        self,
        index,
        path
    ):

        image = cv2.imread(path)

        if image is None:
            return

        quality, brightness, sharpness = (
            check_image_quality(image)
        )

        prediction, score, inference_time = (
            predict_image(
                self.model,
                self.classes,
                image
            )
        )

        gradcam_path = None

        try:

            cam = self.gradcam.generate(
                image.copy()
            )

            gradcam_path = os.path.join(
                RESULTS_DIR,
                f"image_{index + 1}_gradcam.jpg"
            )

            cv2.imwrite(
                gradcam_path,
                cam
            )

        except Exception as e:

            print(
                "Grad-CAM:",
                e
            )

        self.create_result_card(
            index,
            path,
            quality,
            brightness,
            sharpness,
            prediction,
            score,
            inference_time,
            gradcam_path
        )

    # =====================================================
    # RESULT CARD
    # =====================================================

    def create_result_card(
        self,
        index,
        path,
        quality,
        brightness,
        sharpness,
        prediction,
        score,
        inference_time,
        gradcam_path
    ):

        card = ctk.CTkFrame(
            self.scroll_frame,
            fg_color=CARD,
            corner_radius=20,
            border_width=1,
            border_color=BORDER
        )

        card.pack(
            fill="x",
            padx=8,
            pady=8
        )

        # LEFT IMAGE
        try:

            image = Image.open(path)
            image.thumbnail((210, 155))

            preview = ImageTk.PhotoImage(image)

            image_label = ctk.CTkLabel(
                card,
                text="",
                image=preview
            )

            image_label.image = preview

            image_label.grid(
                row=0,
                column=0,
                rowspan=3,
                padx=(18, 10),
                pady=18
            )

        except Exception:
            pass

        # CENTER
        center = ctk.CTkFrame(
            card,
            fg_color="transparent"
        )

        center.grid(
            row=0,
            column=1,
            rowspan=3,
            sticky="nsew",
            padx=15,
            pady=15
        )

        ctk.CTkLabel(
            center,
            text=f"IMAGE {index + 1:02d}",
            text_color=CYAN,
            font=("Arial", 11, "bold")
        ).pack(
            anchor="w"
        )

        ctk.CTkLabel(
            center,
            text=os.path.basename(path),
            text_color=TEXT,
            font=("Arial", 15, "bold")
        ).pack(
            anchor="w",
            pady=(3, 12)
        )

        # RESULT BOX
        result_box = ctk.CTkFrame(
            center,
            fg_color=CARD_2,
            corner_radius=14
        )

        result_box.pack(
            fill="x"
        )

        ctk.CTkLabel(
            result_box,
            text="AI PREDICTION",
            text_color=MUTED,
            font=("Arial", 9, "bold")
        ).pack(
            anchor="w",
            padx=14,
            pady=(10, 0)
        )

        prediction_text = prediction.replace(
            "_",
            " "
        ).upper()

        ctk.CTkLabel(
            result_box,
            text=prediction_text,
            text_color=GREEN if "BENIGN" in prediction.upper() else RED,
            font=("Arial", 20, "bold")
        ).pack(
            anchor="w",
            padx=14
        )

        ctk.CTkLabel(
            result_box,
            text=f"MODEL SCORE   {score:.2f}%",
            text_color=TEXT,
            font=("Arial", 12, "bold")
        ).pack(
            anchor="w",
            padx=14,
            pady=(2, 12)
        )

        # METRICS
        metrics = ctk.CTkFrame(
            center,
            fg_color="transparent"
        )

        metrics.pack(
            fill="x",
            pady=(12, 0)
        )

        metrics_text = (
            f"QUALITY   {quality}\n"
            f"BRIGHTNESS   {brightness:.2f}\n"
            f"SHARPNESS   {sharpness:.2f}\n"
            f"INFERENCE   {inference_time:.2f} ms"
        )

        ctk.CTkLabel(
            metrics,
            text=metrics_text,
            text_color=MUTED,
            font=("Arial", 10),
            justify="left"
        ).pack(
            anchor="w"
        )

        # RIGHT GRAD CAM
        if gradcam_path and os.path.exists(
            gradcam_path
        ):

            try:

                cam_image = Image.open(
                    gradcam_path
                )

                cam_image.thumbnail(
                    (230, 170)
                )

                cam_preview = ImageTk.PhotoImage(
                    cam_image
                )

                cam_box = ctk.CTkFrame(
                    card,
                    fg_color=CARD_2,
                    corner_radius=16
                )

                cam_box.grid(
                    row=0,
                    column=2,
                    rowspan=3,
                    padx=(10, 18),
                    pady=18
                )

                ctk.CTkLabel(
                    cam_box,
                    text="AI ATTENTION MAP",
                    text_color=PURPLE,
                    font=("Arial", 10, "bold")
                ).pack(
                    pady=(10, 5)
                )

                cam_label = ctk.CTkLabel(
                    cam_box,
                    text="",
                    image=cam_preview
                )

                cam_label.image = cam_preview

                cam_label.pack(
                    padx=10,
                    pady=(0, 10)
                )

            except Exception as e:

                print(
                    "CAM display:",
                    e
                )

        card.grid_columnconfigure(
            1,
            weight=1
        )

    # =====================================================
    # CLEAR
    # =====================================================

    def clear_all(self):

        self.image_paths = []
        self.analyzed_count = 0

        self.stat_images.configure(
            text="0"
        )

        self.stat_analyzed.configure(
            text="0"
        )

        self.count_label.configure(
            text="0 images ready"
        )

        self.show_empty_state()

    # =====================================================
    # UPDATE STATS
    # =====================================================

    def update_stats(self):

        self.stat_images.configure(
            text=str(
                len(self.image_paths)
            )
        )

        self.stat_analyzed.configure(
            text="0"
        )

        self.count_label.configure(
            text=(
                f"{len(self.image_paths)} "
                f"images ready"
            )
        )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    app = MultiScreeningApp()

    app.mainloop()