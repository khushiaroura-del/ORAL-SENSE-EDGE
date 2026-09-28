import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import customtkinter as ctk

from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
from torchvision import models, transforms

from referral_guidance import get_referral_guidance
from screening_history import save_screening, get_history
from analytics import get_analytics
from report_generator import generate_screening_report


# ============================================================
# ORAL-SENSE EDGE — WOW DASHBOARD
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "oral_sense_mobilenetv3.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ---------------- MODEL ----------------

def load_model():
    model = models.mobilenet_v3_small(weights=None)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    classes = ["benign_lesions", "malignant_lesions"]

    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict", checkpoint)
        classes = checkpoint.get("classes", classes)
    else:
        state = checkpoint

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, len(classes)
    )
    model.load_state_dict(state)
    model.eval()
    return model, classes


MODEL, CLASSES = load_model()

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def quality_check(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness < 50:
        status = "TOO DARK"
    elif brightness > 210:
        status = "TOO BRIGHT"
    elif sharpness < 50:
        status = "TOO BLURRY"
    else:
        status = "GOOD IMAGE"

    return status, brightness, sharpness


def predict(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = TRANSFORM(rgb).unsqueeze(0)

    with torch.no_grad():
        output = MODEL(x)
        probs = torch.softmax(output, dim=1)[0]

    index = int(torch.argmax(probs))
    return CLASSES[index], float(probs[index] * 100), index


# ---------------- GRAD-CAM ----------------

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        layer = model.features[-1]

        layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach())
        )

        layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach())
        )

    def generate(self, image, class_index):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = TRANSFORM(rgb).unsqueeze(0)
        self.model.zero_grad()

        output = self.model(x)
        output[0, class_index].backward()

        a = self.activations[0]
        g = self.gradients[0]
        weights = g.mean(dim=(1, 2), keepdim=True)

        cam = torch.relu((weights * a).sum(dim=0))
        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        cam = cv2.resize(
            cam.cpu().numpy(),
            (image.shape[1], image.shape[0])
        )

        heat = cv2.applyColorMap(
            np.uint8(cam * 255),
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            image, 0.55, heat, 0.45, 0
        )

        return overlay


CAM = GradCAM(MODEL)


def pretty(name):
    return name.replace("_", " ").title()


# ---------------- DASHBOARD ----------------

class Dashboard(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("ORAL-SENSE EDGE • AI Screening Intelligence")
        self.geometry("1450x900")
        self.minsize(1150, 720)
        self.configure(fg_color="#07111F")

        self.image = None
        self.image_path = None
        self.paths = []
        self.index = 0
        self.visit1 = None
        self.visit2 = None
        self.analytics_values = {}
        self.last_report = None

        self.build()

    # ---------- helpers ----------

    def button(self, parent, text, command, color="#14324A"):
        return ctk.CTkButton(
            parent, text=text, command=command,
            height=36, corner_radius=9,
            fg_color=color, hover_color="#1D5272",
            font=("Arial", 11, "bold")
        )

    def card(self, parent, title):
        frame = ctk.CTkFrame(
            parent, fg_color="#101F33",
            corner_radius=15, border_width=1,
            border_color="#20415F"
        )
        frame.pack(fill="x", padx=15, pady=7)

        ctk.CTkLabel(
            frame, text=title,
            font=("Arial", 11, "bold"),
            text_color="#7591AD"
        ).pack(anchor="w", padx=15, pady=(13, 6))

        return frame

    def value(self, frame, text):
        label = ctk.CTkLabel(
            frame, text=text,
            justify="left", anchor="w",
            font=("Arial", 13, "bold"),
            text_color="#D8E8F7"
        )
        label.pack(fill="x", padx=15, pady=(0, 14))
        return label

    def status(self, text):
        self.status_label.configure(text=text)

    # ---------- UI ----------

    def build(self):

        header = ctk.CTkFrame(
            self, fg_color="#0B1729",
            corner_radius=0, height=82
        )
        header.pack(fill="x")
        header.pack_propagate(False)

        brand = ctk.CTkFrame(header, fg_color="transparent")
        brand.pack(side="left", padx=28)

        ctk.CTkLabel(
            brand, text="ORAL-SENSE",
            font=("Arial", 29, "bold"),
            text_color="#EAF4FF"
        ).pack(side="left")

        ctk.CTkLabel(
            brand, text=" EDGE",
            font=("Arial", 29, "bold"),
            text_color="#42D9FF"
        ).pack(side="left")

        ctk.CTkLabel(
            header,
            text="OFFLINE  •  EXPLAINABLE  •  PRIVACY-FIRST",
            font=("Arial", 11, "bold"),
            text_color="#7894AE"
        ).pack(side="right", padx=28)

        toolbar = ctk.CTkFrame(
            self, fg_color="#091525",
            corner_radius=0, height=62
        )
        toolbar.pack(fill="x")
        toolbar.pack_propagate(False)

        for text, command, color in [
            ("📷  CAPTURE", self.capture, "#14324A"),
            ("＋  UPLOAD", self.upload, "#14324A"),
            ("⚡  ANALYZE", self.analyze, "#087C8C"),
            ("◷  HISTORY", self.history, "#293052"),
            ("▣  EXPORT PDF", self.export_pdf, "#4A315F"),
            ("CLEAR", self.clear, "#30233A")
        ]:
            self.button(
                toolbar, text, command, color
            ).pack(side="left", padx=6, pady=12)

        self.status_label = ctk.CTkLabel(
            toolbar, text="● SYSTEM READY",
            font=("Arial", 11, "bold"),
            text_color="#39D98A"
        )
        self.status_label.pack(side="right", padx=24)

        main = ctk.CTkFrame(self, fg_color="#07111F")
        main.pack(fill="both", expand=True, padx=18, pady=18)

        main.grid_columnconfigure(0, weight=7)
        main.grid_columnconfigure(1, weight=5)
        main.grid_rowconfigure(0, weight=1)

        # LEFT
        left = ctk.CTkFrame(
            main, fg_color="#0B1729",
            corner_radius=18,
            border_width=1, border_color="#1B3855"
        )
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 9))

        ctk.CTkLabel(
            left, text="IMAGE REVIEW",
            font=("Arial", 18, "bold"),
            text_color="#EAF4FF"
        ).pack(anchor="w", padx=22, pady=(20, 2))

        ctk.CTkLabel(
            left, text="Camera capture / multiple image upload",
            font=("Arial", 11),
            text_color="#718AA4"
        ).pack(anchor="w", padx=22)

        self.image_box = ctk.CTkFrame(
            left, fg_color="#050C16",
            corner_radius=16,
            border_width=1, border_color="#19334E"
        )
        self.image_box.pack(
            fill="both", expand=True,
            padx=22, pady=15
        )

        self.image_label = ctk.CTkLabel(
            self.image_box,
            text="NO IMAGE\n\nUpload or capture an image",
            font=("Arial", 16),
            text_color="#58728D"
        )
        self.image_label.pack(fill="both", expand=True)

        nav = ctk.CTkFrame(left, fg_color="transparent")
        nav.pack(fill="x", padx=22, pady=(0, 20))

        self.button(
            nav, "‹ PREVIOUS", self.previous, "#12243A"
        ).pack(side="left")

        self.counter = ctk.CTkLabel(
            nav, text="0 / 0",
            font=("Arial", 12, "bold"),
            text_color="#7894AE"
        )
        self.counter.pack(side="left", expand=True)

        self.button(
            nav, "NEXT ›", self.next, "#12243A"
        ).pack(side="right")

        # RIGHT
        right = ctk.CTkScrollableFrame(
            main, fg_color="#0B1729",
            corner_radius=18,
            border_width=1, border_color="#1B3855"
        )
        right.grid(row=0, column=1, sticky="nsew", padx=(9, 0))

        ctk.CTkLabel(
            right, text="SCREENING INTELLIGENCE",
            font=("Arial", 18, "bold"),
            text_color="#EAF4FF"
        ).pack(anchor="w", padx=20, pady=(20, 2))

        ctk.CTkLabel(
            right,
            text="AI-assisted screening • not a medical diagnosis",
            font=("Arial", 11),
            text_color="#718AA4"
        ).pack(anchor="w", padx=20, pady=(0, 12))

        result = self.card(right, "CLASSIFICATION")
        self.result = ctk.CTkLabel(
            result, text="WAITING",
            font=("Arial", 27, "bold"),
            text_color="#7AA7FF"
        )
        self.result.pack(anchor="w", padx=15)

        self.score = self.value(result, "Model score: —")

        quality = self.card(right, "IMAGE QUALITY")
        self.quality = self.value(quality, "Not analyzed")

        system = self.card(right, "SYSTEM INFORMATION")
        self.system = self.value(
            system,
            "Model: MobileNetV3 Small\n"
            "Device: CPU\n"
            "Mode: Local / Offline"
        )

        attention = self.card(right, "MODEL ATTENTION MAP")
        self.attention = ctk.CTkLabel(
            attention,
            text="Grad-CAM appears after analysis.",
            font=("Arial", 11),
            text_color="#6F8AA5"
        )
        self.attention.pack(padx=15, pady=(0, 15))

        referral = self.card(right, "REFERRAL GUIDANCE")
        self.referral = self.value(
            referral,
            "Waiting for screening result."
        )

        # ---------------- ANALYTICS ----------------
        analytics_card = self.card(right, "SCREENING ANALYTICS")

        ctk.CTkLabel(
            analytics_card,
            text="Local history summary • model outputs, not clinical statistics",
            font=("Arial", 10),
            text_color="#607B95"
        ).pack(anchor="w", padx=15, pady=(0, 10))

        stats = ctk.CTkFrame(analytics_card, fg_color="transparent")
        stats.pack(fill="x", padx=10, pady=(0, 8))
        stats.grid_columnconfigure((0, 1, 2), weight=1)

        for col, (key, label) in enumerate([
            ("total", "TOTAL"),
            ("benign", "MODEL: BENIGN"),
            ("malignant", "MODEL: MALIGNANT"),
        ]):
            box = ctk.CTkFrame(
                stats, fg_color="#0B1729", corner_radius=10,
                border_width=1, border_color="#19334E"
            )
            box.grid(row=0, column=col, sticky="ew", padx=4, pady=4)
            ctk.CTkLabel(
                box, text=label, font=("Arial", 9, "bold"),
                text_color="#6F8AA5"
            ).pack(pady=(9, 2))
            value_label = ctk.CTkLabel(
                box, text="0", font=("Arial", 20, "bold"),
                text_color="#EAF4FF"
            )
            value_label.pack(pady=(0, 9))
            self.analytics_values[key] = value_label

        stats2 = ctk.CTkFrame(analytics_card, fg_color="transparent")
        stats2.pack(fill="x", padx=10, pady=(0, 8))
        stats2.grid_columnconfigure((0, 1), weight=1)

        for col, (key, label) in enumerate([
            ("good_images", "GOOD IMAGES"),
            ("follow_up", "FOLLOW-UP FLAGS"),
        ]):
            box = ctk.CTkFrame(
                stats2, fg_color="#0B1729", corner_radius=10,
                border_width=1, border_color="#19334E"
            )
            box.grid(row=0, column=col, sticky="ew", padx=4, pady=4)
            ctk.CTkLabel(
                box, text=label, font=("Arial", 9, "bold"),
                text_color="#6F8AA5"
            ).pack(pady=(9, 2))
            value_label = ctk.CTkLabel(
                box, text="0", font=("Arial", 20, "bold"),
                text_color="#EAF4FF"
            )
            value_label.pack(pady=(0, 9))
            self.analytics_values[key] = value_label

        ctk.CTkLabel(
            analytics_card, text="MODEL OUTPUT DISTRIBUTION",
            font=("Arial", 9, "bold"), text_color="#6F8AA5"
        ).pack(anchor="w", padx=15, pady=(4, 5))

        self.benign_bar = ctk.CTkProgressBar(
            analytics_card, height=8, corner_radius=5,
            progress_color="#39D98A", fg_color="#172A3D"
        )
        self.benign_bar.pack(fill="x", padx=15, pady=(0, 3))
        self.benign_bar.set(0)

        self.benign_dist = ctk.CTkLabel(
            analytics_card, text="Model: Benign — 0",
            font=("Arial", 10), text_color="#8FA9BF"
        )
        self.benign_dist.pack(anchor="w", padx=15, pady=(0, 7))

        self.malignant_bar = ctk.CTkProgressBar(
            analytics_card, height=8, corner_radius=5,
            progress_color="#FF5C7A", fg_color="#172A3D"
        )
        self.malignant_bar.pack(fill="x", padx=15, pady=(0, 3))
        self.malignant_bar.set(0)

        self.malignant_dist = ctk.CTkLabel(
            analytics_card, text="Model: Malignant — 0",
            font=("Arial", 10), text_color="#8FA9BF"
        )
        self.malignant_dist.pack(anchor="w", padx=15, pady=(0, 12))

        self.refresh_analytics()

        change = self.card(right, "CHANGE MONITORING")

        self.change_text = self.value(
            change,
            "Select two visits to inspect visual difference."
        )

        row = ctk.CTkFrame(change, fg_color="transparent")
        row.pack(fill="x", padx=15, pady=(0, 15))

        self.button(
            row, "VISIT 1", self.select_visit1, "#12243A"
        ).pack(side="left", padx=(0, 5))

        self.button(
            row, "VISIT 2", self.select_visit2, "#12243A"
        ).pack(side="left", padx=5)

        self.button(
            row, "COMPARE", self.compare, "#087C8C"
        ).pack(side="right", padx=(5, 0))

        ctk.CTkLabel(
            right,
            text=(
                "SCREENING ASSISTANCE ONLY — NOT A MEDICAL DIAGNOSIS\n"
                "Model score is not clinical certainty. Grad-CAM shows "
                "model-influential regions, not a confirmed lesion."
            ),
            justify="left",
            font=("Arial", 10),
            text_color="#607B95"
        ).pack(fill="x", padx=20, pady=18)

        footer = ctk.CTkFrame(
            self, fg_color="#050C16",
            corner_radius=0, height=34
        )
        footer.pack(fill="x")
        footer.pack_propagate(False)

        ctk.CTkLabel(
            footer,
            text="ORAL-SENSE EDGE • Privacy-first local prototype",
            font=("Arial", 10),
            text_color="#5D7893"
        ).pack(side="left", padx=20)

        ctk.CTkLabel(
            footer,
            text="● OFFLINE MODE",
            font=("Arial", 10, "bold"),
            text_color="#39D98A"
        ).pack(side="right", padx=20)

    # ---------- image handling ----------

    def show_image(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((
            max(self.image_box.winfo_width() - 30, 450),
            max(self.image_box.winfo_height() - 30, 350)
        ))

        photo = ImageTk.PhotoImage(pil)
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo

    def load_current(self):
        if not self.paths:
            return

        path = self.paths[self.index]
        image = cv2.imread(path)

        if image is None:
            return

        self.image = image
        self.image_path = path
        self.show_image(image)

        self.counter.configure(
            text=f"{self.index + 1} / {len(self.paths)}"
        )

    def upload(self):
        paths = filedialog.askopenfilenames(
            title="Select oral images",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")
            ]
        )

        if paths:
            self.paths = list(paths)
            self.index = 0
            self.load_current()
            self.status(f"● {len(paths)} IMAGE(S) LOADED")

    def capture(self):
        self.status("● CAMERA OPEN")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                "Camera could not be opened."
            )
            return

        frame_saved = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            cv2.imshow("ORAL-SENSE EDGE • Press SPACE to capture / Q to cancel", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 32:
                frame_saved = frame.copy()
                break

        cap.release()
        cv2.destroyAllWindows()

        if frame_saved is None:
            self.status("● SYSTEM READY")
            return

        path = os.path.join(RESULTS_DIR, "current_image.jpg")
        cv2.imwrite(path, frame_saved)

        self.paths = [path]
        self.index = 0
        self.load_current()
        self.status("● IMAGE CAPTURED")

    def previous(self):
        if not self.paths:
            return
        self.index = (self.index - 1) % len(self.paths)
        self.load_current()

    def next(self):
        if not self.paths:
            return
        self.index = (self.index + 1) % len(self.paths)
        self.load_current()

    # ---------- analyze ----------

    def analyze(self):
        if self.image is None:
            messagebox.showwarning(
                "No Image",
                "Please capture or upload an image first."
            )
            return

        try:
            self.status("● ANALYZING...")
            self.update_idletasks()

            quality, brightness, sharpness = quality_check(self.image)
            prediction, score, class_index = predict(self.image)

            overlay = CAM.generate(
                self.image, class_index
            )

            grad_path = os.path.join(
                RESULTS_DIR,
                f"screening_gradcam_{int(time.time())}.jpg"
            )
            cv2.imwrite(grad_path, overlay)

            # Save current image
            current_path = os.path.join(
                RESULTS_DIR, "current_image.jpg"
            )
            cv2.imwrite(current_path, self.image)

            save_screening(
                os.path.basename(self.image_path or "image.jpg"),
                prediction,
                score,
                quality
            )

            # Result
            result_color = (
                "#FF5C7A"
                if "malignant" in prediction.lower()
                else "#39D98A"
                if "benign" in prediction.lower()
                else "#7AA7FF"
            )

            self.result.configure(
                text=pretty(prediction),
                text_color=result_color
            )
            self.score.configure(
                text=f"Model score: {score:.2f}%"
            )

            self.quality.configure(
                text=(
                    f"{quality}\n"
                    f"Brightness: {brightness:.2f}\n"
                    f"Sharpness: {sharpness:.2f}"
                )
            )

            self.system.configure(
                text=(
                    "Model: MobileNetV3 Small\n"
                    "Device: CPU\n"
                    "Mode: Local / Offline\n"
                    "Input: 224 × 224"
                )
            )

            self.show_attention(overlay)

            guidance = get_referral_guidance(
                prediction, score
            )

            self.referral.configure(
                text=(
                    f"{guidance['level']}\n\n"
                    f"{guidance['guidance']}\n\n"
                    f"{guidance['disclaimer']}"
                )
            )

            self.refresh_analytics()

            self.last_report = {
                "image_path": current_path,
                "gradcam_path": grad_path,
                "quality": quality,
                "brightness": brightness,
                "sharpness": sharpness,
                "prediction": prediction,
                "score": score,
                "guidance": guidance,
            }

            self.show_screening_summary(
                quality, prediction, score, grad_path, guidance
            )

            self.status("● ANALYSIS COMPLETE")

        except Exception as e:
            self.status("● ANALYSIS ERROR")
            messagebox.showerror(
                "Analysis Error",
                str(e)
            )

    def show_attention(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((420, 250))

        photo = ImageTk.PhotoImage(pil)
        self.attention.configure(image=photo, text="")
        self.attention.image = photo

    # ---------- smart summary ----------

    def show_screening_summary(
        self, quality, prediction, score, grad_path, guidance
    ):
        summary = ctk.CTkToplevel(self)
        summary.title("ORAL-SENSE EDGE • Screening Summary")
        summary.geometry("620x650")
        summary.minsize(560, 600)
        summary.configure(fg_color="#07111F")

        ctk.CTkLabel(
            summary, text="✓  SCREENING COMPLETE",
            font=("Arial", 26, "bold"), text_color="#42D9FF"
        ).pack(pady=(25, 5))

        ctk.CTkLabel(
            summary, text="AI-assisted screening summary",
            font=("Arial", 12), text_color="#718AA4"
        ).pack(pady=(0, 20))

        card = ctk.CTkFrame(
            summary, fg_color="#101F33", corner_radius=18,
            border_width=1, border_color="#20415F"
        )
        card.pack(fill="x", padx=30, pady=5)

        self.summary_row(card, "IMAGE QUALITY", quality)
        self.summary_row(card, "AI CLASSIFICATION", pretty(prediction))
        self.summary_row(card, "MODEL SCORE", f"{score:.2f}%")
        self.summary_row(card, "ATTENTION MAP", "AVAILABLE")
        self.summary_row(card, "PROCESSING", "LOCAL / OFFLINE")
        self.summary_row(card, "GUIDANCE", guidance["level"])

        note = ctk.CTkFrame(
            summary, fg_color="#0B1729", corner_radius=14
        )
        note.pack(fill="x", padx=30, pady=18)

        ctk.CTkLabel(
            note,
            text=(
                "IMPORTANT\n\n"
                "This tool provides screening assistance only.\n"
                "The model score is not clinical certainty.\n"
                "Grad-CAM shows model-influential regions,\n"
                "not a confirmed lesion location.\n\n"
                "Change Monitoring is visual comparison only."
            ),
            justify="left", font=("Arial", 11),
            text_color="#91A9BF"
        ).pack(padx=18, pady=16)

        action_row = ctk.CTkFrame(summary, fg_color="transparent")
        action_row.pack(pady=(0, 20))

        self.button(
            action_row, "EXPORT PDF", self.export_pdf, "#4A315F"
        ).pack(side="left", padx=5)

        self.button(
            action_row, "CLOSE", summary.destroy, "#14324A"
        ).pack(side="left", padx=5)

    def export_pdf(self):
        if not self.last_report:
            messagebox.showwarning(
                "No Report",
                "Please analyze an image first, then export the screening report."
            )
            return

        default_name = "oral_sense_screening_report.pdf"
        path = filedialog.asksaveasfilename(
            title="Save Screening Report",
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[("PDF files", "*.pdf")]
        )

        if not path:
            return

        try:
            data = self.last_report
            analytics = get_analytics()
            generate_screening_report(
                output_path=path,
                image_path=data["image_path"],
                gradcam_path=data["gradcam_path"],
                quality=data["quality"],
                brightness=data["brightness"],
                sharpness=data["sharpness"],
                prediction=data["prediction"],
                score=data["score"],
                guidance=data["guidance"],
                analytics=analytics,
            )

            self.status("● PDF REPORT EXPORTED")
            messagebox.showinfo(
                "Report Generated",
                f"Professional screening report saved successfully.\n\n{path}"
            )
        except Exception as e:
            messagebox.showerror(
                "PDF Export Error",
                str(e)
            )

    def summary_row(self, parent, label, value):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=8)

        ctk.CTkLabel(
            row, text=label, font=("Arial", 11, "bold"),
            text_color="#718AA4", anchor="w"
        ).pack(side="left")

        ctk.CTkLabel(
            row, text=value, font=("Arial", 12, "bold"),
            text_color="#EAF4FF", anchor="e"
        ).pack(side="right")

    # ---------- analytics ----------

    def refresh_analytics(self):
        try:
            data = get_analytics()
        except Exception:
            data = {
                "total": 0, "benign": 0, "malignant": 0,
                "good_images": 0, "follow_up": 0
            }

        for key, label in self.analytics_values.items():
            label.configure(text=str(data.get(key, 0)))

        total = max(int(data.get("total", 0)), 1)
        benign = int(data.get("benign", 0))
        malignant = int(data.get("malignant", 0))

        self.benign_bar.set(min(benign / total, 1))
        self.malignant_bar.set(min(malignant / total, 1))

        self.benign_dist.configure(
            text=f"Model: Benign — {benign} ({benign / total * 100:.1f}%)"
        )
        self.malignant_dist.configure(
            text=f"Model: Malignant — {malignant} ({malignant / total * 100:.1f}%)"
        )

    # ---------- history ----------

    def history(self):
        win = ctk.CTkToplevel(self)
        win.title("ORAL-SENSE EDGE • Screening History")
        win.geometry("850x650")
        win.configure(fg_color="#07111F")

        ctk.CTkLabel(
            win, text="SCREENING HISTORY",
            font=("Arial", 27, "bold"),
            text_color="#EAF4FF"
        ).pack(pady=(22, 3))

        ctk.CTkLabel(
            win,
            text="Stored locally on this computer",
            font=("Arial", 12),
            text_color="#718AA4"
        ).pack(pady=(0, 15))

        frame = ctk.CTkScrollableFrame(
            win, fg_color="#0B1729",
            corner_radius=15
        )
        frame.pack(fill="both", expand=True, padx=20, pady=10)

        records = get_history()

        if not records:
            ctk.CTkLabel(
                frame,
                text="No screening records yet.",
                font=("Arial", 16),
                text_color="#718AA4"
            ).pack(pady=50)
            return

        for number, record in enumerate(
            reversed(records), 1
        ):
            box = ctk.CTkFrame(
                frame, fg_color="#101F33",
                corner_radius=12,
                border_width=1,
                border_color="#20415F"
            )
            box.pack(fill="x", padx=8, pady=7)

            result = record.get(
                "predicted_class", "Unknown"
            )

            text = (
                f"SCREENING #{number}\n"
                f"Date & Time: {record.get('date_time', '—')}\n"
                f"Image: {record.get('image_name', '—')}\n"
                f"Result: {pretty(result)}\n"
                f"Model Score: {record.get('model_score', 0)}%\n"
                f"Image Quality: {record.get('image_quality', '—')}"
            )

            ctk.CTkLabel(
                box, text=text,
                justify="left", anchor="w",
                font=("Arial", 13),
                text_color="#D8E8F7"
            ).pack(fill="x", padx=16, pady=14)

    # ---------- change monitoring ----------

    def select_visit1(self):
        self.visit1 = filedialog.askopenfilename(
            title="Select Visit 1",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")
            ]
        )

        if self.visit1:
            self.change_text.configure(
                text=f"Visit 1 selected:\n{os.path.basename(self.visit1)}"
            )

    def select_visit2(self):
        self.visit2 = filedialog.askopenfilename(
            title="Select Visit 2",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")
            ]
        )

        if self.visit2:
            self.change_text.configure(
                text=f"Visit 2 selected:\n{os.path.basename(self.visit2)}"
            )

    def compare(self):
        if not self.visit1 or not self.visit2:
            messagebox.showwarning(
                "Missing Images",
                "Please select Visit 1 and Visit 2."
            )
            return

        a = cv2.imread(self.visit1)
        b = cv2.imread(self.visit2)

        if a is None or b is None:
            messagebox.showerror(
                "Image Error",
                "Could not read one of the images."
            )
            return

        b = cv2.resize(
            b, (a.shape[1], a.shape[0])
        )

        g1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(g1, g2)

        percentage = float(
            np.mean(diff) / 255 * 100
        )

        if percentage < 15:
            level = "LOW VISUAL DIFFERENCE"
        elif percentage < 35:
            level = "MODERATE VISUAL DIFFERENCE"
        else:
            level = "HIGH VISUAL DIFFERENCE"

        heat = cv2.applyColorMap(
            diff, cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            b, 0.60, heat, 0.40, 0
        )

        path = os.path.join(
            RESULTS_DIR, "changed_area_overlay.jpg"
        )
        cv2.imwrite(path, overlay)

        self.change_text.configure(
            text=(
                f"{level}\n\n"
                f"Visual Difference: {percentage:.2f}%\n\n"
                "Lighting, camera angle and framing can affect this result.\n"
                "This is visual comparison only — not cancer progression."
            )
        )

        win = ctk.CTkToplevel(self)
        win.title("ORAL-SENSE EDGE • Visit Comparison")
        win.geometry("950x700")
        win.configure(fg_color="#07111F")

        ctk.CTkLabel(
            win, text="VISIT COMPARISON",
            font=("Arial", 24, "bold"),
            text_color="#EAF4FF"
        ).pack(pady=(20, 5))

        ctk.CTkLabel(
            win,
            text=f"{level}  •  {percentage:.2f}% visual difference",
            font=("Arial", 14, "bold"),
            text_color="#42D9FF"
        ).pack()

        rgb = cv2.cvtColor(
            overlay, cv2.COLOR_BGR2RGB
        )
        pil = Image.fromarray(rgb)
        pil.thumbnail((850, 500))

        photo = ImageTk.PhotoImage(pil)

        label = ctk.CTkLabel(
            win, image=photo, text=""
        )
        label.pack(pady=18)
        label.image = photo

        ctk.CTkLabel(
            win,
            text="Visual comparison only — not a clinical assessment.",
            font=("Arial", 11),
            text_color="#718AA4"
        ).pack()

    # ---------- clear ----------

    def clear(self):
        self.image = None
        self.image_path = None
        self.paths = []
        self.index = 0

        self.image_label.configure(
            image=None,
            text="NO IMAGE\n\nUpload or capture an image"
        )
        self.image_label.image = None

        self.counter.configure(text="0 / 0")
        self.result.configure(
            text="WAITING",
            text_color="#7AA7FF"
        )
        self.score.configure(text="Model score: —")
        self.quality.configure(text="Not analyzed")
        self.attention.configure(
            image=None,
            text="Grad-CAM appears after analysis."
        )
        self.attention.image = None
        self.referral.configure(
            text="Waiting for screening result."
        )
        self.change_text.configure(
            text="Select two visits to inspect visual difference."
        )

        self.refresh_analytics()
        self.status("● SYSTEM READY")


if __name__ == "__main__":
    app = Dashboard()
    app.mainloop()
