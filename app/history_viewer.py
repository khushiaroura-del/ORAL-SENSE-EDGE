import customtkinter as ctk
from screening_history import get_history


ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class HistoryViewer(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("ORAL-SENSE EDGE - Screening History")
        self.geometry("900x600")

        # Heading
        title = ctk.CTkLabel(
            self,
            text="SCREENING HISTORY",
            font=("Arial", 26, "bold")
        )

        title.pack(
            pady=(20, 5)
        )

        subtitle = ctk.CTkLabel(
            self,
            text="Local screening records",
            font=("Arial", 14)
        )

        subtitle.pack(
            pady=(0, 15)
        )

        # Scroll area
        self.history_frame = ctk.CTkScrollableFrame(
            self,
            width=820,
            height=450
        )

        self.history_frame.pack(
            padx=25,
            pady=10,
            fill="both",
            expand=True
        )

        self.show_history()

    def show_history(self):

        history = get_history()

        if not history:

            label = ctk.CTkLabel(
                self.history_frame,
                text="No screening history available.",
                font=("Arial", 16)
            )

            label.pack(
                pady=40
            )

            return

        # Latest first
        history = history[::-1]

        for index, record in enumerate(history, start=1):

            card = ctk.CTkFrame(
                self.history_frame
            )

            card.pack(
                fill="x",
                padx=10,
                pady=8
            )

            date_text = record.get(
                "date_time",
                "Unknown"
            )

            image_text = record.get(
                "image_name",
                "Unknown"
            )

            result_text = record.get(
                "predicted_class",
                "Unknown"
            )

            score = record.get(
                "model_score",
                0
            )

            quality = record.get(
                "image_quality",
                "Unknown"
            )

            text = (
                f"Screening #{index}\n"
                f"Date & Time: {date_text}\n"
                f"Image: {image_text}\n"
                f"Result: {result_text}\n"
                f"Model Score: {score}%\n"
                f"Image Quality: {quality}"
            )

            label = ctk.CTkLabel(
                card,
                text=text,
                justify="left",
                anchor="w",
                font=("Arial", 14)
            )

            label.pack(
                padx=15,
                pady=12,
                fill="x"
            )


if __name__ == "__main__":

    app = HistoryViewer()

    app.mainloop()