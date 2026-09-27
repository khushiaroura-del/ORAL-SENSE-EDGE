import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()

app.title("ORAL-SENSE EDGE")
app.geometry("900x600")

title = ctk.CTkLabel(
    app,
    text="ORAL-SENSE EDGE",
    font=("Arial", 32, "bold")
)

title.pack(pady=40)

subtitle = ctk.CTkLabel(
    app,
    text="Privacy-First Oral Screening Assistant",
    font=("Arial", 18)
)

subtitle.pack(pady=10)

button = ctk.CTkButton(
    app,
    text="START SCREENING",
    width=250,
    height=50
)

button.pack(pady=40)

app.mainloop()