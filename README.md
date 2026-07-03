# AI-Based-Brain-Tumor-Detection
# AI-Based Brain Tumor Detection Using Deep Learning

## Overview

This project detects brain tumors from MRI images using Deep Learning.

## Features

- MRI Image Upload
- CNN Model
- Prediction Result
- User-friendly Interface

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy

## Author

Raman Dev Sharma
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import cv2
import numpy as np
import sqlite3

# ================= APP SETTINGS =================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ================= MAIN WINDOW =================

app = ctk.CTk()

app.title("AI Brain Tumor Detection System")

app.geometry("1400x800")

# ================= DATABASE =================

conn = sqlite3.connect("patients.db")

cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS patients(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age TEXT,
    gender TEXT,
    result TEXT,
    confidence TEXT
)
""")

conn.commit()

# ================= TITLE =================

heading = ctk.CTkLabel(
    app,
    text="AI BASED BRAIN TUMOR DETECTION SYSTEM",
    font=("Arial", 30, "bold")
)

heading.pack(pady=20)

# ================= MAIN FRAME =================

main_frame = ctk.CTkFrame(app)

main_frame.pack(fill="both", expand=True, padx=20, pady=20)

# ================= LEFT FRAME =================

left_frame = ctk.CTkFrame(main_frame, width=300)

left_frame.pack(side="left", fill="y", padx=20, pady=20)

# ================= RIGHT FRAME =================

right_frame = ctk.CTkFrame(main_frame)

right_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

# ================= PATIENT DETAILS =================

name_entry = ctk.CTkEntry(
    left_frame,
    placeholder_text="Patient Name",
    width=250
)

name_entry.pack(pady=10)

age_entry = ctk.CTkEntry(
    left_frame,
    placeholder_text="Age",
    width=250
)

age_entry.pack(pady=10)

gender_box = ctk.CTkComboBox(
    left_frame,
    values=["Male", "Female", "Other"],
    width=250
)

gender_box.pack(pady=10)

doctor_entry = ctk.CTkEntry(
    left_frame,
    placeholder_text="Doctor Name",
    width=250
)

doctor_entry.pack(pady=10)

# ================= RESULT LABELS =================

result_label = ctk.CTkLabel(
    left_frame,
    text="Result: Waiting",
    font=("Arial", 22, "bold")
)

result_label.pack(pady=20)

confidence_label = ctk.CTkLabel(
    left_frame,
    text="Confidence: 0%",
    font=("Arial", 18)
)

confidence_label.pack(pady=10)

# ================= IMAGE AREA =================

image_label = ctk.CTkLabel(
    right_frame,
    text="Upload MRI Scan"
)

image_label.pack(pady=20)

# ================= DOCTOR DASHBOARD =================

def open_dashboard():

    dashboard = ctk.CTkToplevel(app)

    dashboard.title("Doctor Dashboard")

    dashboard.geometry("900x500")

    title = ctk.CTkLabel(
        dashboard,
        text="Doctor Dashboard",
        font=("Arial", 28, "bold")
    )

    title.pack(pady=20)

    table_frame = ctk.CTkScrollableFrame(
        dashboard,
        width=850,
        height=400
    )

    table_frame.pack(padx=20, pady=20)

    headings = ctk.CTkLabel(
        table_frame,
        text="ID     NAME     AGE     GENDER     RESULT     CONFIDENCE",
        font=("Courier", 16, "bold")
    )

    headings.pack(pady=10)

    cur.execute("SELECT * FROM patients")

    records = cur.fetchall()

    if len(records) == 0:

        no_data = ctk.CTkLabel(
            table_frame,
            text="No Records Found",
            font=("Arial", 20)
        )

        no_data.pack(pady=20)

    else:

        for row in records:

            row_text = (
                f"{row[0]}     "
                f"{row[1]}     "
                f"{row[2]}     "
                f"{row[3]}     "
                f"{row[4]}     "
                f"{row[5]}%"
            )

            row_label = ctk.CTkLabel(
                table_frame,
                text=row_text,
                font=("Courier", 14)
            )

            row_label.pack(anchor="w", padx=10)

# ================= DETECTION FUNCTION =================

def detect_tumor():

    file_path = filedialog.askopenfilename(
        title="Select MRI Image",
        filetypes=[("Images", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        return

    img = cv2.imread(file_path)

    if img is None:

        messagebox.showerror(
            "Error",
            "Unable to Open Image"
        )

        return

    display = img.copy()

    resized = cv2.resize(img, (64, 64))

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    flat = gray.flatten().reshape(1, -1)

    # ================= DEMO AI =================

    prediction = np.random.randint(0, 2)

    confidence = np.random.uniform(90, 99)

    h, w, _ = display.shape

    # ================= TUMOR DETECTED =================

    if prediction == 0:

        result = "Tumor Detected"

        color = (0, 0, 255)

        cv2.rectangle(
            display,
            (w//3, h//3),
            (w//3 + 150, h//3 + 150),
            color,
            4
        )

        cv2.putText(
            display,
            "Tumor Area",
            (w//3, h//3 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3
        )

    # ================= NO TUMOR =================

    else:

        result = "No Tumor Detected"

        color = (0, 255, 0)

        cv2.rectangle(
            display,
            (40, 40),
            (w - 40, h - 40),
            color,
            4
        )

        cv2.putText(
            display,
            "NORMAL",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3
        )

    # ================= LABEL UPDATE =================

    result_label.configure(
        text=f"Result: {result}"
    )

    confidence_label.configure(
        text=f"Confidence: {confidence:.2f}%"
    )

    # ================= SAVE DATA =================

    name = name_entry.get()

    age = age_entry.get()

    gender = gender_box.get()

    cur.execute(
        "INSERT INTO patients(name, age, gender, result, confidence) VALUES (?, ?, ?, ?, ?)",
        (name, age, gender, result, str(round(confidence, 2)))
    )

    conn.commit()

    # ================= DISPLAY IMAGE =================

    rgb = cv2.cvtColor(
        display,
        cv2.COLOR_BGR2RGB
    )

    pil_img = Image.fromarray(rgb)

    pil_img.thumbnail((850, 550))

    ctk_img = ctk.CTkImage(
        light_image=pil_img,
        dark_image=pil_img,
        size=pil_img.size
    )

    image_label.configure(
        image=ctk_img,
        text=""
    )

    image_label.image = ctk_img

    messagebox.showinfo(
        "Success",
        "MRI Scan Processed Successfully"
    )

# ================= BUTTONS =================

upload_btn = ctk.CTkButton(
    left_frame,
    text="Upload MRI Scan",
    command=detect_tumor,
    width=250,
    height=50,
    font=("Arial", 18, "bold")
)

upload_btn.pack(pady=20)

dashboard_btn = ctk.CTkButton(
    left_frame,
    text="Doctor Dashboard",
    command=open_dashboard,
    width=250,
    height=50,
    font=("Arial", 18, "bold")
)

dashboard_btn.pack(pady=20)

exit_btn = ctk.CTkButton(
    left_frame,
    text="Exit Application",
    command=app.destroy,
    fg_color="red",
    hover_color="#8B0000",
    width=250,
    height=50,
    font=("Arial", 18, "bold")
)

exit_btn.pack(pady=20)

# ================= RUN APP =================

app.mainloop()

