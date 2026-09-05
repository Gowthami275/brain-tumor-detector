import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
import tempfile
import os
from datetime import datetime
from fpdf import FPDF

from gradcam import gradcam

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(
    "brain_tumor_fullfull_model.keras",
    compile=False
)

classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']


# ---------------- PDF FUNCTION ----------------
def generate_pdf(name, age, gender, phone, email,
                 pred_class, confidence, location, severity,
                 img, heatmap_img):

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "M S Ramaiah Institute of Technology", ln=True, align="C")

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, "MSRIT, Bengaluru, Karnataka - 560054", ln=True, align="C")

    pdf.ln(3)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())

    pdf.ln(5)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Brain Tumor Detection Report", ln=True, align="C")

    pdf.ln(5)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Report Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}", ln=True)

    pdf.ln(3)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Patient Details", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(95, 8, f"Name: {name}", border=1)
    pdf.cell(95, 8, f"Age: {age}", border=1, ln=True)

    pdf.cell(95, 8, f"Gender: {gender}", border=1)
    pdf.cell(95, 8, f"Phone: {phone}", border=1, ln=True)

    pdf.cell(190, 8, f"Email: {email}", border=1, ln=True)

    pdf.ln(8)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Diagnosis Result", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Predicted Class: {pred_class}", ln=True)
    pdf.cell(0, 8, f"Confidence: {round(confidence*100,2)}%", ln=True)

    if pred_class != "No Tumor":
        pdf.cell(0, 8, f"Tumor Location: {location}", ln=True)
        pdf.cell(0, 8, f"Severity Level: {severity}", ln=True)
    else:
        pdf.cell(0, 8, "No tumor detected.", ln=True)

    pdf.ln(8)

    img_path = tempfile.mktemp(suffix=".png")
    heatmap_path = tempfile.mktemp(suffix=".png")

    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(heatmap_path, heatmap_img)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "MRI Analysis Images:", ln=True)

    img_width = 80

    pdf.cell(95, 8, "Original MRI", ln=0)
    if pred_class != "No Tumor":
        pdf.cell(95, 8, "Grad-CAM Heatmap", ln=1)
    else:
        pdf.ln()

    y_before = pdf.get_y()

    pdf.image(img_path, x=15, y=y_before, w=img_width)

    if pred_class != "No Tumor":
        pdf.image(heatmap_path, x=110, y=y_before, w=img_width)

    pdf.set_y(y_before + img_width + 40)

    pdf.ln(10)

    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(0, 6,
        "Disclaimer: This report is AI-generated and not a substitute for medical advice.")

    pdf_path = f"{name}_brain_tumor_report.pdf"
    pdf.output(pdf_path)

    return pdf_path


# ---------------- MAIN FUNCTION ----------------
def predict(image, name, age, gender, phone, email):

    img = cv2.resize(image, (299, 299))
    img_array = img / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = float(prediction[0][class_index])
    pred_class = classes[class_index]

    if pred_class == "No Tumor":
        location = "None"
        severity = "None"
        superimposed = img
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        cv2.imwrite(tmp.name, img)

        heatmap, location = gradcam(tmp.name, model, "block14_sepconv2_act")

        heatmap = cv2.resize(heatmap, (299, 299))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        score = np.mean(heatmap)

        if score < 0.3:
            severity = "Mild"
        elif score < 0.6:
            severity = "Moderate"
        else:
            severity = "Severe"

        os.remove(tmp.name)

    # 🔥 Generate PDF here
    pdf_path = generate_pdf(
        name, age, gender, phone, email,
        pred_class, confidence,
        location, severity,
        img, superimposed
    )

    return pred_class, confidence, location, severity, superimposed, pdf_path


# ---------------- GRADIO UI ----------------
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="numpy"),
        gr.Text(label="Name"),
        gr.Number(label="Age"),
        gr.Text(label="Gender"),
        gr.Text(label="Phone"),
        gr.Text(label="Email"),
    ],
    outputs=[
        gr.Text(label="Prediction"),
        gr.Number(label="Confidence"),
        gr.Text(label="Location"),
        gr.Text(label="Severity"),
        gr.Image(label="GradCAM"),
        gr.File(label="Download PDF")   # 🔥 THIS IS THE BUTTON
    ],
    title="Brain Tumor Detection System",
    description="Upload MRI and get AI diagnosis report"
)

interface.launch(server_name="0.0.0.0", server_port=7860)