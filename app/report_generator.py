import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable
)


def _safe(value):
    return str(value).replace("_", " ").title()


def generate_screening_report(
    output_path,
    image_path,
    gradcam_path,
    quality,
    brightness,
    sharpness,
    prediction,
    score,
    guidance,
    analytics=None,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    report_time = datetime.now().strftime("%d %b %Y, %I:%M %p")
    report_id = datetime.now().strftime("OSE-%Y%m%d-%H%M%S")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="ORAL-SENSE EDGE Screening Report",
        author="ORAL-SENSE EDGE",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleOS",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#12304A"),
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "SubtitleOS",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#617789"),
        spaceAfter=8,
    )
    section_style = ParagraphStyle(
        "SectionOS",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#12304A"),
        spaceBefore=8,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyOS",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334B5C"),
    )
    small_style = ParagraphStyle(
        "SmallOS",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#667A89"),
    )
    warning_style = ParagraphStyle(
        "WarningOS",
        parent=body_style,
        textColor=colors.HexColor("#7A4A00"),
    )

    story = []

    story.append(Paragraph("ORAL-SENSE EDGE", title_style))
    story.append(Paragraph("AI-Assisted Oral Screening Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#B8C8D4")))
    story.append(Spacer(1, 7))

    meta = [
        [Paragraph("Report ID", body_style), Paragraph(report_id, body_style),
         Paragraph("Generated", body_style), Paragraph(report_time, body_style)],
        [Paragraph("Processing", body_style), Paragraph("Local / Offline", body_style),
         Paragraph("Model", body_style), Paragraph("MobileNetV3 Small", body_style)],
    ]
    meta_table = Table(meta, colWidths=[25*mm, 52*mm, 25*mm, 70*mm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#F4F7F9")),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#D7E0E6")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 7),
        ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)

    story.append(Paragraph("1. Screening Result", section_style))
    result_text = _safe(prediction)
    result_data = [
        [Paragraph("Model output", body_style), Paragraph(result_text, body_style)],
        [Paragraph("Model score", body_style), Paragraph(f"{float(score):.2f}%", body_style)],
        [Paragraph("Image quality", body_style), Paragraph(str(quality), body_style)],
        [Paragraph("Brightness", body_style), Paragraph(f"{float(brightness):.2f}", body_style)],
        [Paragraph("Sharpness", body_style), Paragraph(f"{float(sharpness):.2f}", body_style)],
    ]
    result_table = Table(result_data, colWidths=[50*mm, 122*mm])
    result_table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#D7E0E6")),
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#F4F7F9")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 7),
        ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(result_table)

    story.append(Paragraph("2. Image Review", section_style))
    if image_path and os.path.exists(image_path):
        img = RLImage(image_path, width=78*mm, height=58*mm)
        img.hAlign = "LEFT"
        story.append(img)
        story.append(Spacer(1, 4))
    story.append(Paragraph(
        "The image quality values above are technical image measurements used by the prototype. "
        "They do not establish clinical image adequacy.", small_style
    ))

    story.append(Paragraph("3. Model Attention Map", section_style))
    if gradcam_path and os.path.exists(gradcam_path):
        cam = RLImage(gradcam_path, width=110*mm, height=72*mm)
        cam.hAlign = "LEFT"
        story.append(cam)
        story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Grad-CAM highlights regions that influenced the model output. It is not a confirmed lesion boundary "
        "or a clinical localization tool.", small_style
    ))

    story.append(Paragraph("4. Screening Guidance", section_style))
    guidance_data = [
        [Paragraph("Status", body_style), Paragraph(str(guidance.get("level", "Professional review")), body_style)],
        [Paragraph("Guidance", body_style), Paragraph(str(guidance.get("guidance", "Consider professional examination for further evaluation.")), body_style)],
    ]
    guidance_table = Table(guidance_data, colWidths=[50*mm, 122*mm])
    guidance_table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#D7E0E6")),
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#F4F7F9")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 7),
        ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(guidance_table)

    if analytics:
        story.append(Paragraph("5. Local Screening History Summary", section_style))
        story.append(Paragraph(
            "These values summarize saved model outputs in this prototype's local history; they are not clinical statistics.",
            small_style,
        ))
        analytics_data = [
            [Paragraph("Total screenings", body_style), Paragraph(str(analytics.get("total", 0)), body_style)],
            [Paragraph("Model: benign", body_style), Paragraph(str(analytics.get("benign", 0)), body_style)],
            [Paragraph("Model: malignant", body_style), Paragraph(str(analytics.get("malignant", 0)), body_style)],
            [Paragraph("Good images", body_style), Paragraph(str(analytics.get("good_images", 0)), body_style)],
            [Paragraph("Follow-up flags", body_style), Paragraph(str(analytics.get("follow_up", 0)), body_style)],
        ]
        analytics_table = Table(analytics_data, colWidths=[50*mm, 122*mm])
        analytics_table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#D7E0E6")),
            ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#F4F7F9")),
            ("LEFTPADDING", (0,0), (-1,-1), 7),
            ("RIGHTPADDING", (0,0), (-1,-1), 7),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(analytics_table)

    story.append(Paragraph("Important Safety & Interpretation Note", section_style))
    story.append(Paragraph(
        "ORAL-SENSE EDGE is a student research prototype for screening assistance. It does not provide a medical diagnosis, "
        "does not replace examination by a qualified dental or medical professional, and should not be used as the sole basis "
        "for treatment decisions. The model score is a prediction confidence, not clinical certainty. Model performance shown "
        "in the project has not been established as clinical diagnostic performance. Change Monitoring is visual image comparison "
        "only and can be affected by lighting, camera angle, framing and other capture conditions.",
        warning_style,
    ))

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Generated locally by ORAL-SENSE EDGE • Privacy-first prototype",
        small_style,
    ))

    def add_page_number(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#71818C"))
        canvas.drawString(16 * mm, 8 * mm, "ORAL-SENSE EDGE • Screening assistance only")
        canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, f"Page {doc_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return output_path
