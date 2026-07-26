from datetime import datetime
import html

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _esc(text: str) -> str:
    return html.escape(text or "").replace("\n", "<br/>")


def _page_bg(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(colors.HexColor("#ffffff"))
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setStrokeColor(colors.HexColor("#e2e8f0"))
    canvas.setLineWidth(0.8)
    canvas.line(36, height - 28, width - 36, height - 28)
    canvas.restoreState()


def generate_report(
    filename,
    prediction,
    confidence,
    original_text,
    cleaned_text,
    word_count,
    character_count,
    sentence_count,
    average_words,
    reading_time,
):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=34,
        bottomMargin=34,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Header",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=10,
            textColor=colors.HexColor("#64748b"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="TitleSoft",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=18,
            textColor=colors.HexColor("#0f172a"),
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubSoft",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#64748b"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionSoft",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=14,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodySoft",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#334155"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="LabelSoft",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#475569"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetricValue",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=16,
            textColor=colors.HexColor("#1e40af"),
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetricLabel",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10,
            textColor=colors.HexColor("#64748b"),
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="FooterSoft",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#64748b"),
        )
    )

    is_real = prediction.lower().startswith("real")
    verdict_bg = colors.HexColor("#dcfce7") if is_real else colors.HexColor("#fee2e2")
    verdict_fg = colors.HexColor("#166534") if is_real else colors.HexColor("#991b1b")

    verdict_style = ParagraphStyle(
        name="VerdictDynamic",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=13,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_CENTER,
    )

    original_text = _esc(original_text)
    cleaned_text = _esc(cleaned_text)

    def metric_card(label, value, tint):
        return Table(
            [[Paragraph(str(value), styles["MetricValue"])], [Paragraph(label, styles["MetricLabel"])]],
            colWidths=[1.18 * 72],
            rowHeights=[22, 18],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                    ("BOX", (0, 0), (-1, -1), 1, tint),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            ),
        )

    def text_box(text, bg="#ffffff"):
        return Table(
            [[Paragraph(text, styles["BodySoft"])]],
            colWidths=[6.8 * 72],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ]
            ),
        )

    story = []
    story.append(Paragraph("Truth Lens AI Report", styles["Header"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Fake News Detection Report", styles["TitleSoft"]))
    story.append(Paragraph("A concise summary of the prediction, confidence, and text details.", styles["SubSoft"]))
    story.append(Spacer(1, 12))

    story.append(
        Table(
            [[Paragraph(f'<font color="#0f172a"><b>{prediction} | Confidence {confidence:.2f}%</b></font>', verdict_style)]],
            colWidths=[6.8 * 72],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), verdict_bg),
                    ("BOX", (0, 0), (-1, -1), 1.2, verdict_fg),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]
            ),
        )
    )
    story.append(Spacer(1, 12))

    metrics = Table(
        [[
            metric_card("Words", word_count, colors.HexColor("#93c5fd")),
            metric_card("Characters", character_count, colors.HexColor("#c4b5fd")),
            metric_card("Sentences", sentence_count, colors.HexColor("#86efac")),
            metric_card("Avg Words/Sentence", f"{average_words:.1f}", colors.HexColor("#fde68a")),
            metric_card("Reading Time", f"{reading_time:.1f} min", colors.HexColor("#f9a8d4")),
        ]],
        colWidths=[1.32 * 72] * 5,
        style=TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        ),
    )
    story.append(metrics)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Article Overview", styles["SectionSoft"]))
    overview = Table(
        [
            [Paragraph("Prediction", styles["LabelSoft"]), Paragraph(prediction, styles["BodySoft"])],
            [Paragraph("Confidence", styles["LabelSoft"]), Paragraph(f"{confidence:.2f}%", styles["BodySoft"])],
            [Paragraph("Generated At", styles["LabelSoft"]), Paragraph(datetime.now().strftime("%d-%m-%Y %H:%M:%S"), styles["BodySoft"])],
        ],
        colWidths=[1.4 * 72, 5.4 * 72],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        ),
    )
    story.append(overview)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Original Article", styles["SectionSoft"]))
    story.append(text_box(original_text, bg="#ffffff"))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Cleaned Text Used by the Model", styles["SectionSoft"]))
    story.append(text_box(cleaned_text, bg="#f8fafc"))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Generated using Fake News Detection AI", styles["FooterSoft"]))
    story.append(Paragraph("Developed by Charmi Khunt", styles["FooterSoft"]))

    doc.build(story, onFirstPage=_page_bg, onLaterPages=_page_bg)
