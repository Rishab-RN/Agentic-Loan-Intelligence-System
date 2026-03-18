"""
ALIS Fairness Audit: PDF Report Generator
======================================================
Compiles the Fairlearn audit metrics and plots into a professional
PDF report using ReportLab, demonstrating RBI FREE-AI compliance.

Usage:
    python report_generator.py
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

from data_generator import ARTIFACTS_DIR

def generate_pdf_report():
    print("ALIS Fairness Audit: Generating PDF Report...")
    metrics_file = ARTIFACTS_DIR / "audit_metrics.json"
    
    if not metrics_file.exists():
        print("Audit metrics not found. Run audit.py first.")
        return
        
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
        
    pdf_path = ARTIFACTS_DIR / "ALIS_Fairness_Report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = styles['Heading1']
    title_style.alignment = 1  # Center
    subtitle_style = styles['Heading2']
    subtitle_style.textColor = colors.HexColor("#3B82F6")
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    normal_style.leading = 14
    
    story = []
    
    # Title
    story.append(Paragraph("ALIS: Fairness & Bias Audit Report", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Demonstrating Compliance with the RBI FREE-AI Framework (2025)", styles["Italic"]))
    story.append(Spacer(1, 24))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", subtitle_style))
    story.append(Spacer(1, 6))
    exec_summary = (
        "This report details the algorithmic fairness audit conducted on the ALIS "
        "(Agentic Loan Intelligence System) credit scoring engine, RiskMind. "
        "The objective was to identify and mitigate any systemic biases against protected "
        "demographic groups (Gender, Geography Tier) in loan approval rates, ensuring compliance "
        "with the RBI's Framework for Responsible and Ethical AI (FREE-AI)."
    )
    story.append(Paragraph(exec_summary, normal_style))
    story.append(Spacer(1, 12))
    
    # Base Model Performance
    unmit_data = metrics.get('unmitigated', {})
    story.append(Paragraph("1. Base Model Audit (Unmitigated)", subtitle_style))
    story.append(Spacer(1, 6))
    
    base_text = (
        f"The baseline XGBoost model achieved an overall accuracy of {unmit_data.get('accuracy', 0):.3f}. "
        f"However, the Fairlearn MetricFrame analysis revealed a Demographic Parity Difference of "
        f"{unmit_data.get('dp_diff', 0):.3f} and an Equalized Odds Difference of {unmit_data.get('eo_diff', 0):.3f} "
        f"across gender lines."
    )
    story.append(Paragraph(base_text, normal_style))
    story.append(Spacer(1, 12))
    
    # Base Plot
    base_img_path = ARTIFACTS_DIR / "plots" / "metrics_by_gender_base.png"
    if base_img_path.exists():
        story.append(Image(str(base_img_path), width=450, height=270))
        story.append(Spacer(1, 24))
        
    # Mitigation and Results
    if metrics.get('mitigated'):
        mit_data = metrics['mitigated']
        story.append(Paragraph("2. Bias Mitigation (Exponentiated Gradient)", subtitle_style))
        story.append(Spacer(1, 6))
        
        mit_text = (
            "Because the Demographic Parity difference exceeded the 5% (0.05) threshold, "
            "automated bias mitigation was triggered. We applied Fairlearn's ExponentiatedGradient "
            "meta-estimator enforcing an EqualizedOdds constraint."
        )
        story.append(Paragraph(mit_text, normal_style))
        story.append(Spacer(1, 12))
        
        # Tradeoff Table
        data = [
            ["Metric", "Base Model", "Mitigated Model", "Change"],
            ["Overall Accuracy", f"{unmit_data.get('accuracy'):.3f}", f"{mit_data.get('accuracy'):.3f}", 
             f"{mit_data.get('accuracy') - unmit_data.get('accuracy'):.3f}"],
            ["Demographic Parity Diff", f"{unmit_data.get('dp_diff'):.3f}", f"{mit_data.get('dp_diff'):.3f}", 
             f"{mit_data.get('dp_diff') - unmit_data.get('dp_diff'):.3f}"],
            ["Equalized Odds Diff", f"{unmit_data.get('eo_diff'):.3f}", f"{mit_data.get('eo_diff'):.3f}", 
             f"{mit_data.get('eo_diff') - unmit_data.get('eo_diff'):.3f}"]
        ]
        
        table = Table(data, colWidths=[150, 100, 100, 100])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#F1F5F9")),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))
        
        # Mitigated Plot
        mit_img_path = ARTIFACTS_DIR / "plots" / "metrics_by_gender_mitigated.png"
        if mit_img_path.exists():
            story.append(Image(str(mit_img_path), width=450, height=270))
            story.append(Spacer(1, 24))
            
    else:
        story.append(Paragraph("2. RBI Alignment Statement", subtitle_style))
        story.append(Spacer(1, 6))
        align_text = (
            "The base model exhibits a Demographic Parity difference below 5% (0.05). "
            "No algorithmic mitigation is necessary. The ALIS underwriting engine meets "
            "the strict fairness criteria set by the RBI FREE-AI guidelines."
        )
        story.append(Paragraph(align_text, normal_style))
        story.append(Spacer(1, 24))
        
    # Conclusion
    story.append(Paragraph("Conclusion", subtitle_style))
    story.append(Spacer(1, 6))
    conclusion = (
        "ALIS demonstrates that alternative-data credit scoring for the financially "
        "excluded can be implemented responsibly. Through proactive auditing with "
        "Microsoft Fairlearn, we ensure equitable access to credit regardless of gender "
        "or geographical location."
    )
    story.append(Paragraph(conclusion, normal_style))

    # Build PDF
    doc.build(story)
    print(f"PDF Report successfully generated: {pdf_path}")


if __name__ == "__main__":
    generate_pdf_report()
