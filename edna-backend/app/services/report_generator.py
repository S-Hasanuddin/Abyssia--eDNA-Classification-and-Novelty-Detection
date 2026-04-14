"""
report_generator.py — PDF biodiversity report generator
==========================================================
Generates a comprehensive PDF report including:
  - Summary statistics
  - Individual classification scores
  - Novel taxa clusters with novelty scores
  - Ecosystem diversity indices & dominant taxa
  - Historical cluster comparison
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)

from app.config import REPORT_DIR, RANKS

logger = logging.getLogger(__name__)

# Colour palette
BLACK = colors.HexColor("#000000")
DARK_GRAY = colors.HexColor("#333333")
LIGHT_GRAY = colors.HexColor("#cccccc")
WHITE = colors.HexColor("#ffffff")
TABLE_BORDER = colors.HexColor("#999999")


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        'ReportTitle', parent=styles['Title'],
        fontSize=22, textColor=BLACK, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        'SectionHead', parent=styles['Heading2'],
        fontSize=14, textColor=BLACK, spaceBefore=18, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        'SubHead', parent=styles['Heading3'],
        fontSize=11, textColor=BLACK, spaceBefore=10, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontSize=9, textColor=BLACK, leading=13,
    ))
    styles.add(ParagraphStyle(
        'Small', parent=styles['Normal'],
        fontSize=8, textColor=DARK_GRAY, leading=10,
    ))
    return styles


def _summary_table(summary: Dict, styles) -> Table:
    data = [
        ["Metric", "Value"],
        ["Total Sequences", f"{summary['total_sequences']:,}"],
        ["Classified (high confidence)", f"{summary['classified']:,}"],
        ["Novel Candidates (low confidence)", f"{summary['novel_candidates']:,}"],
        ["Novel Clusters Found", f"{summary['novel_clusters']:,}"],
        ["Confidence Threshold", f"{summary['confidence_threshold']}"],
    ]
    t = Table(data, colWidths=[3 * inch, 2.5 * inch])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (-1, -1), BLACK),
        ('GRID', (0, 0), (-1, -1), 0.5, TABLE_BORDER),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return t


def _classification_table(classified: List[Dict], styles, max_rows=50) -> Table:
    data = [["ID", "Rank", "Taxon", "Confidence", "Fallback"]]
    for r in classified[:max_rows]:
        data.append([
            r['id'][:25],
            r['reported_rank'].capitalize(),
            r['reported_taxon'][:30],
            f"{r['reported_confidence']:.2%}",
            "Yes" if r['fallback_occurred'] else "No",
        ])
    if len(classified) > max_rows:
        data.append(["", "", f"... +{len(classified) - max_rows} more", "", ""])

    t = Table(data, colWidths=[1.5 * inch, 0.8 * inch, 1.8 * inch, 0.9 * inch, 0.7 * inch])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TEXTCOLOR', (0, 0), (-1, -1), BLACK),
        ('GRID', (0, 0), (-1, -1), 0.4, TABLE_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return t


def _novel_taxa_table(novel_taxa: Dict, styles) -> Table:
    clusters = novel_taxa.get('clusters', {})
    data = [["Cluster", "Size", "Rank", "Taxon", "Confidence", "Novelty"]]
    for cname, cinfo in list(clusters.items())[:40]:
        data.append([
            cname,
            str(cinfo['size']),
            cinfo.get('reported_rank', '?').capitalize(),
            cinfo.get('reported_taxon', '?')[:25],
            f"{cinfo.get('reported_confidence', 0):.2%}",
            f"{cinfo.get('novelty_score', 0):.2%}",
        ])
    if len(clusters) > 40:
        data.append(["", "", "", f"... +{len(clusters) - 40} more", "", ""])

    t = Table(data, colWidths=[0.9 * inch, 0.5 * inch, 0.7 * inch, 1.6 * inch, 0.8 * inch, 0.7 * inch])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TEXTCOLOR', (0, 0), (-1, -1), BLACK),
        ('GRID', (0, 0), (-1, -1), 0.4, TABLE_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return t


def _ecosystem_table(ecosystem: Dict, styles) -> Table:
    dm = ecosystem.get('diversity_metrics', {})
    data = [["Rank", "Shannon", "Simpson", "Richness", "Evenness"]]
    for rank in RANKS:
        data.append([
            rank.capitalize(),
            f"{dm.get(f'{rank}_shannon', 0):.4f}",
            f"{dm.get(f'{rank}_simpson', 0):.4f}",
            str(dm.get(f'{rank}_richness', 0)),
            f"{dm.get(f'{rank}_evenness', 0):.4f}",
        ])
    t = Table(data, colWidths=[1.0 * inch, 1.0 * inch, 1.0 * inch, 0.9 * inch, 1.0 * inch])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (-1, -1), BLACK),
        ('GRID', (0, 0), (-1, -1), 0.4, TABLE_BORDER),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return t


def _dominant_taxa_section(ecosystem: Dict, styles) -> list:
    elements = []
    dominant = ecosystem.get('dominant_taxa', {})
    for rank in RANKS:
        dom_list = dominant.get(rank, [])
        if not dom_list:
            continue
        elements.append(Paragraph(
            f"<b>{rank.capitalize()}</b> — Top taxa by abundance:",
            styles['SubHead']))
        data = [["Taxon", "Count", "Relative Abundance"]]
        for d in dom_list:
            data.append([
                d['taxon'][:35],
                str(d['count']),
                f"{d['relative_abundance']:.2%}",
            ])
        t = Table(data, colWidths=[2.5 * inch, 1.0 * inch, 1.5 * inch])
        t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('TEXTCOLOR', (0, 0), (-1, -1), BLACK),
            ('GRID', (0, 0), (-1, -1), 0.3, TABLE_BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 6))
    return elements


def _historical_section(comparison: Dict, styles) -> list:
    elements = []
    matches = comparison.get('matches', [])
    if not matches:
        elements.append(Paragraph(
            "No similar clusters found in historical database.",
            styles['Body']))
        return elements

    elements.append(Paragraph(
        f"Found {len(matches)} cluster matches above "
        f"{comparison.get('similarity_threshold', 0.85):.0%} cosine similarity "
        f"(compared against {comparison.get('total_historical', 0)} historical clusters).",
        styles['Body']))
    elements.append(Spacer(1, 6))

    data = [["Current Cluster", "Historical Match", "Job ID", "Similarity"]]
    for m in matches[:20]:
        data.append([
            f"{m['current_cluster']} ({m['current_taxon'][:20]})",
            f"{m['historical_cluster_id']} ({m['historical_taxon'][:20]})",
            m['historical_job_id'],
            f"{m['cosine_similarity']:.2%}",
        ])

    t = Table(data, colWidths=[1.8 * inch, 1.8 * inch, 0.9 * inch, 0.8 * inch])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TEXTCOLOR', (0, 0), (-1, -1), BLACK),
        ('GRID', (0, 0), (-1, -1), 0.4, TABLE_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(t)
    return elements


def generate_pdf_report(
    job_id: str,
    result: Dict,
    historical_comparison: Optional[Dict] = None,
    filename: str = "unknown",
) -> str:
    """
    Generate a comprehensive PDF report and return its file path.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = str(REPORT_DIR / f"edna_report_{job_id}.pdf")
    styles = _build_styles()

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
    )

    story = []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # ── Title ────────────────────────────────────────────────────────────
    story.append(Paragraph(
        "eDNA Biodiversity Analysis Report", styles['ReportTitle']))
    story.append(Paragraph(
        f"Generated: {now} | Input: {filename} | Job: {job_id}",
        styles['Small']))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(
        width="100%", thickness=1, color=BLACK, spaceAfter=12))

    # ── Summary ──────────────────────────────────────────────────────────
    summary = result.get('summary', {})
    story.append(Paragraph("1. Analysis Summary", styles['SectionHead']))
    story.append(_summary_table(summary, styles))
    story.append(Spacer(1, 12))

    # ── Classification Results ───────────────────────────────────────────
    classified = result.get('classified', [])
    if classified:
        story.append(Paragraph(
            f"2. Classification Results ({len(classified)} sequences)",
            styles['SectionHead']))
        story.append(Paragraph(
            "High-confidence sequences routed through the bottom-up "
            "cascaded classifier (Family -> Domain). Showing taxonomy at "
            "the most specific confident rank.",
            styles['Body']))
        story.append(Spacer(1, 6))
        story.append(_classification_table(classified, styles))
        story.append(Spacer(1, 12))

    # ── Novel Taxa ───────────────────────────────────────────────────────
    novel_taxa = result.get('novel_taxa')
    if novel_taxa and novel_taxa.get('n_clusters', 0) > 0:
        story.append(PageBreak())
        story.append(Paragraph(
            f"3. Novel Taxa Detection ({novel_taxa['n_clusters']} clusters)",
            styles['SectionHead']))
        story.append(Paragraph(
            f"Low-confidence sequences were clustered using HDBSCAN. "
            f"Noise points: {novel_taxa['noise_count']} "
            f"({novel_taxa['noise_fraction']:.1%}). "
            f"Novelty scores represent cosine distance from nearest "
            f"learned prototype (higher = more novel).",
            styles['Body']))
        story.append(Spacer(1, 6))
        story.append(_novel_taxa_table(novel_taxa, styles))

        # Noise fallback distribution
        nrd = novel_taxa.get('noise_rank_dist', {})
        if nrd:
            story.append(Spacer(1, 8))
            story.append(Paragraph(
                "Noise point fallback rank distribution:", styles['SubHead']))
            noise_data = [["Rank", "Count"]]
            for rank in RANKS:
                if rank in nrd:
                    noise_data.append([rank.capitalize(), str(nrd[rank])])
            nt = Table(noise_data, colWidths=[1.5 * inch, 1.0 * inch])
            nt.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('TEXTCOLOR', (0, 0), (-1, -1), BLACK),
                ('GRID', (0, 0), (-1, -1), 0.4, TABLE_BORDER),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(nt)

    # ── Ecosystem Analysis ───────────────────────────────────────────────
    ecosystem = result.get('ecosystem')
    if ecosystem:
        story.append(PageBreak())
        story.append(Paragraph(
            "4. Ecosystem Diversity Analysis", styles['SectionHead']))
        story.append(Paragraph(
            "Biodiversity indices computed per taxonomic rank from "
            "classifier predictions. Shannon index measures entropy of "
            "species distribution; Simpson measures dominance; Evenness "
            "normalizes Shannon by maximum possible diversity.",
            styles['Body']))
        story.append(Spacer(1, 6))
        story.append(_ecosystem_table(ecosystem, styles))
        story.append(Spacer(1, 12))

        # Dominant taxa
        story.append(Paragraph(
            "4.1 Dominant Taxa by Rank", styles['SectionHead']))
        story.extend(_dominant_taxa_section(ecosystem, styles))

    # ── Historical Comparison ────────────────────────────────────────────
    if historical_comparison:
        story.append(PageBreak())
        story.append(Paragraph(
            "5. Historical Cluster Comparison", styles['SectionHead']))
        story.append(Paragraph(
            "Novel clusters from this analysis were compared against "
            "clusters from all previous analyses using cosine similarity "
            "of cluster centroids in the 256-dim projection space.",
            styles['Body']))
        story.append(Spacer(1, 6))
        story.extend(_historical_section(historical_comparison, styles))

    # ── Footer ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 24))
    story.append(HRFlowable(
        width="100%", thickness=0.5, color=LIGHT_GRAY, spaceAfter=6))
    story.append(Paragraph(
        "Generated by Abyssia — Deep-Sea eDNA Biodiversity Pipeline | "
        "BERT-style Transformer (96M params) | "
        "Methodist College of Engineering and Technology",
        styles['Small']))

    doc.build(story)
    logger.info(f"PDF report generated: {pdf_path}")
    return pdf_path