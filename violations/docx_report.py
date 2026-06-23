"""
docx_report -- generate the per-speeding-violation ``.docx`` report with ZERO third-party deps.

The backend contract asks, for a SPEEDING violation, for a Word document describing the 10-second
session and the actual calculated speed. A ``.docx`` is just a ZIP of a few XML parts (Open Packaging
Conventions / WordprocessingML), so we build a minimal, VALID one with the stdlib ``zipfile`` alone --
no ``python-docx`` to install, which also keeps the export pipeline unit-testable on a bare
interpreter (same philosophy as the rest of :mod:`violations`).

Public API:
  * :func:`build_docx(title, paragraphs)`  -- generic: title + list of (text, bold) paragraphs -> bytes
  * :func:`speeding_report(event, ...)`     -- the speeding-specific document -> bytes

The document opens in Word / LibreOffice / Google Docs. Headings are rendered as BOLD runs (rather
than referencing a styles part we would otherwise have to ship), keeping the package to three parts.
"""
from __future__ import annotations

import io
import zipfile
from typing import Any, Optional, Sequence
from xml.sax.saxutils import escape

# The three parts of a minimal WordprocessingML package.
_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    '</Types>'
)
_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
    'Target="word/document.xml"/>'
    '</Relationships>'
)
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _paragraph_xml(text: str, *, bold: bool = False, size_half_pt: Optional[int] = None) -> str:
    """One ``<w:p>`` paragraph. ``size_half_pt`` is Word's half-point unit (e.g. 28 == 14pt)."""
    rpr_bits = ""
    if bold:
        rpr_bits += "<w:b/>"
    if size_half_pt:
        rpr_bits += f'<w:sz w:val="{int(size_half_pt)}"/>'
    rpr = f"<w:rPr>{rpr_bits}</w:rPr>" if rpr_bits else ""
    body = escape(text or "")
    return (f"<w:p><w:r>{rpr}"
            f'<w:t xml:space="preserve">{body}</w:t></w:r></w:p>')


def _document_xml(title: str, paragraphs: Sequence[tuple]) -> str:
    parts = [f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             f'<w:document xmlns:w="{_W_NS}"><w:body>']
    parts.append(_paragraph_xml(title, bold=True, size_half_pt=32))      # 16pt bold title
    for text, bold in paragraphs:
        parts.append(_paragraph_xml(text, bold=bold))
    parts.append('<w:sectPr/></w:body></w:document>')
    return "".join(parts)


def build_docx(title: str, paragraphs: Sequence[tuple]) -> bytes:
    """Build a minimal valid ``.docx`` -> bytes. ``paragraphs`` is a list of ``(text, is_bold)``."""
    doc_xml = _document_xml(title, paragraphs)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        # [Content_Types].xml must be the first entry per the OPC spec; order otherwise free.
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("word/document.xml", doc_xml)
    return buf.getvalue()


def _fmt_speed(v: Any) -> str:
    try:
        return f"{float(v):.1f} km/h"
    except (TypeError, ValueError):
        return "unknown"


def speeding_report(event, *, video_meta: Optional[dict] = None,
                    window: Optional[dict] = None) -> bytes:
    """The SPEEDING ``.docx``: vehicle, the calculated speed vs the limit, the margin, and the
    10-second session window. Reads ``event.details`` (est_speed_kmh / speed_limit_kmh /
    over_by_kmh); tolerant of missing fields (recall-first: still emit a document)."""
    d = dict(getattr(event, "details", None) or {})
    est = d.get("est_speed_kmh")
    lim = d.get("speed_limit_kmh")
    over = d.get("over_by_kmh")
    if over is None and est is not None and lim is not None:
        over = est - lim

    paras: list[tuple] = [
        ("Road Guard -- Speeding Violation Report", True),
        ("", False),
        (f"Vehicle ID: {getattr(event, 'vehicle_id', 'n/a')}", False),
        (f"Violation type: SPEEDING", False),
        (f"Key frame: {getattr(event, 'key_frame', 'n/a')}", False),
        ("", False),
        ("Measured speed", True),
        (f"  Estimated speed: {_fmt_speed(est)}", False),
        (f"  Posted limit:    {_fmt_speed(lim)}", False),
        (f"  Over the limit by: {_fmt_speed(over) if over is not None else 'unknown'}", False),
    ]

    if window:
        paras += [
            ("", False),
            ("Evidence session (10s before / 5s after the incident)", True),
            (f"  Frames: {window.get('start_frame')} - {window.get('end_frame')} "
             f"(key frame {window.get('key_frame')})", False),
            (f"  Time:   {window.get('start_sec')}s - {window.get('end_sec')}s "
             f"({window.get('duration_sec')}s total)", False),
        ]

    if video_meta:
        paras += [
            ("", False),
            ("Source video", True),
            (f"  File: {video_meta.get('filename')}", False),
            (f"  Resolution: {video_meta.get('width')}x{video_meta.get('height')} "
             f"@ {video_meta.get('fps')} fps", False),
        ]

    paras += [
        ("", False),
        ("Note: speed is a monocular estimate; a small margin over the limit carries lower "
         "confidence and is routed to human review.", False),
    ]
    return build_docx("Speeding Violation Report", paras)
