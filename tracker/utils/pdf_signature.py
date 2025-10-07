from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional

from PIL import Image
from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


class SignatureEmbedError(Exception):
    """Raised when a signature cannot be embedded into the provided PDF."""


def _scale_dimensions(
    page_width: float,
    page_height: float,
    image_width: int,
    image_height: int,
    max_width_ratio: float = 0.3,
    max_height_ratio: float = 0.2,
) -> Tuple[float, float]:
    """Compute scaled signature dimensions preserving aspect ratio within limits."""
    if image_width <= 0 or image_height <= 0:
        raise SignatureEmbedError("Signature image has invalid dimensions.")

    target_width = page_width * max_width_ratio
    target_height = page_height * max_height_ratio

    width_ratio = target_width / float(image_width)
    height_ratio = target_height / float(image_height)
    scale = min(width_ratio, height_ratio)

    scaled_width = float(image_width) * scale
    scaled_height = float(image_height) * scale

    return scaled_width, scaled_height


def embed_signature_in_pdf(
    pdf_bytes: bytes,
    signature_bytes: bytes,
    *,
    margin: float = 36.0,
    max_width_ratio: float = 0.3,
    max_height_ratio: float = 0.2,
) -> bytes:
    """Return a PDF with the signature image embedded on the final page."""
    if not pdf_bytes:
        raise SignatureEmbedError("No PDF content provided.")
    if not signature_bytes:
        raise SignatureEmbedError("No signature content provided.")

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as exc:  # pragma: no cover - defensive for malformed PDFs
        raise SignatureEmbedError("Could not read the provided PDF document.") from exc

    if len(reader.pages) == 0:
        raise SignatureEmbedError("The PDF has no pages to sign.")

    try:
        signature_image = Image.open(BytesIO(signature_bytes))
        signature_image = signature_image.convert("RGBA")
    except Exception as exc:  # pragma: no cover - defensive for malformed images
        raise SignatureEmbedError("Could not decode the signature image.") from exc

    last_page = reader.pages[-1]
    page_width = float(last_page.mediabox.width)
    page_height = float(last_page.mediabox.height)

    scaled_width, scaled_height = _scale_dimensions(
        page_width,
        page_height,
        signature_image.width,
        signature_image.height,
        max_width_ratio=max_width_ratio,
        max_height_ratio=max_height_ratio,
    )

    x_position = max(margin, page_width - scaled_width - margin)
    y_position = margin

    overlay_stream = BytesIO()
    signature_buffer = BytesIO()
    signature_image.save(signature_buffer, format="PNG")
    signature_buffer.seek(0)

    overlay_canvas = canvas.Canvas(overlay_stream, pagesize=(page_width, page_height))
    overlay_canvas.drawImage(
        ImageReader(signature_buffer),
        x_position,
        y_position,
        width=scaled_width,
        height=scaled_height,
        mask="auto",
    )
    overlay_canvas.save()
    overlay_stream.seek(0)

    overlay_reader = PdfReader(overlay_stream)
    overlay_page = overlay_reader.pages[0]

    writer = PdfWriter()
    total_pages = len(reader.pages)
    for index, page in enumerate(reader.pages):
        if index == total_pages - 1:
            page.merge_page(overlay_page)
        writer.add_page(page)

    output_stream = BytesIO()
    writer.write(output_stream)
    output_stream.seek(0)
    return output_stream.read()


def build_signed_filename(original_name: str, suffix: str = "signed") -> str:
    """Return a descriptive filename for the signed PDF."""
    base = Path(original_name or "document").stem or "document"
    return f"{base}-{suffix}.pdf"


def build_signed_name(original_name: str, suffix: str = "signed", preferred_ext: Optional[str] = None) -> str:
    """Return a descriptive filename preserving extension when possible.

    If preferred_ext is provided, it will be used (including the dot), otherwise
    the original file's extension is preserved. Falls back to .bin.
    """
    p = Path(original_name or "document")
    base = p.stem or "document"
    if preferred_ext:
        ext = preferred_ext if preferred_ext.startswith(".") else f".{preferred_ext}"
    else:
        ext = p.suffix or ".bin"
    return f"{base}-{suffix}{ext}"


def embed_signature_in_image(
    image_bytes: bytes,
    signature_bytes: bytes,
    *,
    margin: int = 12,
    max_width_ratio: float = 0.3,
    max_height_ratio: float = 0.2,
    output_format: Optional[str] = None,
) -> bytes:
    """Overlay signature onto the bottom-right of an image and return bytes.

    - Preserves aspect ratio for the signature.
    - Places the signature inside the image bounds with a small margin.
    - If output_format is not provided, use the original image format (or PNG if unknown).
    """
    if not image_bytes:
        raise SignatureEmbedError("No image content provided.")
    if not signature_bytes:
        raise SignatureEmbedError("No signature content provided.")

    try:
        base_img = Image.open(BytesIO(image_bytes))
    except Exception as exc:
        raise SignatureEmbedError("Could not read the provided image document.") from exc

    try:
        sig_img = Image.open(BytesIO(signature_bytes)).convert("RGBA")
    except Exception as exc:
        raise SignatureEmbedError("Could not decode the signature image.") from exc

    base_mode = base_img.mode
    base_format = (base_img.format or "").upper() or None
    base_img = base_img.convert("RGBA")

    page_w, page_h = float(base_img.width), float(base_img.height)
    scaled_w, scaled_h = _scale_dimensions(
        page_w, page_h, sig_img.width, sig_img.height,
        max_width_ratio=max_width_ratio, max_height_ratio=max_height_ratio,
    )

    # Resize signature maintaining aspect ratio
    sig_resized = sig_img.resize((int(max(1, scaled_w)), int(max(1, scaled_h))), Image.LANCZOS)

    x = max(margin, page_w - sig_resized.width - margin)
    y = max(margin, page_h - sig_resized.height - margin)

    composed = Image.new("RGBA", base_img.size)
    composed.paste(base_img, (0, 0))
    composed.paste(sig_resized, (int(x), int(y)), mask=sig_resized)

    # Convert back if original was not RGBA
    if base_mode != "RGBA":
        if base_mode in ("RGB", "L"):
            composed = composed.convert(base_mode)
        else:
            composed = composed.convert("RGB")
            base_format = base_format or "PNG"

    out = BytesIO()
    fmt = (output_format or base_format or "PNG").upper()
    if fmt == "JPG":
        fmt = "JPEG"
    composed.save(out, format=fmt)
    out.seek(0)
    return out.read()
