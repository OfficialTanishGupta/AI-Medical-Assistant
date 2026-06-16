import os
import pytesseract
from pdf2image import convert_from_path
import pdfplumber

# If Tesseract isn't on PATH, point to it explicitly
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

POPPLER_PATH = (
    r"C:\Program Files\poppler\poppler-26.02.0\Library\bin"  # adjust if needed
)


def ocr_pdf(pdf_path, lang="eng"):
    """Convert each page to an image and run Tesseract OCR on it."""
    print(f"\n[OCR] Converting pages to images...")
    try:
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
    except Exception as e:
        print(f"[!] pdf2image failed: {e}")
        print("    Make sure Poppler is installed and POPPLER_PATH is correct.")
        return []

    print(f"[OCR] Running Tesseract on {len(pages)} pages...")
    page_texts = []
    for i, img in enumerate(pages):
        text = pytesseract.image_to_string(img, lang=lang)
        if text.strip():
            page_texts.append((i + 1, text.strip()))
            print(f"    Page {i+1}: {len(text)} chars extracted")
        else:
            print(f"    Page {i+1}: (empty)")

    return page_texts


def try_pdfplumber(pdf_path):
    """Try native text extraction first (fast path for text-based PDFs)."""
    page_texts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text(layout=True) or page.extract_text()
                if text and text.strip():
                    page_texts.append((i + 1, text.strip()))
    except Exception as e:
        print(f"[pdfplumber] Error: {e}")
    return page_texts


def chunk_text(page_texts, chunk_size=500, overlap=50):
    """Sliding window chunker with sentence-boundary snapping."""
    merged = "\n\n".join(f"[Page {p}]\n{t}" for p, t in page_texts)

    chunks = []
    start = 0
    total = len(merged)

    while start < total:
        end = min(start + chunk_size, total)
        if end < total:
            boundary = merged.rfind(".", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = merged[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def extract_and_chunk_pdf(pdf_path, chunk_size=500, overlap=50):
    print(f"\nLoading PDF: {pdf_path}")

    page_texts = try_pdfplumber(pdf_path)

    if not page_texts:
        print("[!] No native text found — falling back to OCR...")
        page_texts = ocr_pdf(pdf_path)

    if not page_texts:
        print("[!] OCR also produced no text. Check the PDF and Tesseract install.")
        return []

    # 3. Chunk
    chunks = chunk_text(page_texts, chunk_size, overlap)
    print(f"\n✓ {len(page_texts)} pages processed → {len(chunks)} chunks created.")
    return chunks


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    pdf_path = os.path.join(project_root, "data", "medical_guide.pdf")

    if not os.path.exists(pdf_path):
        print(f"[!] File not found: {pdf_path}")
    else:
        chunks = extract_and_chunk_pdf(pdf_path)
        if chunks:
            print(f"\nSample chunk:\n{'─'*50}\n{chunks[0]}\n{'─'*50}")
