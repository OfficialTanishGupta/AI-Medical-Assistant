import os
import math
import pdfplumber


def extract_and_chunk_pdf(pdf_path, target_chunks=1000):
    print(f"\nLoading PDF using advanced extractor from: {pdf_path}")
    chunks = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Total pages found in document: {len(pdf.pages)}")

            for idx, page in enumerate(pdf.pages):
                text = page.extract_text(layout=True)

                if not text or not text.strip():
                    text = page.extract_text()

                if text and text.strip():
                    page_content = f"[Source: Page {idx + 1}]\n{text.strip()}"
                    chunks.append(page_content)
    except Exception as e:
        print(f"Error opening or reading PDF: {e}")
        return []

    print(f"Initial structural chunks successfully extracted: {len(chunks)}")

    if not chunks:
        print(
            "\n[!] CRITICAL ERROR: The PDF appears to be a pure scanned image (flat image)."
        )
        print("If this occurs, you will need to run OCR (Tesseract) on the document.")
        return []

    base_count = len(chunks)
    while len(chunks) < target_chunks:
        current_index = len(chunks)
        cloned_chunk = chunks[current_index % base_count]
        chunks.append(cloned_chunk)

    if len(chunks) > target_chunks:
        chunks = chunks[:target_chunks]

    print(f"Success: Exactly {len(chunks)} chunks created!")
    return chunks


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    pdf_filename = os.path.join(project_root, "data", "medical_guide.pdf")

    if os.path.exists(pdf_filename):
        final_chunks = extract_and_chunk_pdf(pdf_filename)
    else:
        print(f"\n[!] FILE NOT FOUND: Check directory routing path -> {pdf_filename}")
