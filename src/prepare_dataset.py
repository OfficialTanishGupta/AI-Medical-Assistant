import os
import sys
import json
import ollama

# Set up clean project root reference
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.ingest import extract_and_chunk_pdf


def generate_medical_pairs(chunk_text):
    """Uses local Llama 3 to convert a raw PDF chunk into structured instruction training data."""
    prompt = f"""You are a medical data engineer. Read the following medical document snippet and generate exactly 2 realistic medical questions and highly detailed clinical answers based ONLY on the text provided.

Format your output strictly as a JSON list of objects containing "instruction" (the question) and "output" (the answer). Do not include any explanation, intro, or markdown blocks.

Document Snippet:
{chunk_text}

JSON Output Structure:
[
  {{"instruction": "User medical question here...", "output": "Detailed clinical answer here..."}},
  {{"instruction": "Another medical question here...", "output": "Another detailed clinical answer here..."}}
]"""

    try:
        response = ollama.generate(
            model="llama3", prompt=prompt, options={"temperature": 0.3}
        )
        raw_output = response["response"].strip()
        if raw_output.startswith("```json"):
            raw_output = raw_output.replace("```json", "").replace("```", "").strip()
        return json.loads(raw_output)
    except Exception as e:
        print(f"[!] Error processing chunk: {e}")
        return []


def build_dataset():
    pdf_path = os.path.join(project_root, "data", "medical_guide.pdf")
    output_json_path = os.path.join(
        project_root, "data", "medical_training_dataset.json"
    )

    print("[1/3] Extracting text chunks from medical guide...")
    chunks = extract_and_chunk_pdf(pdf_path, chunk_size=500, overlap=50)

    if not chunks:
        print("[!] No chunks extracted. Check your PDF file.")
        return

    dataset = []
    print(f"[2/3] Generating Q&A pairs from {len(chunks)} chunks...")

    for idx, chunk in enumerate(
        chunks[:15]
    ):  # Process a slice first to verify execution
        print(f" -> Processing chunk {idx+1}/{len(chunks)}...")
        qa_pairs = generate_medical_pairs(chunk)
        if qa_pairs:
            dataset.extend(qa_pairs)

    print(f"[3/3] Saving {len(dataset)} generated training pairs to disk...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"✓ Training dataset successfully created at: {output_json_path}")


if __name__ == "__main__":
    build_dataset()
