import os
import sys
import chromadb
from chromadb.utils import embedding_functions

current_dir = os.path.dirname(os.path.abspath(__file__))  # vector_db folder
project_root = os.path.dirname(current_dir)  # AI-Medical-Assistant folder
sys.path.append(project_root)


from src.ingest import extract_and_chunk_pdf


def initialize_chroma(db_path):
    """Initializes a persistent ChromaDB client and sets up the embedding function."""
    os.makedirs(db_path, exist_ok=True)

    client = chromadb.PersistentClient(path=db_path)

    embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name="medical_guide_collection",
        embedding_function=embedding_model,
        metadata={"description": "AYUSH Homoeopathy Essential Drugs List"},
    )

    return collection


def store_chunks_in_chroma(collection, chunks):
    """Processes chunks, generates sequential IDs, and stores them."""
    if not chunks:
        print("[!] No chunks available to store.")
        return

    documents = []
    metadatas = []
    ids = []

    print(f"\n[ChromaDB] Preparing {len(chunks)} chunks for storage...")

    for idx, chunk in enumerate(chunks):
        page_num = 1
        if chunk.startswith("[Page "):
            try:
                page_num = int(chunk.split("]")[0].replace("[Page ", ""))
            except ValueError:
                pass

        documents.append(chunk)
        metadatas.append({"page": page_num, "source": "medical_guide.pdf"})
        ids.append(f"chunk_{idx:04d}")  # chunk_0000, chunk_0001, etc.

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print("✓ Successfully stored all embeddings in ChromaDB.")


def query_test(collection, search_query, n_results=2):
    """Verification function to test database retrieval accuracy."""
    print(f"\n[Test Query] Searching database for: '{search_query}'")
    results = collection.query(query_texts=[search_query], n_results=n_results)

    for i in range(len(results["ids"][0])):
        print(f"\nResult #{i+1} (Match Distance: {results['distances'][0] [i]:.4f})")
        print(f"Source: Page {results['metadatas'][0][i]['page']}")
        print(f"Text Chunk:\n{results['documents'][0][i]}")
        print("─" * 50)


if __name__ == "__main__":
    pdf_path = os.path.join(project_root, "data", "medical_guide.pdf")
    db_storage_path = os.path.join(project_root, "chroma_db")

    medical_collection = initialize_chroma(db_path=db_storage_path)

    if medical_collection.count() == 0:
        print("[Database] Collection empty. Commencing processing pipeline...")
        if not os.path.exists(pdf_path):
            print(f"[!] Target PDF file not found at: {pdf_path}")
        else:
            chunks = extract_and_chunk_pdf(pdf_path, chunk_size=300, overlap=50)
            if chunks:
                store_chunks_in_chroma(medical_collection, chunks)
    else:
        print(
            f"[Database] Loaded existing database with {medical_collection.count()} chunks."
        )

    query_test(medical_collection, "Homoeopathy Drug Control Cell", n_results=1)
