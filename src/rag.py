import os
import sys
import chromadb
from chromadb.utils import embedding_functions
import ollama

# Direct path routing from src folder to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


def query_rag_engine(question: str, n_results: int = 2):
    """Executes full RAG flow with an empathetic, clinical tone."""
    db_storage_path = os.path.join(project_root, "chroma_db")

    client = chromadb.PersistentClient(path=db_storage_path)

    embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_collection(
        name="medical_guide_collection", embedding_function=embedding_model
    )

    results = collection.query(query_texts=[question], n_results=n_results)

    # FIX: Safely flatten the nested list returned by ChromaDB
    retrieved_chunks = []
    if results and "documents" in results and results["documents"]:
        # Extract the inner list from results["documents"][0]
        retrieved_chunks = results["documents"][0]

    context_str = "\n\n".join(retrieved_chunks)

    # System prompt to mandate a smooth, human clinician persona
    system_prompt = (
        "You are an expert AI medical assistant. Answer the user's question using the "
        "provided operational guidelines and medical reference material. "
        "CRITICAL RULES FOR TONE:\n"
        "1. Speak naturally, empathetically, and professionally like a real clinical assistant.\n"
        "2. Never use robotic phrases like 'based on the provided context', 'according to the chunks', "
        "'the text states', or 'in this document'.\n"
        "3. Integrate the factual guidelines smoothly into your sentences as if it is your own direct expertise.\n"
        "4. Structure your response cleanly using bullet points if it helps readability."
    )

    user_prompt = (
        f"Reference Material:\n{context_str}\n\nUser Question:\n{question}\n\nAnswer:"
    )

    try:
        response = ollama.generate(
            model="llama3",
            system=system_prompt,
            prompt=user_prompt,
            options={
                "temperature": 0.3
            },  # Balanced for clinical tone while staying factual
        )
        return response["response"]

    except Exception as e:
        return f"[!] Error communicating with local Ollama service: {str(e)}"


if __name__ == "__main__":
    print("=" * 60)
    print("AI Medical Assistant — Live")
    print("Type your medical question below. Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        user_question = input("\nAsk a Question: ").strip()

        if not user_question:
            continue
        if user_question.lower() in ["exit", "quit"]:
            print("Shutting down the assistant. Take care!")
            break

        print("Reviewing medical details...")
        answer = query_rag_engine(user_question)

        print("\nAssistant Response:")
        print(answer)
        print("─" * 60)
