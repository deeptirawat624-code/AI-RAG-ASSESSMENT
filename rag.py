from pathlib import Path
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from groq import Groq

# ---------------- SETUP ----------------

DATA_FOLDER = Path("data")

model = SentenceTransformer("all-MiniLM-L6-v2")
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ---------------- DOCUMENT LOADING ----------------

def load_documents():
    documents = []

    for file in DATA_FOLDER.glob("*.txt"):
        text = file.read_text(encoding="utf-8")

        documents.append({
            "source": file.name,
            "text": text
        })

    return documents


documents = load_documents()

print(f"Loaded {len(documents)} document(s).")

for doc in documents:
    print(f"\nSource: {doc['source']}")
    print(doc["text"][:200])


# ---------------- CHUNKING ----------------

def chunk_text(text, chunk_size=800):
    chunks = []

    paragraphs = text.split("\n\n")

    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


all_chunks = []

for doc in documents:
    chunks = chunk_text(doc["text"])

    for chunk in chunks:
        all_chunks.append({
            "source": doc["source"],
            "text": chunk
        })


print(f"Created {len(all_chunks)} chunks.")


# ---------------- EMBEDDINGS ----------------

texts = [chunk["text"] for chunk in all_chunks]

embeddings = model.encode(texts)

print("Embeddings created successfully.")
print("Embeddings shape:", embeddings.shape)


# ---------------- CHROMADB ----------------

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Delete old collection so it can be recreated cleanly
try:
    chroma_client.delete_collection(name="company_documents")
except:
    pass

collection = chroma_client.get_or_create_collection(
    name="company_documents",
    metadata={"hnsw:space": "cosine"}
)

collection.upsert(
    ids=[str(i) for i in range(len(all_chunks))],
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=[
        {"source": chunk["source"]}
        for chunk in all_chunks
    ]
)

print("ChromaDB collection created successfully.")
print("Documents stored:", collection.count())


# ---------------- RETRIEVAL ----------------

def search(query, top_k=2, threshold=0.40):

    # Convert question into embedding
    query_embedding = model.encode([query])[0]

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    final_results = []

    documents_found = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    for text, distance, metadata in zip(
        documents_found,
        distances,
        metadatas
    ):

        # For cosine distance:
        score = 1 - distance

        if score >= threshold:
            final_results.append({
                "source": metadata["source"],
                "text": text,
                "score": score
            })

    return final_results


# ---------------- TEST SEARCH ----------------

query = "What does the company do?"

results = search(query)

print("\n----- SEARCH RESULTS -----")

for result in results:
    print("\nSource:", result["source"])
    print("Score:", round(result["score"], 4))
    print("Text:", result["text"])

# ---------------- GROQ ANSWER GENERATION ----------------

def generate_answer(query, results):

    context = "\n\n".join(
        result["text"]
        for result in results
    )

    prompt = f"""
You are a helpful assistant.

Answer the user's question using ONLY the information provided in the context below.

If the answer is not present in the context, say:
"I could not find this information in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content
# ---------------- RAG QUERY ----------------

while True:

    query = input("\nAsk a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    results = search(query)

    if not results:
        print("\nNo relevant information found in the documents.")
        continue

    context = "\n\n".join(
        result["text"] for result in results
    )

    prompt = f"""
Answer the question using only the information provided below.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer only from the provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    print("\n----- ANSWER -----")
    print(answer)

    print("\n----- SOURCES -----")

    for result in results:
        print(
            f"{result['source']} "
            f"(score: {result['score']:.4f})"
        )