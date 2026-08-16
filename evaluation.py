from pathlib import Path
import time
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------- SETUP ----------------

DATA_FOLDER = Path("data")

model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="company_documents",
    metadata={"hnsw:space": "cosine"}
)

# ---------------- EVALUATION DATASET ----------------

evaluation_data = [
    {
        "question": "What does the company do?",
        "expected": "software development, web development, mobile application development, cloud services, and artificial intelligence solutions"
    },
    {
        "question": "When was the company founded?",
        "expected": "2020"
    },
    {
        "question": "What is the internship duration?",
        "expected": "3 months"
    },
    {
        "question": "What days can employees work?",
        "expected": "Monday to Friday"
    },
    {
        "question": "Does the company provide web development internships?",
        "expected": "web development"
    },
    {
        "question": "Does the company provide artificial intelligence internships?",
        "expected": "artificial intelligence"
    },
    {
        "question": "Does the company provide data science internships?",
        "expected": "data science"
    },
    {
        "question": "Does the company provide software development internships?",
        "expected": "software development"
    },
    {
        "question": "Does the company use Python?",
        "expected": "Python"
    },
    {
        "question": "Does the company use MongoDB?",
        "expected": "MongoDB"
    }
]

# ---------------- RETRIEVAL ----------------

def search(query, top_k=2, threshold=0.40):

    query_embedding = model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    documents_found = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    final_results = []

    for text, distance, metadata in zip(
        documents_found,
        distances,
        metadatas
    ):

        score = 1 - distance

        if score >= threshold:
            final_results.append({
                "source": metadata["source"],
                "text": text,
                "score": score
            })

    return final_results


# ---------------- EVALUATION ----------------

correct = 0
total = len(evaluation_data)
latencies = []

print("\n========== RAG EVALUATION ==========\n")

for i, item in enumerate(evaluation_data, start=1):

    start_time = time.perf_counter()

    results = search(item["question"])

    end_time = time.perf_counter()

    latency = end_time - start_time
    latencies.append(latency)

    retrieved_text = " ".join(
        result["text"] for result in results
    )

    expected = item["expected"].lower()

    if expected in retrieved_text.lower():
        status = "PASS"
        correct += 1
    else:
        status = "FAIL"

    print(f"{i}. {item['question']}")
    print(f"Expected: {item['expected']}")
    print(f"Result: {status}")
    print(f"Latency: {latency:.4f} seconds")
    print("-" * 50)


# ---------------- METRICS ----------------

accuracy = (correct / total) * 100
average_latency = sum(latencies) / len(latencies)

print("\n========== EVALUATION RESULTS ==========")

print(f"Total Questions: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Average Retrieval Latency: {average_latency:.4f} seconds")

print("\nEvaluation completed successfully.")