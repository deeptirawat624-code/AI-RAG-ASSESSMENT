# AI RAG Assessment

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about company information and receive answers based only on the available documents.

## Features

- Loads company information from `.txt` documents
- Splits documents into meaningful chunks
- Generates embeddings using Sentence Transformers
- Stores and retrieves document embeddings using ChromaDB
- Uses Groq LLM for generating answers
- Shows the source document and similarity score
- Prevents unsupported answers when relevant information is not available
- Uses environment variables to keep the API key secure

## Technologies Used

- Python
- Sentence Transformers
- ChromaDB
- Groq
- NumPy
- python-dotenv
- PyPDF

## Project Structure

```text
AI-RAG-ASSESSMENT/
│
├── data/
│   └── company_info.txt
│
├── rag.py
├── requirements.txt
├── .gitignore
├── .env
└── README.md
```

## How It Works

1. The application loads documents from the `data` folder.
2. Documents are divided into chunks.
3. Sentence Transformers generates embeddings for the chunks.
4. ChromaDB stores the document embeddings.
5. The user's question is converted into an embedding.
6. ChromaDB retrieves the most relevant information.
7. Groq generates an answer using the retrieved information.
8. If no relevant information is found, the application reports that instead of generating an unsupported answer.

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root and add your Groq API key:

```text
GROQ_API_KEY=your_api_key_here
```

**Never upload the `.env` file or your API key to GitHub.**

## Run the Application

```bash
python rag.py
```

Then enter questions such as:

```text
What is the internship duration?
```

The application returns the answer along with the source document.

For questions that are not covered by the documents, such as:

```text
What is the CEO's name?
```

the application returns:

```text
No relevant information found in the documents.
```

## Example

**Question:**

`What is the internship duration?`

**Answer:**

`The internship duration is generally 3 months.`

**Source:**

`company_info.txt`

## Security

The Groq API key is stored in `.env` and excluded from Git using `.gitignore`.

## Author

AI RAG Assessment Project