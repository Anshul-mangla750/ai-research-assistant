import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from flashrank import Ranker, RerankRequest

# 1. Load Environment Variables
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    print("[WARNING] GOOGLE_API_KEY is not set in your .env file!")
    print("Please add 'GOOGLE_API_KEY=your_key_here' to the .env file in this directory.")

# 2. Embedding Model (runs locally on CPU)
print("Loading embedding model (BAAI/bge-small-en-v1.5)...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 3. Vector Store: Load from disk cache if exists, otherwise build and save
INDEX_DIR = "faiss_index"

if os.path.exists(INDEX_DIR):
    print(f"Loading existing vector store from '{INDEX_DIR}'...")
    vector_store = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
else:
    print("Loading PDF documents from 'data/' directory...")
    loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    print(f"Loaded {len(docs)} document pages.")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Generating embeddings and building FAISS vector store...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(INDEX_DIR)
    print(f"Saved FAISS index to '{INDEX_DIR}' for faster future loads.")

# 4. Initialize FlashRank Cross-Encoder Re-ranker (ultra-fast, runs locally)
print("Loading FlashRank re-ranker...")
ranker = Ranker()

def retrieve_and_rerank(query: str, initial_k: int = 15, final_k: int = 4):
    """
    1. Retrieve top initial_k candidates from FAISS vector store.
    2. Re-rank them using FlashRank cross-encoder.
    3. Return top final_k most accurate chunks.
    """
    initial_docs = vector_store.similarity_search(query, k=initial_k)
    if not initial_docs:
        return []

    passages = [
        {"id": i, "text": doc.page_content, "meta": doc.metadata}
        for i, doc in enumerate(initial_docs)
    ]
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked = ranker.rerank(rerank_request)

    # Reconstruct top final_k formatted documents
    top_results = []
    for item in reranked[:final_k]:
        top_results.append({
            "text": item["text"],
            "score": item.get("score", 0.0),
            "source": Path(item["meta"].get("source", "Unknown")).name,
            "page": item["meta"].get("page", 0) + 1,
        })
    return top_results

def format_reranked_docs(reranked_docs):
    formatted = []
    for doc in reranked_docs:
        header = f"[Document: {doc['source']} | Page: {doc['page']} | Relevance Score: {doc['score']:.3f}]"
        formatted.append(f"{header}\n{doc['text']}")
    return "\n\n---\n\n".join(formatted)

# 5. Gemini LLM Initialization (gemini-3.8-flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
)

# 6. Structured Research Prompt Template
prompt_template = """You are an expert AI research assistant. 
Answer the user's question accurately and thoroughly based ONLY on the provided context.
If the answer cannot be found in the context, clearly state that the provided papers do not contain enough information.

Format your response using this clean structure:
### 1. Executive Summary
(A concise 2-3 sentence overview of the answer)

### 2. Key Findings & Details
(Detailed, bulleted points addressing the question directly)

### 3. Methodology & Technical Nuances
(Any algorithms, architectures, metrics, or methods highlighted in the text)

### 4. Sources & Citations
(Reference specific paper titles, sections, or page numbers found in the context)

---
Context:
{context}

Question:
{question}

Answer:"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# Helper function to run the RAG chain with streaming
def stream_answer(query: str):
    reranked_docs = retrieve_and_rerank(query, initial_k=15, final_k=4)
    if not reranked_docs:
        yield "No relevant information found in the documents."
        return

    context_text = format_reranked_docs(reranked_docs)
    chain = prompt | llm | StrOutputParser()

    for chunk in chain.stream({"context": context_text, "question": query}):
        yield chunk

# 7. Interactive Query Loop with Real-time Streaming
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   AI RESEARCH ASSISTANT (Re-Ranking + Streaming Active)")
    print("=" * 60)
    print("Type your question and press Enter. (Type 'exit' to quit)\n")

    while True:
        try:
            query = input("Ask a question: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting. Have a great day!")
                break

            print("\n[Re-ranking top 15 candidate chunks down to top 4...]")
            print("-" * 60 + "\n")

            # Stream response token-by-token
            for token in stream_answer(query):
                print(token, end="", flush=True)

            print("\n\n" + "=" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as e:
            print(f"\n[Error] {e}\n")
