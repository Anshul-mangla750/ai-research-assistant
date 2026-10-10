import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from flashrank import Ranker, RerankRequest

# 1. Page Configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment
load_dotenv()
DATA_DIR = Path("data")
INDEX_DIR = "faiss_index"
DATA_DIR.mkdir(exist_ok=True)

# 2. Cached Backend Resources
@st.cache_resource(show_spinner="Loading Embedding Model (BAAI/bge-small-en-v1.5)...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource(show_spinner="Loading FlashRank Cross-Encoder Re-ranker...")
def get_ranker():
    return Ranker()

@st.cache_resource(show_spinner="Loading / Building FAISS Vector Index...")
def get_vector_store(_embeddings):
    if os.path.exists(INDEX_DIR):
        return FAISS.load_local(
            INDEX_DIR, _embeddings, allow_dangerous_deserialization=True
        )
    
    # Build index if not present
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        return None

    loader = DirectoryLoader(str(DATA_DIR), glob="*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, _embeddings)
    vector_store.save_local(INDEX_DIR)
    return vector_store

def rebuild_index(_embeddings):
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        return None

    loader = DirectoryLoader(str(DATA_DIR), glob="*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, _embeddings)
    vector_store.save_local(INDEX_DIR)
    st.cache_resource.clear()
    return vector_store

# Initialize models
embeddings = get_embeddings()
ranker = get_ranker()
vector_store = get_vector_store(embeddings)

# 3. Sidebar UI
with st.sidebar:
    st.title("🔬 Research Assistant")
    st.caption("Advanced RAG with FlashRank & Gemini")
    st.divider()

    # Internal optimal configuration
    selected_model = "gemini-3.5-flash-lite"
    use_reranker = True
    initial_k = 15
    final_k = 4

    # Only show warning if API key is missing
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ GOOGLE_API_KEY is missing in .env")

    st.divider()
    st.subheader("📁 Document Library")
    existing_pdfs = list(DATA_DIR.glob("*.pdf"))
    st.write(f"**Loaded Papers ({len(existing_pdfs)}):**")
    for f in existing_pdfs:
        st.caption(f"📄 {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    # Upload New Papers
    uploaded_files = st.file_uploader("Upload Additional Papers (PDF)", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            target_path = DATA_DIR / uploaded_file.name
            if not target_path.exists():
                with open(target_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
        st.success("Uploaded! Click Rebuild Index below.")

    if st.button("🔄 Rebuild Vector Index", use_container_width=True):
        with st.spinner("Re-indexing all PDF documents..."):
            vector_store = rebuild_index(embeddings)
        st.success("Vector store updated successfully!")
        st.rerun()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 4. Prompt Template
PROMPT_TEMPLATE = """You are an expert AI research assistant. 
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

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def get_llm(model_name: str):
    return ChatGoogleGenerativeAI(
        model=model_name,
    )

# Retrieval + Reranking Logic
def retrieve_contexts(query: str, initial_k: int, final_k: int, apply_rerank: bool):
    if vector_store is None:
        return [], ""

    raw_docs = vector_store.similarity_search(query, k=initial_k)
    if not raw_docs:
        return [], ""

    if apply_rerank:
        passages = [
            {"id": i, "text": d.page_content, "meta": d.metadata}
            for i, d in enumerate(raw_docs)
        ]
        rerank_req = RerankRequest(query=query, passages=passages)
        reranked = ranker.rerank(rerank_req)[:final_k]

        selected_chunks = []
        formatted_list = []
        for item in reranked:
            source = Path(item["meta"].get("source", "Unknown")).name
            page = item["meta"].get("page", 0) + 1
            score = item.get("score", 0.0)
            selected_chunks.append({
                "source": source,
                "page": page,
                "score": f"{score:.3f}",
                "text": item["text"]
            })
            formatted_list.append(
                f"[Document: {source} | Page: {page} | Score: {score:.3f}]\n{item['text']}"
            )
        context_text = "\n\n---\n\n".join(formatted_list)
        return selected_chunks, context_text
    else:
        selected_chunks = []
        formatted_list = []
        for d in raw_docs[:final_k]:
            source = Path(d.metadata.get("source", "Unknown")).name
            page = d.metadata.get("page", 0) + 1
            selected_chunks.append({
                "source": source,
                "page": page,
                "score": "N/A",
                "text": d.page_content
            })
            formatted_list.append(
                f"[Document: {source} | Page: {page}]\n{d.page_content}"
            )
        context_text = "\n\n---\n\n".join(formatted_list)
        return selected_chunks, context_text

# 5. Main Chat Interface
st.header("🔬 AI Research Assistant")
st.markdown("Ask deep questions across your research library and get structured, cited answers in seconds.")

# Initialize chat session
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your AI Research Assistant. Ask me anything about your loaded research papers.",
            "sources": []
        }
    ]

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Retrieved Sources ({len(msg['sources'])} chunks)"):
                for idx, src in enumerate(msg["sources"], start=1):
                    st.markdown(f"**Chunk {idx}:** `{src['source']}` (Page {src['page']}) | *Relevance Score: {src['score']}*")
                    st.text(src["text"][:350] + ("..." if len(src["text"]) > 350 else ""))

# User query input
if user_query := st.chat_input("Ask a research question (e.g., 'What are the main paradigms of Modular RAG?')..."):
    # Render user prompt
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Generate assistant response with streaming
    with st.chat_message("assistant"):
        with st.spinner("Retrieving candidates & re-ranking passages..."):
            sources, context_str = retrieve_contexts(
                user_query,
                initial_k=initial_k,
                final_k=final_k,
                apply_rerank=use_reranker
            )

        if not context_str:
            response_text = "No document content was found to answer this question. Please ensure PDFs are in `data/` and indexed."
            st.warning(response_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "sources": []
            })
        else:
            llm = get_llm(selected_model)
            chain = prompt | llm | StrOutputParser()

            def stream_response():
                for token in chain.stream({"context": context_str, "question": user_query}):
                    yield token

            # Real-time token streaming
            response_text = st.write_stream(stream_response())

            # Show retrieved sources accordion
            if sources:
                with st.expander(f"📚 Retrieved Sources ({len(sources)} chunks)"):
                    for idx, src in enumerate(sources, start=1):
                        st.markdown(f"**Chunk {idx}:** `{src['source']}` (Page {src['page']}) | *Relevance Score: {src['score']}*")
                        st.text(src["text"][:350] + ("..." if len(src["text"]) > 350 else ""))

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "sources": sources
            })
