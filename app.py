import os
import tempfile
import hashlib
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document
from pdf2image import convert_from_path
import pytesseract
import pdfplumber
from langchain_community.document_loaders import UnstructuredFileLoader

# ------------------ CONFIG ------------------ #
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
st.set_page_config(page_title="📄 Multi-File Document Q&A", layout="centered")
st.title("📄 Ask Questions from Your Uploaded Documents")

# ------------------ INIT SESSION ------------------ #
if "qa_memory_all" not in st.session_state:
    st.session_state.qa_memory_all = {}

if "all_docs" not in st.session_state:
    st.session_state.all_docs = []

if "indexed_file_hashes" not in st.session_state:
    st.session_state.indexed_file_hashes = set()

if "combined_vectorstore" not in st.session_state:
    st.session_state.combined_vectorstore = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ------------------ UTILS ------------------ #
def compute_file_hash(uploaded_file):
    file_content = uploaded_file.read()
    uploaded_file.seek(0)  # Reset for further reading
    return hashlib.sha256(file_content).hexdigest()

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name

def extract_structured_blocks(text: str):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    structured = {
        "headings": [],
        "key_values": [],
        "table_rows": []
    }

    for line in lines:
        if line.isupper() and len(line) < 50:
            structured["headings"].append(line)
        elif ":" in line:
            parts = line.split(":", 1)
            key = parts[0].strip()
            value = parts[1].strip()
            structured["key_values"].append(f"{key}: {value}")
        elif len(line.split()) >= 2:
            structured["table_rows"].append(line)
    return structured

def format_structured_blocks(blocks, source_name):
    content = f"[From {source_name}]\n\n"
    if blocks["headings"]:
        content += "### Headings:\n" + "\n".join(blocks["headings"]) + "\n\n"
    if blocks["key_values"]:
        content += "### Key-Value Pairs:\n" + "\n".join(blocks["key_values"]) + "\n\n"
    if blocks["table_rows"]:
        content += "### Table Rows:\n" + "\n".join(blocks["table_rows"]) + "\n"
    return content.strip()

def load_document(file_path, original_filename):
    docs = []
    if file_path.lower().endswith(".pdf"):
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except:
            pass
        if not text.strip():
            try:
                images = convert_from_path(file_path)
                for img in images:
                    ocr_text = pytesseract.image_to_string(img)
                    text += ocr_text + "\n"
            except Exception as e:
                raise ValueError(f"OCR failed: {str(e)}")
        if text.strip():
            structured_blocks = extract_structured_blocks(text)
            structured_text = format_structured_blocks(structured_blocks, original_filename)
            doc = Document(page_content=structured_text, metadata={"source": original_filename})
            docs.append(doc)
    else:
        loader = UnstructuredFileLoader(file_path, mode="elements")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = original_filename
    return docs

def create_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    if len(chunks) == 0:
        return None, 0
    for chunk in chunks:
        source = chunk.metadata.get("source", "Unknown")
        chunk.page_content = f"[From {source}]\n{chunk.page_content}"
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(chunks)

def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = Ollama(model="llama3.2:3b")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ------------------ FILE UPLOAD ------------------ #
uploaded_files = st.file_uploader(
    "📂 Upload PDF, DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    files_to_index = []

    for file in uploaded_files:
        filename = file.name
        file_hash = compute_file_hash(file)

        if file_hash in st.session_state.indexed_file_hashes:
            st.info(f"ℹ️ `{filename}` already uploaded and indexed (based on content). Skipping.")
            continue

        file_path = save_uploaded_file(file)
        try:
            docs = load_document(file_path, filename)
            if not docs:
                st.warning(f"⚠️ No text found in {filename}. It might be a scanned image or empty.")
                continue
            files_to_index.append((filename, docs, file_hash))
            st.success(f"✅ Loaded `{filename}`")
        except Exception as e:
            st.error(f"❌ Error loading {filename}: {str(e)}")

    if files_to_index:
        for filename, docs, file_hash in files_to_index:
            with st.spinner(f"🔍 Indexing `{filename}`..."):
                new_vectorstore, num_chunks = create_vectorstore(docs)

                if new_vectorstore:
                    if st.session_state.combined_vectorstore:
                        st.session_state.combined_vectorstore.merge_from(new_vectorstore)
                    else:
                        st.session_state.combined_vectorstore = new_vectorstore

                    st.session_state.all_docs.extend(docs)
                    st.session_state.indexed_file_hashes.add(file_hash)

                    st.session_state.qa_chain = build_qa_chain(st.session_state.combined_vectorstore)

                    st.success(f"✅ Indexed `{filename}` with {num_chunks} chunks.")
                else:
                    st.warning(f"⚠️ `{filename}` has no valid chunks and was skipped.")

# ------------------ Q&A ------------------ #
if st.session_state.combined_vectorstore and st.session_state.qa_chain:
    query = st.text_input("💬 Ask your question (mention file name to target it):")

    if query and st.button("🧠 Get Answer"):
        with st.spinner("🤔 Thinking..."):
            memory = st.session_state.qa_memory_all

            if query in memory:
                st.markdown("### 💡 Answer (from memory):")
                st.write(memory[query]['answer'])
                st.caption(f"📄 Source: {memory[query]['source']}")
            else:
                all_docs = st.session_state.all_docs
                filenames = [doc.metadata.get("source", "").lower() for doc in all_docs]
                matched_file = next((f for f in filenames if f in query.lower()), None)

                if matched_file:
                    filtered_docs = [
                        doc for doc in all_docs if doc.metadata.get("source", "").lower() == matched_file
                    ]
                    filtered_vectorstore, _ = create_vectorstore(filtered_docs)
                    qa_chain = build_qa_chain(filtered_vectorstore)
                    docs = filtered_vectorstore.similarity_search(query, k=1)
                    answer = qa_chain.run(query)
                    source = docs[0].metadata.get("source", "Unknown") if docs else matched_file
                else:
                    docs = st.session_state.combined_vectorstore.similarity_search(query, k=1)
                    answer = st.session_state.qa_chain.run(query)
                    source = docs[0].metadata.get("source", "Unknown") if docs else "Unknown"

                memory[query] = {"answer": answer, "source": source}
                st.markdown("### 💡 Answer:")
                st.write(answer)
                st.caption(f"📄 Source: {source}")

    if st.session_state.qa_memory_all:
        st.markdown("### 📝 Previous Questions:")
        for q, info in st.session_state.qa_memory_all.items():
            with st.expander(q):
                st.write(info['answer'])
                st.caption(f"📄 Source: {info['source']}")
