# Imports
import os, re, time, io
from flask import Flask, render_template, request, redirect, Response
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# OCR imports for image-based PDFs
from pdf2image import convert_from_bytes
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# OpenAI client
from openai import OpenAI

# --- Configuration ---
DATA_DIR = "__data__"
os.makedirs(DATA_DIR, exist_ok=True)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

openai_client = OpenAI(api_key=api_key)

# Flask app
app = Flask(__name__)

# Shared progress state
progress = {"value": 0}

vectorstore = None
conversation_chain = None
chat_history = []

# --- Message classes ---
class HumanMessage:
    def __init__(self, content): self.content = content
    def __repr__(self): return f'HumanMessage(content={self.content})'

class AIMessage:
    def __init__(self, content): self.content = content
    def __repr__(self): return f'AIMessage(content={self.content})'


# --- PDF processing helpers ---
def extract_text_from_pdf(pdf_file):
    """Extract text from text-based PDF."""
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def extract_text_from_image_pdf(pdf_file):
    """Extract text from image-based PDF using OCR."""
    POPPLER_PATH = r"C:\poppler-25.07.0\Library\bin"
    pdf_bytes = pdf_file.read()
    pages = convert_from_bytes(pdf_bytes, poppler_path=POPPLER_PATH)
    text = ""
    total_pages = len(pages)
    for i, page in enumerate(pages):
        text += pytesseract.image_to_string(page)
        progress["value"] = int(((i + 1) / total_pages) * 100)
        time.sleep(0.3)  # simulate load for smoother updates
    return text


def get_pdf_text(pdf_docs):
    """Process uploaded PDFs (text or image) and update progress."""
    full_text = ""
    total_docs = len(pdf_docs)
    for doc_index, pdf in enumerate(pdf_docs):
        filename = os.path.join(DATA_DIR, pdf.filename)

        pdf.seek(0)
        text = extract_text_from_pdf(pdf)

        pdf.seek(0)
        if not text.strip():
            text = extract_text_from_image_pdf(pdf)

        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)

        full_text += text
        progress["value"] = int(((doc_index + 1) / total_docs) * 100)
    return full_text


# --- LangChain helpers ---
def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)


# --- Flask routes ---
@app.route('/')
def home():
    return render_template('new_home.html')


@app.route('/process', methods=['POST'])
def process_documents():
    """Handle PDF uploads and update progress."""
    global vectorstore, conversation_chain, progress
    progress["value"] = 0

    pdf_docs = request.files.getlist('pdf_docs')
    if not pdf_docs:
        return "No PDFs uploaded.", 400

    raw_text = get_pdf_text(pdf_docs)
    progress["value"] = 50  # halfway mark

    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)
    progress["value"] = 100  # done

    return "Processing complete!"


@app.route('/progress')
def progress_stream():
    """Stream live progress to frontend."""
    def generate():
        last_value = -1
        while True:
            if progress["value"] != last_value:
                last_value = progress["value"]
                yield f"data:{last_value}\n\n"
            if progress["value"] >= 100:
                break
            time.sleep(0.3)
    return Response(generate(), mimetype='text/event-stream')


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global conversation_chain, chat_history
    if request.method == 'POST':
        user_question = request.form.get('user_question', '')
        if conversation_chain:
            response = conversation_chain({'question': user_question})
            chat_history = response.get('chat_history', [])
        else:
            chat_history.append(AIMessage("Vectorstore not initialized."))

    return render_template('new_chat.html', chat_history=chat_history)


@app.route('/pdf_chat', methods=['GET', 'POST'])
def pdf_chat():
    return render_template('new_pdf_chat.html')


# --- Run Flask ---
if __name__ == '__main__':
    app.run(debug=True)
