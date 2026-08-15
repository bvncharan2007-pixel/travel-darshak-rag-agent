%%writefile app.py

import os
import fitz

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="RAG Agent - PDF Question Answering",
    version="1.0.0"
)


# ==========================================
# GEMINI CONFIGURATION
# ==========================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set.")


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY
)


# ==========================================
# GLOBAL VECTOR STORE
# ==========================================

vectorstore = None
uploaded_filename = None


# ==========================================
# RAG FUNCTION
# ==========================================

def generate_answer(question):

    global vectorstore

    if vectorstore is None:
        return "Please upload a PDF first."

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
You are a helpful RAG assistant.

Answer the question using ONLY the provided PDF context.

If the answer is not present in the context, say:

"I couldn't find that information in the provided PDF."

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, list):
        return "\n".join(
            block["text"]
            for block in response.content
            if isinstance(block, dict)
            and block.get("type") == "text"
        )

    return response.content


# ==========================================
# HOME PAGE
# ==========================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
    <!DOCTYPE html>
    <html>
    <head>

        <title>KT RAG Agent</title>

        <style>

            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f5f5;
            }

            .container {
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }

            h1 {
                text-align: center;
            }

            input[type="file"],
            input[type="text"] {
                width: 100%;
                padding: 12px;
                margin: 10px 0;
                box-sizing: border-box;
            }

            button {
                padding: 12px 20px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                background: #222;
                color: white;
            }

            .section {
                margin-top: 25px;
            }

            #result {
                margin-top: 20px;
                padding: 15px;
                background: #f0f0f0;
                border-radius: 8px;
                white-space: pre-wrap;
            }

            #status {
                margin-top: 10px;
                font-weight: bold;
            }

        </style>

    </head>

    <body>

        <div class="container">

            <h1>📚 KT RAG Agent</h1>

            <p>
                Upload your KT PDF and ask questions about its content.
            </p>


            <div class="section">

                <h3>1. Upload PDF</h3>

                <input
                    type="file"
                    id="pdfFile"
                    accept=".pdf"
                >

                <button onclick="uploadPDF()">
                    Upload PDF
                </button>

                <div id="status"></div>

            </div>


            <div class="section">

                <h3>2. Ask a Question</h3>

                <input
                    type="text"
                    id="question"
                    placeholder="Ask something about the PDF..."
                >

                <button onclick="askQuestion()">
                    Ask
                </button>

            </div>


            <div id="result"></div>

        </div>


        <script>

            async function uploadPDF() {

                const fileInput =
                    document.getElementById("pdfFile");

                const status =
                    document.getElementById("status");

                if (!fileInput.files.length) {

                    status.innerText =
                        "Please select a PDF first.";

                    return;
                }

                const formData = new FormData();

                formData.append(
                    "file",
                    fileInput.files[0]
                );

                status.innerText =
                    "Uploading and processing PDF...";

                const response =
                    await fetch("/upload", {

                        method: "POST",

                        body: formData

                    });

                const data =
                    await response.json();

                if (response.ok) {

                    status.innerText =
                        "✅ " + data.message +
                        " (" + data.chunks +
                        " chunks created)";

                } else {

                    status.innerText =
                        "❌ " + data.detail;

                }

            }


            async function askQuestion() {

                const question =
                    document.getElementById(
                        "question"
                    ).value;

                const result =
                    document.getElementById(
                        "result"
                    );

                if (!question) {

                    result.innerText =
                        "Please enter a question.";

                    return;

                }

                result.innerText =
                    "Thinking...";

                const formData =
                    new FormData();

                formData.append(
                    "question",
                    question
                );

                const response =
                    await fetch("/ask", {

                        method: "POST",

                        body: formData

                    });

                const data =
                    await response.json();

                if (response.ok) {

                    result.innerText =
                        data.answer;

                } else {

                    result.innerText =
                        "❌ " + data.detail;

                }

            }

        </script>

    </body>
    </html>
    """


# ==========================================
# PDF UPLOAD
# ==========================================

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    global vectorstore
    global uploaded_filename

    if not file.filename.lower().endswith(".pdf"):

        return {
            "error": "Only PDF files are supported."
        }

    pdf_bytes = await file.read()

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    text = ""

    for page in doc:
        text += page.get_text() + "\n"

    doc.close()

    if not text.strip():

        return {
            "error": "Could not extract text from PDF."
        }

    # ======================================
    # TEXT CHUNKING
    # ======================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)


    # ======================================
    # CREATE VECTOR STORE
    # ======================================

    vectorstore = FAISS.from_texts(
        chunks,
        embedding=embeddings
    )

    uploaded_filename = file.filename

    return {
        "message": "PDF uploaded successfully!",
        "filename": uploaded_filename,
        "chunks": len(chunks)
    }


# ==========================================
# ASK QUESTION
# ==========================================

@app.post("/ask")
async def ask_question(
    question: str = Form(...)
):

    if vectorstore is None:

        return {
            "error": "Please upload a PDF first."
        }

    answer = generate_answer(question)

    return {
        "question": question,
        "answer": answer
    }
