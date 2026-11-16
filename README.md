# LegalGPT — Legal Research Chatbot (Ollama + RAG)

LegalGPT is a **retrieval-augmented legal information chatbot**. It searches a supplied legal-records dataset, retrieves the most relevant records, and uses a locally hosted Ollama language model to draft an answer grounded in those records.

The project includes a ChatGPT-style web interface, chat history for the current browser session, and support for image, PDF, and Word attachments. It can be run in **Google Colab** or locally on a computer with Ollama installed.

> **Disclaimer:** LegalGPT is an educational information-retrieval project, not a lawyer or a substitute for professional legal advice. Its answers depend on the supplied dataset, which may be incomplete or out of date. Verify important information against current official sources and consult a qualified lawyer.

## Features

- **RAG-based legal search:** creates sentence embeddings from legal records and searches them with FAISS.
- **Local LLM:** uses Ollama with `llama3.2:3b` by default; no OpenAI API key is required.
- **ChatGPT-style interface:** browser-based chat UI with a message composer and attachment button.
- **File attachments:** accepts PNG, JPG/JPEG, WEBP, PDF, DOCX, and legacy DOC files.
- **Document extraction:** extracts text from PDFs (up to 100 pages) and Word documents (including DOCX tables).
- **Image OCR:** extracts visible text from images using Tesseract. Optional Ollama vision-model support can be configured for visual image analysis.
- **Chat history:** stores chats and messages in a local SQLite database, associated with a browser session identifier.
- **Dataset refresh:** the terminal chatbot can clean the source CSV and rebuild the FAISS index using the `refresh` command.

## How it works

```text
dataset.csv
    |
    v
Cleaned_data.py  ---> dataset_cleaned.csv
    |
    v
build_index.py
    |  Sentence Transformers embeddings
    v
index/legal.faiss + index/metadata.json
    |
    v
search_rag.py  <--- user's question
    |                 |
    | retrieves       |
    | relevant legal  |
    | records         |
    v                 v
             chatbot.py / app.py
                    |
                    v
             Ollama LLM
                    |
                    v
          Answer + retrieved sources
```

For the web application, `app.py` serves the FastAPI backend and `static/index.html` provides the browser interface. Uploaded documents are converted to text and passed to the model with the question and retrieved legal records.

## Repository structure

```text
LegalGPT_Colab_No_Firebase/
├── app.py                         # FastAPI web application and chat API
├── chatbot.py                     # Ollama connection, RAG pipeline, terminal chatbot
├── Cleaned_data.py                # Cleans dataset.csv and removes duplicate rows
├── build_index.py                 # Builds the FAISS vector index
├── search_rag.py                  # Retrieves the most relevant legal records
├── dataset.csv                    # Source legal dataset
├── dataset_cleaned.csv            # Cleaned dataset
├── index/
│   ├── legal.faiss                # FAISS vector index
│   ├── metadata.json              # Indexed legal-record metadata
│   └── index_config.json          # Embedding model and index information
├── static/
│   └── index.html                 # ChatGPT-style frontend
├── LegalGPT_No_Firebase_Colab.ipynb # Google Colab setup and launch notebook
├── requirements.txt               # Python/RAG/attachment dependencies
├── requirements-web.txt           # FastAPI web-server dependencies
└── README.md
```

## Run in Google Colab

The included notebook automates the Colab setup, including Python packages, system tools, Ollama, model download, index creation, backend launch, and creation of a Colab proxy URL.

1. Upload this repository to GitHub, or download the ZIP from the repository.
2. Open `LegalGPT_No_Firebase_Colab.ipynb` in Google Colab.
3. Run the notebook cells **from top to bottom**.
4. When prompted by the upload cell, upload the project ZIP.
5. Wait for the dependencies and system packages to install.
6. The notebook installs/starts Ollama and downloads `llama3.2:3b`.
7. It cleans the CSV and builds the legal search index.
8. Start the backend using the notebook cell and open the generated web URL.
9. Ask a question in the browser. Use the **＋** attachment control to attach supported files.

**Colab note:** The public proxy URL is temporary. It works only while the Colab runtime and backend process are running. It is not permanent hosting.

## Run locally on Windows

### 1. Install prerequisites

- Install Python 3.10 or newer.
- Install the Ollama desktop application from [https://ollama.com/download](https://ollama.com/download).
- Open PowerShell and download the default model:

```powershell
ollama pull llama3.2:3b
ollama list
```

Make sure the Ollama application is running before launching LegalGPT.

### 2. Open the project folder

In PowerShell, change directory to the folder containing `app.py`:

```powershell
cd path\to\LegalGPT_Colab_No_Firebase
```

### 3. Create and activate a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use this for the current terminal window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 4. Install Python dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-web.txt
```

### 5. Install OCR tools (for image text extraction)

Install Tesseract OCR and ensure its executable is on your Windows PATH. Legacy `.doc` extraction also requires `antiword`; if it is not installed, use `.docx` instead.

### 6. Start the web application

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

Stop the server with `Ctrl+C`.

## Run the terminal chatbot instead

The terminal version rebuilds the cleaned dataset and FAISS index when it starts, then opens an interactive command-line chat:

```powershell
python chatbot.py
```

Available commands inside the terminal chatbot:

- `refresh` — clean `dataset.csv` and rebuild/reload the RAG index.
- `sources` — show the retrieved records for the most recent answer.
- `clear` — clear the current terminal conversation.
- `exit` — quit.

The terminal chatbot's conversation history is held only in memory for that run. The web app uses SQLite for chat/message persistence in the current runtime.

## Dataset format

The source CSV is expected to contain these exact columns:

| Column | Purpose |
|---|---|
| `Jurisdictional Reference` | Legal reference or identifier |
| `Type` | Record category/type |
| `Description` | Text describing the legal record |

To use another dataset, keep these column names or update `REQUIRED_COLUMNS` in `build_index.py`. Place the replacement file at `dataset.csv`, then rebuild the index.

## Attachments and limits

| File type | Processing |
|---|---|
| PNG, JPG/JPEG, WEBP | OCR with Tesseract; optional Ollama vision model |
| PDF | Extracts text from up to the first 100 pages |
| DOCX | Extracts paragraphs and table contents |
| DOC | Uses `antiword` if installed |

- Up to **5 files per message**.
- Maximum **15 MB per file**.
- Extracted attachment text is capped before being sent to the model.
- Image understanding beyond visible text requires an optional compatible Ollama vision model. OCR-only mode does not interpret charts, scenes, or other non-text visual content.

## Optional vision model

The default configuration uses OCR for images. For richer image understanding, install a compatible Ollama vision model, for example:

```bash
ollama pull llava:7b
```

Set `OLLAMA_VISION_MODEL=llava:7b` in the environment **before starting the backend**. The model is large and may exceed the memory available in some machines or Colab runtimes.

## Privacy, storage, and security notes

- This version does **not** include Google login, Firebase, or account-level authentication.
- The web app stores chat records in `legalgpt.sqlite3` on the machine/runtime where it runs.
- In Colab, the runtime filesystem is temporary; chats and files stored there may disappear when the runtime is reset.
- Uploaded original files are processed by the backend; extracted text and attachment names are stored with message history. Do not upload confidential or sensitive legal documents to a shared/public session.
- Do not expose the server directly to the public internet without adding authentication, access controls, HTTPS, and appropriate data-protection measures.
- Keep secrets and local environment files out of GitHub. The included `.gitignore` excludes `.env` and common local Python artifacts.

## Troubleshooting

**`Cannot connect to Ollama`**
- Open the Ollama application.
- In PowerShell, run `ollama list`.
- Confirm the default model is installed with `ollama pull llama3.2:3b`.

**`model not found`**
- Run `ollama pull llama3.2:3b`.
- Check that `OLLAMA_MODEL` (if configured) matches the output of `ollama list`.

**Missing or mismatched FAISS index**
- Run `python Cleaned_data.py` followed by `python build_index.py`, or start `python chatbot.py` to run the terminal pipeline automatically.

**Image OCR returns no text**
- Check that Tesseract is installed and available on PATH.
- OCR works best with clear, high-resolution images containing readable text.

**Legacy `.doc` upload fails**
- Install `antiword`, or save the document as `.docx` and upload that format.

## Technology stack

Python · FastAPI · Ollama · LangChain Ollama · Sentence Transformers · FAISS · SQLite · Pandas · NumPy · Tesseract OCR · HTML/CSS/JavaScript · Google Colab

## Demo video

A ready-to-record walkthrough is included in [`VIDEO_DEMO_SCRIPT.md`](VIDEO_DEMO_SCRIPT.md). It contains a suggested 2–3 minute screen recording, narration, and scene timings.
