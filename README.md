# LegalGPT — Colab edition with ChatGPT-style file attachments

This update builds on the supplied Firebase-free LegalGPT project. It keeps the FastAPI UI, Ollama, FAISS legal RAG search, and SQLite chat history, and adds a **＋ Attach** button.

## Supported attachments

- Images: PNG, JPG/JPEG, WEBP
- PDF: text extraction from up to the first 100 pages
- Word: DOCX (paragraphs and tables) and legacy DOC (via `antiword`)
- Up to 5 files per message; 15 MB per file
- Multiple files can be attached to one message
- Attachment names are shown in the chat history

PDF and Word text is extracted and passed to the LLM alongside the legal RAG evidence. Images use OCR by default. To have the model interpret visual content (charts, scenes, signatures, layouts, etc.), install an Ollama vision model such as `llava:7b` and set `OLLAMA_VISION_MODEL=llava:7b` before starting the backend. This optional model is large and may not fit every Colab runtime. Without it, images are still accepted and visible text is extracted using OCR.

## Run in Google Colab

1. Open `LegalGPT_No_Firebase_Colab.ipynb` in Google Colab.
2. Run the cells from top to bottom.
3. In the upload cell, select this ZIP file.
4. Wait for the Python packages and system tools to install.
5. Ollama starts and downloads `llama3.2:3b` as the text model.
6. Run the index check/build cells. If the supplied index is present and the dataset has not changed, you can skip rebuilding.
7. Start the backend and open the generated Colab proxy URL.
8. Click **＋** next to the message box, choose one or more files, type a question, and send.

### Optional visual image understanding

If the Colab runtime has enough memory, run this in a notebook cell after Ollama starts:

```python
!ollama pull llava:7b
import os
os.environ["OLLAMA_VISION_MODEL"] = "llava:7b"
```

Then start/restart the Uvicorn backend cell. If the vision model is not installed or the environment variable is not set before the backend starts, the app uses OCR for images.

## Local setup (optional)

Install the system tools on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y antiword tesseract-ocr
```

Install Python dependencies:

```bash
pip install -r requirements.txt -r requirements-web.txt
```

Make sure Ollama is running and `llama3.2:3b` is downloaded, then:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## Important limitations and privacy

- The Colab proxy URL works only while the runtime and backend remain active. This is not permanent hosting.
- Chat history is stored in `legalgpt.sqlite3` on the temporary Colab runtime disk.
- Original uploaded files are processed in memory and are not saved as downloadable files. Extracted text and attachment names are included in the chat message history.
- Image bytes are sent to the optional Ollama vision model only when it is configured. Otherwise OCR runs locally in the backend.
- There is no Google login or account-level user authentication. Do not share a running session or upload confidential legal documents to a public/shared notebook.
- This app provides legal information, not legal advice. Verify current law and consult a qualified lawyer.
