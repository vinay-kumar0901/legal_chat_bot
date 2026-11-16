# LegalGPT — 2–3 Minute GitHub Demo Video Script

This is a recording plan and narration script. It is not a pre-rendered video.

## Recording setup

- Open the running LegalGPT web app in a browser.
- Use a small, non-confidential sample question and sample PDF/DOCX/image.
- Keep the terminal visible briefly when showing the Ollama/backend startup.
- Avoid showing private data, API keys, local credentials, or personal documents.
- Record at 1080p if possible. OBS Studio, Clipchamp, or the built-in Windows screen recorder can be used.

## Timeline and narration

### 0:00–0:15 — Introduction
**On screen:** Project title, then LegalGPT chat interface.

**Say:**  
"LegalGPT is a legal information chatbot built with retrieval-augmented generation, or RAG. It uses a legal dataset to retrieve relevant records and Ollama to generate a natural-language response."

### 0:15–0:40 — Explain the architecture
**On screen:** Briefly show the repository files: `Cleaned_data.py`, `build_index.py`, `search_rag.py`, `chatbot.py`, and `app.py`.

**Say:**  
"First, the source CSV is cleaned. The project converts legal references and descriptions into embeddings and stores them in a FAISS index. When a user asks a question, the search module retrieves the closest matching records. Those records are supplied to the local Ollama model as evidence for the answer."

### 0:40–1:15 — Ask a legal-dataset question
**On screen:** Enter a question that is answerable from the included dataset. Show the response and any source references.

**Say:**  
"Here I am asking a question about the legal records. LegalGPT searches the dataset first, then generates an answer using the retrieved evidence. If the dataset does not provide enough information, the assistant is instructed to say so rather than invent a legal reference."

### 1:15–1:45 — Demonstrate attachments
**On screen:** Click the plus attachment control, attach a sample PDF or DOCX, ask for a summary or explanation.

**Say:**  
"The interface also accepts PDF and Word documents. It extracts their text and includes that content with the question. Images are accepted too; by default, visible text is extracted using OCR. An optional Ollama vision model can be configured for richer image understanding."

### 1:45–2:10 — Chat history and local model
**On screen:** Show a second chat or the chat list; briefly show terminal with Ollama/backend running.

**Say:**  
"The web application uses FastAPI and SQLite for chat storage in the current runtime. The language model runs through Ollama, so this project does not require an OpenAI API key."

### 2:10–2:30 — Close with limitations
**On screen:** Show README's setup section or project repository.

**Say:**  
"The project can be launched through the included Google Colab notebook or run locally with Python and Ollama. The Colab URL is temporary, and this version does not include user login or production-grade authentication. LegalGPT is an educational legal-information tool, not a replacement for a qualified lawyer."

## Suggested video title

**LegalGPT — RAG-Based Legal Chatbot with Ollama, FAISS, FastAPI & File Uploads**

## Suggested video description

"Demo of LegalGPT, a legal information chatbot built with Python, FastAPI, Ollama, LangChain, Sentence Transformers, FAISS, and SQLite. The project retrieves relevant legal records from a CSV dataset and uses a local LLM to generate grounded responses. It includes a ChatGPT-style web UI and supports PDF, Word, and image attachments with text extraction/OCR. Includes a Google Colab notebook and local setup instructions.

Note: Educational project only; not legal advice. The Colab proxy URL is temporary, and the demo version does not include account-level authentication."
