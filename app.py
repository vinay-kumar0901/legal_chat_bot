"""LegalGPT: FastAPI backend with SQLite history and multi-file attachments."""
from __future__ import annotations
import base64, json, mimetypes, os, sqlite3, subprocess, sys, threading, uuid
from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "legalgpt.sqlite3"
UPLOAD_DIR = BASE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_FILES = 5
ALLOWED = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
    ".pdf": "pdf", ".docx": "docx", ".doc": "doc",
}
app = FastAPI(title="LegalGPT", version="4.0-attachments")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
_db_lock = threading.Lock()
_llm_lock = threading.Lock()

def connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with connect() as db:
        db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS chats (
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
          title TEXT NOT NULL DEFAULT 'New chat',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_chats_session ON chats(session_id, updated_at);
        CREATE TABLE IF NOT EXISTS messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT NOT NULL,
          role TEXT NOT NULL, content TEXT NOT NULL,
          sources TEXT NOT NULL DEFAULT '[]',
          attachments TEXT NOT NULL DEFAULT '[]',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
        );
        """)
        cols = {r["name"] for r in db.execute("PRAGMA table_info(messages)")}
        if "attachments" not in cols:
            db.execute("ALTER TABLE messages ADD COLUMN attachments TEXT NOT NULL DEFAULT '[]'")
init_db()

from chatbot import check_ollama, create_llm, ask_legal_chatbot, create_vision_llm
from search_rag import search_legal_documents
check_ollama()
llm = create_llm()
vision_llm = create_vision_llm()

def extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(data))
    chunks = []
    for i, page in enumerate(reader.pages[:100], start=1):
        text = page.extract_text() or ""
        if text.strip(): chunks.append(f"[PDF page {i}]\n{text}")
    return "\n\n".join(chunks)

def extract_docx(data: bytes) -> str:
    from docx import Document
    doc = Document(BytesIO(data))
    chunks = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            chunks.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(chunks)

def extract_image_ocr(data: bytes) -> str:
    try:
        from PIL import Image
        import pytesseract
        image = Image.open(BytesIO(data)).convert("RGB")
        return pytesseract.image_to_string(image).strip()
    except Exception as exc:
        return f"[Image attached. OCR was unavailable: {type(exc).__name__}.]"

async def process_upload(upload: UploadFile, chat_id: str):
    filename = Path(upload.filename or "attachment").name
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type: {filename}. Use PNG/JPG/WEBP, PDF, DOCX or DOC.")
    data = await upload.read()
    if not data:
        raise HTTPException(400, f"{filename} is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, f"{filename} exceeds the 15 MB per-file limit.")
    kind = ALLOWED[ext]
    text = ""
    image_url = None
    if kind == "pdf":
        try: text = extract_pdf(data)
        except Exception as exc: raise HTTPException(400, f"Could not read {filename}: {exc}")
    elif kind == "docx":
        try: text = extract_docx(data)
        except Exception as exc: raise HTTPException(400, f"Could not read {filename}: {exc}")
    elif kind == "doc":
        # Legacy .doc extraction uses antiword, installed in the Colab notebook.
        import tempfile
        temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.doc"
        temp_path.write_bytes(data)
        try:
            result = subprocess.run(["antiword", str(temp_path)], capture_output=True, text=True, timeout=20)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "antiword could not parse this file")
            text = result.stdout.strip()
        except FileNotFoundError:
            raise HTTPException(400, "Legacy .doc support requires antiword. Run the notebook's system-package install cell.")
        except Exception as exc:
            raise HTTPException(400, f"Could not read {filename}: {exc}")
        finally:
            temp_path.unlink(missing_ok=True)
    elif kind == "image":
        mime = mimetypes.guess_type(filename)[0] or "image/jpeg"
        image_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        text = extract_image_ocr(data)
    return {
        "name": filename, "type": kind, "size": len(data),
        "text": text[:60000], "image_url": image_url,
    }

class NewChat(BaseModel):
    session_id: str
    title: str = "New chat"
class RenameBody(BaseModel):
    session_id: str
    title: str

def get_chat(chat_id: str, session_id: str):
    with connect() as db:
        row = db.execute("SELECT * FROM chats WHERE id=? AND session_id=?", (chat_id, session_id)).fetchone()
    if not row: raise HTTPException(status_code=404, detail="Chat not found")
    return row

@app.get("/")
def home(): return FileResponse(BASE / "static" / "index.html")
@app.get("/healthz")
def health(): return {"status":"ok","service":"LegalGPT","storage":"SQLite","attachments":["images","pdf","docx","doc"]}
@app.get("/api/chats")
def list_chats(session_id: str):
    if len(session_id) < 20: raise HTTPException(400, "Invalid browser session")
    with connect() as db:
        rows = db.execute("SELECT id,title,created_at,updated_at FROM chats WHERE session_id=? ORDER BY updated_at DESC", (session_id,)).fetchall()
    return [dict(r) for r in rows]
@app.post("/api/chats")
def create_chat(body: NewChat):
    if len(body.session_id) < 20: raise HTTPException(400, "Invalid browser session")
    chat_id = uuid.uuid4().hex; title = body.title.strip() or "New chat"
    with connect() as db: db.execute("INSERT INTO chats(id,session_id,title) VALUES(?,?,?)", (chat_id,body.session_id,title))
    return {"id":chat_id,"title":title}
@app.patch("/api/chats/{chat_id}")
def rename_chat(chat_id: str, body: RenameBody):
    get_chat(chat_id,body.session_id)
    with connect() as db: db.execute("UPDATE chats SET title=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",(body.title.strip(),chat_id))
    return {"id":chat_id,"title":body.title.strip()}
@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, session_id: str):
    get_chat(chat_id,session_id)
    with connect() as db:
        db.execute("DELETE FROM messages WHERE chat_id=?",(chat_id,)); db.execute("DELETE FROM chats WHERE id=?",(chat_id,))
    return {"deleted":True}
@app.get("/api/chats/{chat_id}/messages")
def get_messages(chat_id: str, session_id: str):
    get_chat(chat_id,session_id)
    with connect() as db:
        rows = db.execute("SELECT role,content,sources,attachments FROM messages WHERE chat_id=? ORDER BY id",(chat_id,)).fetchall()
    return [{"role":r["role"],"content":r["content"],"sources":json.loads(r["sources"]),"attachments":json.loads(r["attachments"] or "[]")} for r in rows]

@app.post("/api/chats/{chat_id}/ask")
async def ask(chat_id: str, session_id: str = Form(...), question: str = Form(""), files: list[UploadFile] = File(default=[])):
    get_chat(chat_id,session_id)
    question = question.strip()
    if len(files) > MAX_FILES: raise HTTPException(400, f"Attach up to {MAX_FILES} files per message.")
    if not question and not files: raise HTTPException(400,"Enter a question or attach a file.")
    processed = [await process_upload(f,chat_id) for f in files]
    attachment_text = []
    for item in processed:
        if item["type"] == "image":
            attachment_text.append(f"Image: {item['name']}\nOCR-extracted text (may be empty):\n{item['text'] or '[No readable text detected]'}")
        else:
            attachment_text.append(f"Document: {item['name']}\nExtracted text:\n{item['text'] or '[No extractable text found]'}")
    user_content = question or "Please analyse the attached file(s)."
    if attachment_text:
        user_content += "\n\n[ATTACHED FILE CONTENT]\n" + "\n\n".join(attachment_text)
    metadata = [{"name":x["name"],"type":x["type"],"size":x["size"]} for x in processed]
    image_urls = [x["image_url"] for x in processed if x["image_url"]]
    with connect() as db:
        rows = db.execute("SELECT role,content FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT 12",(chat_id,)).fetchall()
        history = [dict(r) for r in reversed(rows)]
        db.execute("INSERT INTO messages(chat_id,role,content,attachments) VALUES(?,?,?,?)",(chat_id,"user",user_content,json.dumps(metadata,ensure_ascii=False)))
    try:
        with _llm_lock:
            answer,sources = ask_legal_chatbot(question or "Analyse the attached file(s).",history,search_legal_documents,llm,
                attachment_text="\n\n".join(attachment_text), image_urls=image_urls, vision_llm=vision_llm)
    except Exception as exc:
        raise HTTPException(503,f"AI service error: {type(exc).__name__}. Check Ollama and the selected models.")
    safe_sources = [{k:s.get(k) for k in ("reference","category","description","similarity")} for s in sources]
    with connect() as db:
        db.execute("INSERT INTO messages(chat_id,role,content,sources) VALUES(?,?,?,?)",(chat_id,"assistant",str(answer),json.dumps(safe_sources,ensure_ascii=False)))
        count=db.execute("SELECT COUNT(*) FROM messages WHERE chat_id=?",(chat_id,)).fetchone()[0]
        if count<=2:
            title=(question or (metadata[0]["name"] if metadata else "New chat"))[:55]
            db.execute("UPDATE chats SET title=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",(title,chat_id))
        else: db.execute("UPDATE chats SET updated_at=CURRENT_TIMESTAMP WHERE id=?",(chat_id,))
    return {"answer":str(answer),"sources":safe_sources,"attachments":metadata}

@app.post("/api/refresh")
def refresh_index():
    import importlib, search_rag
    try:
        subprocess.run([sys.executable,str(BASE/"Cleaned_data.py")],cwd=BASE,check=True)
        subprocess.run([sys.executable,str(BASE/"build_index.py")],cwd=BASE,check=True)
        importlib.reload(search_rag)
        return {"status":"ok","message":"Dataset and index rebuilt. Restart the server to reload the RAG module."}
    except subprocess.CalledProcessError as exc: raise HTTPException(500,f"Dataset refresh failed with exit code {exc.returncode}")
