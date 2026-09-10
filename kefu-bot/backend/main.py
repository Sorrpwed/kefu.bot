import json
import os
import re
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
FRONTEND_DIR = BASE_DIR / "frontend"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
ALLOWED_KNOWLEDGE_SUFFIXES = {".txt", ".pdf"}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.55"))
RETRIEVE_K = int(os.getenv("RAG_TOP_K", "4"))

NO_MATCH_REPLY = "这个问题我需要帮您转接人工客服哦，请稍等"

vectorstore = None


def load_knowledge_documents() -> list[Document]:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    documents: list[Document] = []
    for path in sorted(KNOWLEDGE_DIR.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".txt":
                loader = TextLoader(str(path), encoding="utf-8")
                documents.extend(loader.load())
            elif suffix == ".pdf":
                loader = PyPDFLoader(str(path))
                documents.extend(loader.load())
        except Exception as exc:
            print(f"[knowledge] skip {path.name}: {exc}")
    return documents


def build_vectorstore() -> Chroma | None:
    if not OPENAI_API_KEY:
        print("[rag] OPENAI_API_KEY is not set; RAG disabled until configured.")
        return None

    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        timeout=60,
        max_retries=2
    )

    raw_docs = load_knowledge_documents()
    if not raw_docs:
        print(f"[rag] no TXT/PDF found in {KNOWLEDGE_DIR}")
        return Chroma(
            collection_name="kefu_knowledge",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="kefu_knowledge",
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"[rag] indexed {len(chunks)} chunks from {len(raw_docs)} documents")
    return store


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global vectorstore
    vectorstore = build_vectorstore()
    yield


app = FastAPI(title="AI 客服机器人", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户问题")
    session_id: str | None = Field(default=None, description="可选会话 ID")


def _llm_kwargs() -> dict:
    kwargs: dict = {
        "api_key": OPENAI_API_KEY,
        "timeout": 60.0,
        "max_retries": 2,
    }
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return kwargs


def retrieve_context(question: str) -> list[Document]:
    if vectorstore is None:
        return []
    try:
        results = vectorstore.similarity_search_with_relevance_scores(
            question, k=RETRIEVE_K
        )
    except Exception:
        scored = vectorstore.similarity_search_with_score(question, k=RETRIEVE_K)
        results = [(doc, max(0.0, 1.0 - float(score))) for doc, score in scored]

    matched: list[Document] = []
    for doc, score in results:
        if score >= SCORE_THRESHOLD and (doc.page_content or "").strip():
            matched.append(doc)
    return matched


def build_messages(question: str, docs: list[Document]) -> list:
    context = "\n\n".join(
        f"[资料{i}]\n{doc.page_content}" for i, doc in enumerate(docs, start=1)
    )
    system = (
        "你是专业、友好的中文在线客服助手。"
        "只能根据下面提供的知识库资料回答用户问题。"
        "如果资料不足以回答，不要猜测，也不要编造。"
        "回答简洁、礼貌，使用中文。\n\n"
        f"知识库资料：\n{context}"
    )
    return [SystemMessage(content=system), HumanMessage(content=question)]


def sse_pack(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def stream_chat(question: str) -> AsyncGenerator[str, None]:
    docs = retrieve_context(question)
    if not docs:
        yield sse_pack({"content": NO_MATCH_REPLY, "handoff": True, "done": False})
        yield sse_pack({"done": True, "handoff": True})
        return

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        streaming=True,
        temperature=0.2,
        **_llm_kwargs(),
    )
    async for chunk in llm.astream(build_messages(question, docs)):
        text = chunk.content if isinstance(chunk.content, str) else ""
        if text:
            yield sse_pack({"content": text, "handoff": False, "done": False})
    yield sse_pack({"done": True, "handoff": False})


def _safe_filename(name: str) -> str:
    base = Path(name or "document").name
    cleaned = re.sub(r"[^\w\u4e00-\u9fff.\-]+", "_", base).strip("._")
    return cleaned or "document"


@app.get("/")
def index_page():
    page = FRONTEND_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="前端页面不存在")
    return FileResponse(page)


@app.post("/upload-knowledge")
async def upload_knowledge(files: list[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="请选择 TXT 或 PDF 文件")

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in ALLOWED_KNOWLEDGE_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {upload.filename}，仅支持 TXT/PDF"
            )
        dest = KNOWLEDGE_DIR / _safe_filename(upload.filename or f"doc{suffix}")
        content = await upload.read()
        dest.write_bytes(content)
        saved.append(dest.name)

    return {"saved": saved, "message": f"已保存 {len(saved)} 个文件，请调用 /reload-knowledge 刷新知识库"}


@app.post("/reload-knowledge")
async def reload_knowledge() -> dict:
    global vectorstore
    vectorstore = build_vectorstore()
    status = "ready" if vectorstore is not None else "empty"
    return {"status": status, "message": "知识库已重新加载"}


@app.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "model": CHAT_MODEL,
        "knowledge_dir": str(KNOWLEDGE_DIR),
        "has_api_key": bool(OPENAI_API_KEY),
        "vectorstore_ready": vectorstore is not None,
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        stream_chat(request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )