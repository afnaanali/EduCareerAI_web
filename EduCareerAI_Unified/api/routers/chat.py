import io
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from core.chatbot import (
    generate_llm_response,
    build_system_prompt,
)
from core.rag_engine import (
    process_uploaded_file,
    rebuild_index_from_chunks,
    search_relevant_chunks,
    format_doc_context_for_prompt,
    build_source_attribution,
)
from core.database import (
    save_chat_message,
    load_user_chat_history,
    clear_user_chat_history,
)

router = APIRouter(prefix="/api/chat", tags=["AI Assistant & RAG Studio"])

# In-memory document session cache (per session/user)
_SESSION_DOC_STORE: Dict[str, Dict[str, Any]] = {}


def _get_or_init_session(session_id: str) -> Dict[str, Any]:
    if session_id not in _SESSION_DOC_STORE:
        _SESSION_DOC_STORE[session_id] = {
            "chunks": [],
            "vectorizer": None,
            "tfidf_matrix": None,
            "filenames": [],
        }
    return _SESSION_DOC_STORE[session_id]


class ChatMessageRequest(BaseModel):
    session_id: Optional[str] = "default"
    user_id: Optional[int] = None
    message: str
    mode: Optional[str] = "💼 Career Assistant"
    resume_context: Optional[str] = None
    ats_score: Optional[int] = None
    career_recs: Optional[List[Dict[str, Any]]] = None
    course_recs: Optional[List[Dict[str, Any]]] = None
    history: Optional[List[Dict[str, str]]] = None


@router.post("/message")
def chat_message(payload: ChatMessageRequest):
    try:
        session = _get_or_init_session(payload.session_id or "default")
        
        # 1. RAG Search if documents are uploaded
        doc_context = ""
        citations = None
        if session["vectorizer"] is not None and session["tfidf_matrix"] is not None and session["chunks"]:
            relevant_chunks = search_relevant_chunks(
                query=payload.message,
                vectorizer=session["vectorizer"],
                tfidf_matrix=session["tfidf_matrix"],
                chunks=session["chunks"],
                top_k=3,
            )
            if relevant_chunks:
                doc_context = format_doc_context_for_prompt(relevant_chunks)
                citations = build_source_attribution(relevant_chunks)

        # 2. Build system prompt
        sys_prompt = build_system_prompt(
            mode=payload.mode or "💼 Career Assistant",
            resume_text=payload.resume_context,
            ats_score=payload.ats_score,
            career_recs=payload.career_recs,
            course_recs=payload.course_recs,
        )

        # 3. Format history and messages list for generator
        chat_hist = payload.history or []
        model_messages = [{"role": "system", "content": sys_prompt}]
        for m in chat_hist[-6:]:
            if isinstance(m, dict) and "content" in m:
                model_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

        if not chat_hist or chat_hist[-1].get("content") != payload.message:
            model_messages.append({"role": "user", "content": payload.message})

        # 4. Generate AI response
        ai_reply = generate_llm_response(
            messages=model_messages,
            user_query=payload.message,
            chat_history=chat_hist,
            doc_context_text=doc_context,
            system_prompt=sys_prompt,
            max_tokens=800,
            temperature=0.7,
        )

        # 5. Save to database if user_id is provided
        if payload.user_id:
            try:
                save_chat_message(payload.user_id, "user", payload.message)
                save_chat_message(payload.user_id, "assistant", ai_reply)
            except Exception as db_err:
                print(f"[Warning] Failed to save chat history to DB: {db_err}")

        return {
            "success": True,
            "response": ai_reply,
            "citations": citations,
            "has_rag_context": bool(doc_context),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/upload")
async def upload_rag_document(
    file: UploadFile = File(...),
    session_id: str = Form("default"),
):
    session = _get_or_init_session(session_id)
    content = await file.read()
    filename = file.filename or "uploaded_doc"

    try:
        file_obj = io.BytesIO(content)
        new_chunks, status_msg = process_uploaded_file(file_obj, filename)
        if not new_chunks:
            raise HTTPException(status_code=400, detail=status_msg)

        # Merge chunks
        session["chunks"].extend(new_chunks)
        if filename not in session["filenames"]:
            session["filenames"].append(filename)

        # Rebuild index
        vec, mat = rebuild_index_from_chunks(session["chunks"])
        session["vectorizer"] = vec
        session["tfidf_matrix"] = mat

        return {
            "success": True,
            "filename": filename,
            "chunks_added": len(new_chunks),
            "total_chunks": len(session["chunks"]),
            "files": session["filenames"],
            "message": status_msg,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process document: {str(e)}")


@router.get("/rag/status/{session_id}")
def get_rag_status(session_id: str):
    session = _get_or_init_session(session_id)
    return {
        "files": session["filenames"],
        "total_chunks": len(session["chunks"]),
        "has_index": session["vectorizer"] is not None,
    }


@router.delete("/rag/clear/{session_id}")
def clear_rag(session_id: str):
    if session_id in _SESSION_DOC_STORE:
        _SESSION_DOC_STORE[session_id] = {
            "chunks": [],
            "vectorizer": None,
            "tfidf_matrix": None,
            "filenames": [],
        }
    return {"success": True, "message": "RAG session cleared"}


@router.get("/history/{user_id}")
def get_chat_history(user_id: int):
    history = load_user_chat_history(user_id)
    return {"user_id": user_id, "history": history}


@router.delete("/history/{user_id}")
def clear_chat_history_db(user_id: int):
    ok = clear_user_chat_history(user_id)
    return {"success": ok}
